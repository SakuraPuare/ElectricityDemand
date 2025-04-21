import dask.dataframe as dd
from loguru import logger
import pandas as pd


def _resample_group_pd(pdf: pd.DataFrame) -> pd.DataFrame:
    """Helper function to resample a single group (Pandas DataFrame)."""
    # This function should NOT reset the index here. Let apply handle the group key index.
    if pdf.empty:
        # Return structure expected *before* reset_index in the main function
        return pd.DataFrame({'y': pd.Series(dtype='float64')}, index=pd.DatetimeIndex([], dtype='datetime64[ns]', name='timestamp'))
    try:
        pdf_indexed = pdf.set_index('timestamp')
        pdf_resampled = pdf_indexed['y'].resample('1H').sum()
        # Ensure correct type even if empty
        pdf_resampled = pdf_resampled.astype('float64')
        return pdf_resampled  # Return Series with timestamp index
    except Exception as e:
        logger.error(
            f"Error resampling group for unique_id {pdf['unique_id'].iloc[0] if not pdf.empty else 'unknown'}: {e}")
        # Return empty structure matching meta on error
        return pd.Series([], dtype='float64', index=pd.DatetimeIndex([], dtype='datetime64[ns]', name='timestamp'))


def resample_demand_to_hourly(ddf_demand: dd.DataFrame) -> dd.DataFrame | None:
    """
    Resamples the demand Dask DataFrame to hourly frequency using groupby().apply().

    Aggregates the 'y' column (demand) using sum within each hour
    for each unique_id.

    Args:
        ddf_demand: Input Dask DataFrame with 'unique_id', 'timestamp', 'y'.
                    'timestamp' column must be datetime objects.

    Returns:
        Resampled Dask DataFrame with 'unique_id', 'timestamp' (hourly), 'y' (summed).
        Returns None if input is None or empty or on error.
    """
    if ddf_demand is None:
        logger.warning("Input demand DataFrame is None. Skipping resampling.")
        return None
    required_cols = {'unique_id', 'timestamp', 'y'}
    if not required_cols.issubset(ddf_demand.columns):
        logger.error(
            f"Input DataFrame missing required columns (need {required_cols}, have {ddf_demand.columns}). Cannot resample.")
        return None

    logger.info("开始将 Demand 数据重采样至小时频率 (使用 groupby.apply)...")
    logger.info(f"原始分区数：{ddf_demand.npartitions}")

    try:
        # 确保 timestamp 是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(ddf_demand['timestamp'].dtype):
            logger.warning(
                "Demand 'timestamp' column is not datetime type. Attempting conversion...")
            ddf_demand = ddf_demand.map_partitions(
                lambda pdf: pdf.assign(timestamp=pd.to_datetime(
                    pdf['timestamp'], errors='coerce')),
                meta={'unique_id': 'str',
                      'timestamp': 'datetime64[ns]', 'y': 'float64'}
            ).dropna(subset=['timestamp'])

        # --- Revised Meta Definition ---
        # Define meta reflecting the structure *after* apply but *before* reset_index.
        # Expect 'unique_id' and 'timestamp' in the index, and 'y' as the column.
        meta_apply = pd.DataFrame(
            {'y': pd.Series([], dtype='float64')},
            index=pd.MultiIndex.from_tuples(
                [], names=['unique_id', 'timestamp']  # Explicitly define expected MultiIndex
            )
        ).astype({'y': 'float64'})  # Ensure correct dtype for the column

        # Apply the resampling function to each group
        logger.info("开始分组和应用重采样函数...")
        # Pass the Series-returning function to apply
        # Dask should create a DataFrame with unique_id and timestamp in the index
        resampled_ddf = ddf_demand.groupby('unique_id', group_keys=True).apply(
            _resample_group_pd,
            meta=meta_apply  # Provide the meta with MultiIndex
        )
        logger.info("应用重采样函数完成。")

        # --- Simplified Index Handling ---
        # Now we strongly expect unique_id and timestamp in the index based on meta.
        # Directly reset the index.
        logger.info("重置索引以将 'unique_id' 和 'timestamp' 转换为列...")
        # Check if it's actually MultiIndex as expected
        if isinstance(resampled_ddf.index, pd.MultiIndex):
            resampled_ddf = resampled_ddf.reset_index()
            logger.info("重置 MultiIndex 完成。")
        else:
            logger.warning(
                f"预期得到 MultiIndex，但实际得到：{type(resampled_ddf.index)}. 尝试 reset_index()...")
            try:
                resampled_ddf = resampled_ddf.reset_index()
                logger.info("尝试重置索引完成。")
            except Exception as idx_e:
                logger.error(f"重置索引失败：{idx_e}")
                logger.info("重采样后的数据结构预览 (meta):")
                logger.info(f"\n{resampled_ddf._meta.head()}")
                return None

        # Final check for columns
        if not required_cols.issubset(resampled_ddf.columns):
            logger.error(
                f"重置索引后仍然缺少必需列。需要：{required_cols}, 实际：{resampled_ddf.columns}")
            logger.info("重采样后的数据结构预览 (meta):")
            logger.info(f"\n{resampled_ddf._meta.head()}")
            return None

        # 打印一些信息以供验证
        logger.info("重采样后的数据结构预览 (meta):")
        # Log the meta data head
        logger.info(f"\n{resampled_ddf._meta.head()}")
        logger.info(f"重采样后的分区数：{resampled_ddf.npartitions}")
        logger.info("Demand 数据小时重采样完成。")

        return resampled_ddf

    except Exception as e:
        logger.exception(f"重采样过程中发生错误：{e}")
        return None


def validate_resampling(ddf_resampled: dd.DataFrame | None, n_check: int = 5):
    """Validates the resampling by checking timestamp frequency."""
    if ddf_resampled is None:
        logger.warning("Resampled DataFrame is None. Skipping validation.")
        return

    logger.info("--- 开始验证重采样结果 ---")
    try:
        # Ensure required columns exist before proceeding
        required_cols = {'timestamp', 'unique_id', 'y'}
        if not required_cols.issubset(ddf_resampled.columns):
            logger.error(
                f"验证失败：重采样后的 DataFrame 缺少必需列。需要：{required_cols}, 实际：{ddf_resampled.columns}")
            logger.info("重采样后的数据结构预览 (meta):")
            logger.info(f"\n{ddf_resampled._meta.head()}")
            return

        # 检查时间戳是否都是整点小时
        logger.info(f"计算前 {n_check} 条记录的时间戳分钟数和秒数...")
        sample_df_pd = ddf_resampled[['timestamp', 'unique_id']].head(
            n_check, compute=True)
        if sample_df_pd.empty:
            logger.warning("无法获取样本时间戳进行验证 (可能数据为空？)")
            return

        all_hourly = all((ts.minute == 0 and ts.second == 0)
                         for ts in sample_df_pd['timestamp'])

        if all_hourly:
            logger.success("样本时间戳验证成功：所有检查的时间戳都是整点小时。")
        else:
            logger.warning("样本时间戳验证失败：部分时间戳不是整点小时。")
            logger.warning(f"前 {n_check} 个时间戳和 ID:\n{sample_df_pd}")

    except Exception as e:
        logger.error(f"验证重采样时出错：{e}")
    logger.info("--- 完成验证重采样结果 ---")
