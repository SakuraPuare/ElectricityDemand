import pandas as pd
# 移除 Dask 导入
# import dask.dataframe as dd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, TimestampType, DateType
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from pathlib import Path  # Use pathlib for paths
from tqdm import tqdm  # For timeseries sample progress

# 使用相对导入
from ..utils.eda_utils import plot_numerical_distribution, log_value_counts, save_plot, plot_categorical_distribution
# 移除 dask_compute_context


# Changed sample frac default
def analyze_weather_numerical(sdf_weather: DataFrame, columns_to_analyze=None, plot_sample_frac=0.05, plots_dir=None, random_state=42):
    """分析 Weather Spark DataFrame 中指定数值列的分布并记录/绘图。"""
    if sdf_weather is None:
        logger.warning("输入的 Weather Spark DataFrame 为空，跳过数值特征分析。")
        return
    plot = plots_dir is not None
    if plot:
        plots_dir = Path(plots_dir)  # Convert to Path object
        plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    else:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Weather 数值特征图。")

    if columns_to_analyze is None:
        # 根据数据集 README 扩展列
        columns_to_analyze = [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
            'precipitation', 'rain', 'snowfall', 'snow_depth', 'pressure_msl',
            'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
            'cloud_cover_high', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
            'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
            # Typo in original name 'tepmerature' kept if that's the column name
            'soil_tepmerature_0_to_7cm', 'soil_moisture_0_to_7cm',
            'direct_radiation', 'diffuse_radiation', 'sunshine_duration'
        ]
    logger.info(f"--- 开始分析 Weather 数值特征分布 (Spark) ---")
    logger.debug(f"分析的列: {', '.join(columns_to_analyze)}")

    # 1. 过滤出实际存在且为数值类型的列
    numerical_cols = []
    datetime_cols = []
    other_cols = []
    schema = sdf_weather.schema
    for col in columns_to_analyze:
        if col in schema.names:
            if isinstance(schema[col].dataType, NumericType):
                numerical_cols.append(col)
            # Check for time types
            elif isinstance(schema[col].dataType, (TimestampType, DateType)):
                datetime_cols.append(col)
            else:
                other_cols.append(col)
                logger.warning(
                    f"列 '{col}' 类型为 {schema[col].dataType}，将从数值分析中排除。")
        else:
            logger.warning(f"列 '{col}' 不在 Weather DataFrame 中，跳过。")

    if not numerical_cols:
        logger.warning("在指定列中未找到可分析的数值列。")
        return

    # 2. 计算描述性统计 (Spark)
    desc_stats_pd = None
    logger.info("使用 Spark 计算数值特征的描述性统计...")
    try:
        # Spark describe() 返回字符串，改用 agg()
        agg_exprs = [F.count(F.col(c)).alias(
            f"{c}_count") for c in numerical_cols]
        agg_exprs.extend([F.mean(F.col(c)).alias(
            f"{c}_mean") for c in numerical_cols])
        agg_exprs.extend([F.stddev(F.col(c)).alias(
            f"{c}_stddev") for c in numerical_cols])
        agg_exprs.extend([F.min(F.col(c)).alias(
            f"{c}_min") for c in numerical_cols])
        agg_exprs.extend([F.expr(f"percentile_approx({c}, 0.25)").alias(
            f"{c}_25%") for c in numerical_cols])
        agg_exprs.extend([F.expr(f"percentile_approx({c}, 0.5)").alias(
            f"{c}_50%") for c in numerical_cols])
        agg_exprs.extend([F.expr(f"percentile_approx({c}, 0.75)").alias(
            f"{c}_75%") for c in numerical_cols])
        agg_exprs.extend([F.max(F.col(c)).alias(
            f"{c}_max") for c in numerical_cols])

        stats_row = sdf_weather.agg(*agg_exprs).first()

        if stats_row:
            # 将 Row 结果重新组织为类似 describe() 的 Pandas DataFrame
            stats_dict = stats_row.asDict()
            desc_data = {}
            for c in numerical_cols:
                desc_data[c] = {
                    "count": stats_dict.get(f"{c}_count"),
                    "mean": stats_dict.get(f"{c}_mean"),
                    "stddev": stats_dict.get(f"{c}_stddev"),
                    "min": stats_dict.get(f"{c}_min"),
                    "25%": stats_dict.get(f"{c}_25%"),
                    "50%": stats_dict.get(f"{c}_50%"),
                    "75%": stats_dict.get(f"{c}_75%"),
                    "max": stats_dict.get(f"{c}_max"),
                }
            desc_stats_pd = pd.DataFrame(desc_data)
            logger.info(f"Spark 计算的描述性统计:\n{desc_stats_pd.to_string()}")
        else:
            logger.error("未能计算描述性统计。")

    except Exception as e:
        logger.exception(f"使用 Spark 计算描述性统计时出错: {e}")

    # 3. 检查负值 (Spark)
    precipitation_cols = ['precipitation', 'rain', 'snowfall', 'snow_depth', 'et0_fao_evapotranspiration',
                          'sunshine_duration', 'direct_radiation', 'diffuse_radiation']  # 这些理论上不应为负
    neg_check_cols = [
        col for col in precipitation_cols if col in numerical_cols]  # 只检查存在的数值列
    if neg_check_cols:
        logger.info(f"使用 Spark 检查列 {neg_check_cols} 中的负值...")
        try:
            for col in neg_check_cols:
                negative_count = sdf_weather.filter(
                    F.col(col) < 0).count()  # Action
                if negative_count > 0:
                    logger.warning(
                        f"列 '{col}' 检测到 {negative_count:,} 个负值！请检查数据源或处理。")
                else:
                    logger.info(f"列 '{col}' 未检测到负值。")
        except Exception as e:
            logger.exception(f"使用 Spark 检查负值时出错: {e}")

    # 4. 绘图 (Spark 抽样 -> Pandas 绘图)
    if plot and numerical_cols:
        logger.info(f"开始绘制 Weather 特征分布图 (抽样比例: {plot_sample_frac:.1%})...")
        for col in numerical_cols:
            logger.info(f"绘制列: {col}")
            try:
                # Spark 抽样
                logger.debug(f"对列 '{col}' 进行 Spark 抽样...")
                # 先过滤 NaN/Null，然后抽样
                col_sample_sdf = sdf_weather.select(col).dropna().sample(
                    False, plot_sample_frac, seed=random_state)

                # 收集到 Pandas
                logger.debug(f"将列 '{col}' 的抽样结果收集到 Pandas...")
                col_sample_pd = col_sample_sdf.toPandas()[
                    col]  # 获取 Pandas Series
                logger.debug(f"列 '{col}' 抽样完成，样本大小: {len(col_sample_pd):,}")

                if col_sample_pd is None or col_sample_pd.empty:
                    logger.warning(f"列 '{col}' 的抽样结果为空或收集失败，跳过绘图。")
                    continue

                # 使用辅助函数绘图 (接收 Pandas Series)
                plot_numerical_distribution(col_sample_pd, col,
                                            f'weather_distribution_{col}', plots_dir,
                                            title_prefix="Weather ", kde=True,  # Use KDE for weather features
                                            showfliers=False)  # Hide outliers for potentially wide ranges
            except Exception as e:
                logger.exception(f"绘制列 '{col}' 的分布图时出错: {e}")

    logger.info("Weather 数值特征分析完成。")


def analyze_weather_categorical(sdf_weather: DataFrame, columns_to_analyze=None, top_n=20, plots_dir=None):
    """分析 Weather Spark DataFrame 中指定分类列的分布并记录/绘图。"""
    if sdf_weather is None:
        logger.warning("输入的 Weather Spark DataFrame 为空，跳过分类特征分析。")
        return
    plot = plots_dir is not None
    if plot:
        plots_dir = Path(plots_dir)  # Convert to Path object
        plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    else:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Weather 分类特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['weather_code', 'is_day']  # is_day 也是分类的
    logger.info(
        f"--- 开始分析 Weather 分类特征分布 (Spark) ({', '.join(columns_to_analyze)}) ---")

    # 过滤出实际存在的列 (不一定是分类类型，但可以尝试计数)
    relevant_cols = [
        col for col in columns_to_analyze if col in sdf_weather.columns]

    for col in relevant_cols:
        logger.info(f"--- 分析列: {col} ---")
        try:
            logger.info(f"使用 Spark 计算列 '{col}' 的值分布...")
            # Spark 实现 value_counts: groupBy + count + orderBy
            # .limit(top_n + 5) # 限制返回数量，避免收集过多不同值
            counts_sdf = sdf_weather.groupBy(
                col).count().orderBy(F.desc("count"))

            # 收集到 Pandas DataFrame (只收集 Top N + buffer)
            logger.debug(f"将列 '{col}' 的 Top {top_n} 值分布结果收集到 Pandas...")
            # Use take instead of toPandas if counts_sdf might be huge
            # Take more than top_n to see if there's a long tail
            counts_list = counts_sdf.take(top_n + 5)
            if not counts_list:
                logger.warning(f"未能计算或计算结果为空，跳过记录/绘制列 '{col}' 的值分布。")
                continue

            # Convert list of Rows to Pandas DataFrame
            counts_pd = pd.DataFrame(counts_list, columns=[col, 'count'])

            # 记录日志 (log_value_counts 接收 Pandas DF)
            # 需要确保列名是 'value' 和 'count' 或传递正确的列名
            # 重命名 Pandas DF 列以匹配 log_value_counts 预期 (或修改 log_value_counts)
            counts_pd_renamed = counts_pd.rename(
                columns={col: "value", "count": "计数"})
            # top_n controls logging limit
            log_value_counts(counts_pd_renamed, col,
                             top_n=top_n, is_already_counts=True)

            # 绘图 (plot_categorical_distribution 接收 Pandas Series 或 DF)
            if plot:
                logger.info(f"绘制列: {col}")
                # 将 Pandas DF 转换为 Series (value 作为 index, count 作为 values) 以便绘图
                # Only plot the actual top N
                counts_series_top_n = counts_pd.set_index(
                    col)['count'].head(top_n)
                if counts_series_top_n.empty:
                    logger.warning(f"列 '{col}' 的 Top {top_n} 计数为空，跳过绘图。")
                    continue

                plot_categorical_distribution(
                    counts_series_top_n,  # 传递 Top N Series
                    col,
                    f"weather_distribution_{col}",
                    plots_dir,
                    top_n=top_n,  # top_n controls plot limit
                    title_prefix="Weather "
                )

        except Exception as e:
            logger.exception(f"分析列 '{col}' 时出错: {e}")

    logger.info("Weather 分类特征分析完成。")


# --- 新增: analyze_weather_timeseries_sample ---
def analyze_weather_timeseries_sample(sdf_weather: DataFrame, n_samples=5, plots_dir=None, random_state=42):
    """
    使用 Spark 抽取 n_samples 个 location_id 的天气时间序列数据到 Pandas，
    然后分析时间戳频率。
    """
    logger.info(
        f"--- Starting Spark analysis of Weather time series frequency (sampling {n_samples} location_id) ---")
    if plots_dir is None:  # Plots dir might not be needed if only analyzing frequency
        logger.warning(
            "plots_dir not provided, frequency analysis results will only be logged.")
    required_cols = ['location_id', 'timestamp']
    if not all(col in sdf_weather.columns for col in required_cols):
        logger.error(
            f"Input Weather Spark DataFrame missing required columns ({', '.join(required_cols)}). Cannot analyze frequency.")
        return

    pdf_sample = None
    try:
        # 1. Spark 获取并抽样 location_id
        logger.info("使用 Spark 获取所有 distinct location_ids...")
        all_location_ids_sdf = sdf_weather.select('location_id').distinct()
        num_distinct_ids = all_location_ids_sdf.count()

        if num_distinct_ids == 0:
            logger.warning("Weather 数据中没有发现 location_id。")
            return

        if n_samples <= 0:
            logger.warning(
                f"Requested sample size ({n_samples}) is not positive, skipping weather frequency analysis.")
            return

        actual_n_samples = min(n_samples, num_distinct_ids)
        if num_distinct_ids < n_samples:
            logger.warning(
                f"Total location_id count ({num_distinct_ids}) is less than requested sample size ({n_samples}), using all {num_distinct_ids} location_ids.")

        logger.info(
            f"Randomly sampling {actual_n_samples} location_ids from {num_distinct_ids:,} using Spark RDD takeSample...")
        sampled_ids_rows = all_location_ids_sdf.rdd.takeSample(
            False, actual_n_samples, seed=random_state)
        sampled_ids = [row.location_id for row in sampled_ids_rows]
        if not sampled_ids:
            logger.error("未能成功抽样 location_ids for weather frequency analysis.")
            return
        logger.info(f"Selected location_ids (first 5): {sampled_ids[:5]}")

        # 2. Spark 过滤数据
        logger.info("使用 Spark 过滤 Weather 数据以获取样本 ID 的时间序列...")
        sampled_ids_sdf = sdf_weather.sql_ctx.createDataFrame(
            [(id,) for id in sampled_ids], ['location_id'])
        sdf_sample_filtered = sdf_weather.join(F.broadcast(
            sampled_ids_sdf), on='location_id', how='inner')

        # 3. 收集到 Pandas
        logger.info("将过滤后的 Weather 样本数据收集到 Pandas DataFrame...")
        try:
            pdf_sample = sdf_sample_filtered.select(required_cols).toPandas()
            if pdf_sample is None or pdf_sample.empty:
                logger.error("收集 Weather 样本数据到 Pandas 失败或结果为空。")
                return
            logger.info(
                f"Weather Pandas DataFrame created from Spark sample, containing {len(pdf_sample):,} rows for {len(sampled_ids)} location_ids.")
        except Exception as collect_e:
            logger.exception(f"收集 Spark Weather 样本数据到 Pandas 时出错: {collect_e}")
            # Add Arrow fallback similar to demand analysis if needed
            return

        # 4. 确保 Pandas DataFrame 按 ID 和时间排序
        logger.info(
            "Sorting Weather Pandas DataFrame by location_id and timestamp...")
        pdf_sample = pdf_sample.sort_values(['location_id', 'timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(pdf_sample['timestamp']):
            logger.info(
                "Converting timestamp column to datetime objects in Pandas...")
            pdf_sample['timestamp'] = pd.to_datetime(
                pdf_sample['timestamp'], errors='coerce')
            if pdf_sample['timestamp'].isnull().any():
                logger.warning(
                    "Some timestamp values failed to convert to datetime in Pandas for weather.")
                pdf_sample = pdf_sample.dropna(subset=['timestamp'])

        # 5. 在 Pandas 上进行频率分析
        logger.info(
            "Starting to analyze weather timestamp frequency for each sample (in Pandas)...")
        grouped_data = pdf_sample.groupby('location_id')

        for location_id, df_id in tqdm(grouped_data, desc="Analyzing weather frequency (Pandas)"):
            if df_id.empty:
                logger.warning(
                    f"No weather data found for location_id '{location_id}' in the Pandas sample, skipping frequency analysis.")
                continue

            try:
                time_diffs = df_id['timestamp'].diff().dropna()
                if not time_diffs.empty:
                    freq_counts = time_diffs.astype(str).value_counts()
                    if not freq_counts.empty:
                        logger.info(
                            f"--- location_id: {location_id} Weather Timestamp Interval Frequency (Pandas Sample) ---")
                        log_str = f"Frequency Stats (Top 5):\n{freq_counts.head().to_string()}"
                        if len(freq_counts) > 5:
                            log_str += "\n..."
                        logger.info(log_str)
                        if len(freq_counts) > 1:
                            logger.warning(
                                f" location_id '{location_id}' has multiple time intervals detected in weather sample.")
                    else:
                        logger.info(
                            f" location_id '{location_id}' has only one unique time interval after diff/dropna in weather sample.")
                else:
                    logger.info(
                        f" location_id '{location_id}' has less than two timestamps in weather sample, cannot calculate intervals.")
            except Exception as freq_e:
                logger.exception(
                    f"Error analyzing weather frequency for location_id '{location_id}' in sample: {freq_e}")

        logger.info(
            "Weather time series frequency sample analysis (Spark filter -> Pandas analyze) complete.")

    except Exception as e:
        logger.exception(
            f"Error analyzing Weather time series frequency samples with Spark: {e}")
