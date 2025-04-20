import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

# 使用相对导入
from ..utils.eda_utils import dask_compute_context, plot_numerical_distribution, save_plot

def analyze_demand_y_distribution(ddf_demand, sample_frac=0.005, random_state=42):
    """分析 Demand 数据 'y' 列的分布 (基于抽样)。"""
    logger.info(f"--- 开始分析 Demand 'y' 列分布 (抽样比例: {sample_frac:.1%}) ---")
    y_sample_pd = None
    try:
        with dask_compute_context(ddf_demand['y']) as persisted_data:
            y_col = persisted_data[0] # 获取持久化的 'y' 列
            logger.info("对 'y' 列进行抽样...")
            # 确保先dropna再抽样
            y_sample = y_col.dropna().sample(frac=sample_frac, random_state=random_state)

            # 检查抽样结果是否为空
            if not isinstance(y_sample, (dd.Series, pd.Series)) or y_sample.npartitions == 0:
                 logger.warning("抽样结果为空或无效，无法进行分布分析。")
                 return None

            with dask_compute_context(y_sample) as persisted_sample:
                y_sample_persisted = persisted_sample[0]
                # 在计算之前检查分区数
                if y_sample_persisted.npartitions == 0:
                    logger.warning("持久化后的抽样结果分区数为0，无法计算。")
                    return None

                logger.info("计算抽样数据的长度...")
                num_samples = len(y_sample_persisted) # len 在持久化后更快
                logger.info(f"抽样完成，得到 {num_samples:,} 个非空样本。")

                if num_samples == 0:
                    logger.warning("抽样结果长度为 0，无法进行分布分析。")
                    return None

                logger.info("计算抽样数据的描述性统计信息...")
                # Handle potential compute errors or empty results
                try:
                    desc_stats = y_sample_persisted.describe().compute()
                    if desc_stats is None or desc_stats.empty:
                        logger.warning("计算描述性统计信息失败或结果为空。")
                        return None
                    logger.info(f"'y' 列 (抽样) 描述性统计:\n{desc_stats.to_string()}")
                except Exception as desc_e:
                    logger.exception(f"计算描述性统计信息时出错: {desc_e}")
                    return None


                logger.info("检查抽样数据中的非正值 (<= 0)...")
                try:
                    non_positive_count = (y_sample_persisted <= 0).sum().compute()
                    if non_positive_count is None:
                         logger.warning("计算非正值数量失败。")
                         non_positive_count = 0 # Assume zero if compute fails
                    non_positive_perc = (non_positive_count / num_samples) * 100 if num_samples > 0 else 0
                    logger.info(f"抽样数据中 'y' <= 0 的数量: {non_positive_count:,} ({non_positive_perc:.2f}%)")
                except Exception as nonpos_e:
                    logger.exception(f"检查非正值时出错: {nonpos_e}")
                    # Proceed without non-positive info if it fails

                logger.info("将抽样结果转换为 Pandas Series...")
                # Handle potential compute errors or empty results
                try:
                    y_sample_pd = y_sample_persisted.compute()
                    if y_sample_pd is None or y_sample_pd.empty:
                        logger.warning("将抽样结果转换为 Pandas Series 失败或结果为空。")
                        return None
                    logger.info("转换为 Pandas Series 完成。")
                    return y_sample_pd
                except Exception as compute_e:
                    logger.exception(f"将抽样结果转换为 Pandas Series 时出错: {compute_e}")
                    return None

    except Exception as e:
        logger.exception(f"分析 Demand 'y' 列分布时发生错误: {e}")
        return None


def plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000, random_state=42):
    """绘制 Demand 'y' 列 (抽样) 的分布图并保存。"""
    if y_sample_pd is None or y_sample_pd.empty:
        logger.warning("Input 'y' sample is empty, skipping plotting.")
        return

    # --- Further sampling for plotting ---
    if len(y_sample_pd) > plot_sample_size:
        logger.info(f"Original sample size {len(y_sample_pd):,} is large, further sampling {plot_sample_size:,} points for plotting.")
        y_plot_sample = y_sample_pd.sample(n=plot_sample_size, random_state=random_state)
    else:
        y_plot_sample = y_sample_pd
    logger.info(f"--- Starting plotting for Demand 'y' distribution (Plot sample size: {len(y_plot_sample):,}) ---")

    # --- Plot original scale ---
    plot_numerical_distribution(y_plot_sample, 'y',
                                'demand_y_distribution_original_scale', plots_dir,
                                title_prefix="Demand ", kde=False, showfliers=False)

    # --- Plot log scale ---
    epsilon = 1e-6
    # Ensure we only select non-negative values before transformation
    y_plot_sample_non_neg = y_plot_sample[y_plot_sample >= 0].copy() # Use .copy() to avoid SettingWithCopyWarning

    if y_plot_sample_non_neg.empty:
        logger.warning("No non-negative values in plot sample, skipping log scale plot.")
        return

    try:
        # Apply log1p transformation safely
        y_log_transformed = np.log1p(y_plot_sample_non_neg + epsilon)

        # Check for NaN/inf in transformed data
        if y_log_transformed.isnull().all() or not np.isfinite(y_log_transformed).any():
             logger.warning("Log transformation resulted in all NaN or infinite values, skipping log plot.")
             return

        plot_numerical_distribution(y_log_transformed, 'log1p(y + epsilon)',
                                    'demand_y_distribution_log1p_scale', plots_dir,
                                    title_prefix="Demand ", kde=True, showfliers=True)
    except Exception as log_plot_e:
         logger.exception(f"Error plotting log-transformed 'y' distribution: {log_plot_e}")


    logger.info("Demand 'y' distribution plotting complete.")


def analyze_demand_timeseries_sample(ddf_demand, n_samples=5, plots_dir=None, random_state=42):
    """抽取 n_samples 个 unique_id 的完整时间序列数据，绘制图形并分析频率。"""
    logger.info(f"--- Starting analysis of Demand time series characteristics (sampling {n_samples} unique_id) ---")
    if plots_dir is None:
        logger.error("plots_dir not provided, cannot save time series plots.")
        return

    pdf_sample = None # Initialize
    try:
        # 1. Get and sample unique_id
        logger.info("Getting all unique_ids...")
        with dask_compute_context(ddf_demand['unique_id'].unique()) as persisted_ids:
            # Check if computation was successful
            if not persisted_ids:
                logger.error("Failed to persist unique_ids.")
                return
            try:
                all_unique_ids = persisted_ids[0].compute() # Compute to get Pandas Series
                if all_unique_ids is None or all_unique_ids.empty:
                     logger.error("Failed to compute unique_ids or result is empty.")
                     return
            except Exception as compute_id_e:
                logger.exception(f"Error computing unique IDs: {compute_id_e}")
                return


        if len(all_unique_ids) < n_samples:
            logger.warning(f"Total unique_id count ({len(all_unique_ids)}) is less than requested sample size ({n_samples}), using all unique_ids.")
            sampled_ids = all_unique_ids.tolist()
            n_samples = len(sampled_ids)
        else:
            logger.info(f"Randomly sampling {n_samples} unique_ids from {len(all_unique_ids):,}...")
            np.random.seed(random_state)
            sampled_ids = np.random.choice(all_unique_ids, n_samples, replace=False).tolist()
        logger.info(f"Selected unique_ids: {sampled_ids}")

        # 2. Filter and compute Pandas DataFrame
        logger.info("Filtering Dask DataFrame to get sample data...")
        ddf_sample_filtered = ddf_demand[ddf_demand['unique_id'].isin(sampled_ids)]

        logger.info("Converting sample data to Pandas DataFrame (might take time and memory)...")
        with dask_compute_context(ddf_sample_filtered) as persisted_sample:
             if not persisted_sample:
                 logger.error("Failed to persist filtered sample data.")
                 return
             try:
                 pdf_sample = persisted_sample[0].compute()
                 if pdf_sample is None or pdf_sample.empty:
                     logger.error("Failed to compute sample Pandas DataFrame or result is empty.")
                     return
             except Exception as compute_sample_e:
                  logger.exception(f"Error computing sample Pandas DataFrame: {compute_sample_e}")
                  return
        logger.info(f"Pandas DataFrame created, containing {len(pdf_sample):,} rows.")

        pdf_sample = pdf_sample.sort_values(['unique_id', 'timestamp'])

        # 3. Plot and analyze
        logger.info("Starting to plot time series for each sample and analyze frequency...")
        plt.style.use('seaborn-v0_8-whitegrid')

        for unique_id in tqdm(sampled_ids, desc="Processing samples"):
            df_id = pdf_sample[pdf_sample['unique_id'] == unique_id]
            if df_id.empty:
                logger.warning(f"No data found for unique_id '{unique_id}', skipping.")
                continue

            # Plot
            try:
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(df_id['timestamp'], df_id['y'], label=f'ID: {unique_id}')
                ax.set_title(f'Demand (y) Time Series - unique_id: {unique_id}')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('Electricity Demand (y) in kWh')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                save_plot(fig, f'timeseries_sample_{unique_id}.png', plots_dir) # Use helper
            except Exception as plot_e:
                 logger.exception(f"Error plotting time series for unique_id '{unique_id}': {plot_e}")
                 plt.close(fig) # Ensure plot is closed even on error


            # Analyze frequency
            try:
                 # Convert to Pandas Timestamp if not already
                 if not pd.api.types.is_datetime64_any_dtype(df_id['timestamp']):
                     df_id['timestamp'] = pd.to_datetime(df_id['timestamp'])

                 # Ensure it's sorted by timestamp before diff
                 df_id = df_id.sort_values('timestamp')
                 time_diffs = df_id['timestamp'].diff()

                 if len(time_diffs) > 1:
                     # Drop the first NA value from diff()
                     freq_counts = time_diffs.dropna().value_counts()
                     if not freq_counts.empty:
                         logger.info(f"--- unique_id: {unique_id} Timestamp Interval Frequency ---")
                         log_str = f"Frequency Stats (Top 5):\n{freq_counts.head().to_string()}"
                         if len(freq_counts) > 5: log_str += "\n..."
                         logger.info(log_str)
                         if len(freq_counts) > 1:
                             logger.warning(f" unique_id '{unique_id}' has multiple time intervals detected, possible missing data or frequency change.")
                     else:
                         logger.info(f" unique_id '{unique_id}' has only one timestamp after diff/dropna, cannot calculate frequency.")
                 else:
                     logger.info(f" unique_id '{unique_id}' has only one or zero timestamps, cannot calculate frequency.")
            except Exception as freq_e:
                logger.exception(f"Error analyzing frequency for unique_id '{unique_id}': {freq_e}")


        logger.info("Time series sample analysis complete.")

    except Exception as e:
        logger.exception(f"Error analyzing Demand time series samples: {e}")
    # No explicit cleanup needed due to context manager 