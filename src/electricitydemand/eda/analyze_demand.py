import numpy as np
import pandas as pd
# 移除 Dask 相关导入
# import dask.dataframe as dd
# 引入 Spark 相关库
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType # 用于检查数值类型
import matplotlib.pyplot as plt
from tqdm import tqdm # tqdm 仍然可以在 Pandas 部分使用
from loguru import logger

# 使用相对导入 (utils 保持不变，假设绘图函数仍用 Pandas)
from ..utils.eda_utils import plot_numerical_distribution, save_plot # 移除 dask_compute_context

# --- 迁移后的 analyze_demand_y_distribution ---
def analyze_demand_y_distribution(sdf_demand: DataFrame, sample_frac=0.005, random_state=42):
    """使用 Spark 分析 Demand 数据 'y' 列的分布 (基于 Spark 计算和抽样)。"""
    logger.info(f"--- 开始使用 Spark 分析 Demand 'y' 列分布 (抽样比例: {sample_frac:.1%}) ---")
    if 'y' not in sdf_demand.columns:
        logger.error("输入 Spark DataFrame 缺少 'y' 列。")
        return None
    # 确保 'y' 是数值类型
    if not isinstance(sdf_demand.schema['y'].dataType, NumericType):
         logger.error(f"'y' 列不是数值类型 ({sdf_demand.schema['y'].dataType})。无法进行分析。")
         return None

    y_sample_pd = None
    try:
        # --- 1. Spark 计算描述性统计 ---
        logger.info("使用 Spark 计算 'y' 列的描述性统计...")
        desc_stats_spark = sdf_demand.select('y').summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").toPandas()
        # 格式化输出
        desc_stats_spark.set_index('summary', inplace=True)
        logger.info(f"'y' 列 (Spark 全量统计) 描述性统计:\n{desc_stats_spark.to_string()}")

        # 获取总行数 (用于计算百分比)
        total_count = sdf_demand.count()
        if total_count == 0:
             logger.warning("DataFrame 为空，无法进行进一步分析。")
             return None

        # --- 2. Spark 计算非正值 ---
        logger.info("使用 Spark 检查 'y' 列中的非正值 (<= 0)...")
        non_positive_count = sdf_demand.filter(F.col('y') <= 0).count()
        non_positive_perc = (non_positive_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"全量数据中 'y' <= 0 的数量: {non_positive_count:,} ({non_positive_perc:.2f}%)")

        # --- 3. Spark 抽样并收集到 Pandas ---
        logger.info(f"对 'y' 列进行 Spark 抽样 (比例: {sample_frac:.1%})...")
        # 先过滤掉 NaN/Null 值再抽样，以匹配 Dask 版本逻辑
        y_col_sdf = sdf_demand.select('y').dropna()
        # 使用 sample 方法
        # withReplacement=False 表示不放回抽样
        y_sample_sdf = y_col_sdf.sample(withReplacement=False, fraction=sample_frac, seed=random_state)

        logger.info("将抽样结果收集到 Pandas Series...")
        # .rdd.flatMap(lambda x: x).collect() 是将单列 DataFrame 转换为 list 的常用方法
        # y_sample_list = y_sample_sdf.rdd.flatMap(lambda x: x).collect()
        # 或者更直接地使用 toPandas (对于单列抽样通常可行)
        y_sample_pd = y_sample_sdf.toPandas()['y'] # 转换为 Pandas Series

        if y_sample_pd is None or y_sample_pd.empty:
            logger.warning("Spark 抽样结果收集到 Pandas 后为空。")
            return None

        num_samples = len(y_sample_pd)
        logger.info(f"抽样并收集完成，得到 {num_samples:,} 个非空样本 (Pandas)。")

        # 可以在这里也计算一下 Pandas 样本的描述性统计，与 Spark 全量统计对比
        if num_samples > 0:
            logger.info("计算 Pandas 抽样数据的描述性统计信息...")
            desc_stats_pd = y_sample_pd.describe()
            logger.info(f"'y' 列 (Pandas 抽样) 描述性统计:\n{desc_stats_pd.to_string()}")

        return y_sample_pd

    except Exception as e:
        logger.exception(f"使用 Spark 分析 Demand 'y' 列分布时发生错误: {e}")
        return None

# --- plot_demand_y_distribution ---
# 此函数接收 Pandas Series，无需修改
def plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000, random_state=42):
    """绘制 Demand 'y' 列 (抽样得到的 Pandas Series) 的分布图并保存。"""
    if y_sample_pd is None or y_sample_pd.empty:
        logger.warning("Input 'y' sample (Pandas Series) is empty, skipping plotting.")
        return

    # --- Further sampling for plotting (logic remains the same) ---
    if len(y_sample_pd) > plot_sample_size:
        logger.info(f"Original sample size {len(y_sample_pd):,} is large, further sampling {plot_sample_size:,} points for plotting.")
        y_plot_sample = y_sample_pd.sample(n=plot_sample_size, random_state=random_state)
    else:
        y_plot_sample = y_sample_pd
    logger.info(f"--- Starting plotting for Demand 'y' distribution (Plot sample size: {len(y_plot_sample):,}) ---")

    # --- Plot original scale (logic remains the same) ---
    plot_numerical_distribution(y_plot_sample, 'y',
                                'demand_y_distribution_original_scale', plots_dir,
                                title_prefix="Demand ", kde=False, showfliers=False)

    # --- Plot log scale (logic remains the same) ---
    epsilon = 1e-6
    y_plot_sample_non_neg = y_plot_sample[y_plot_sample >= 0].copy()

    if y_plot_sample_non_neg.empty:
        logger.warning("No non-negative values in plot sample, skipping log scale plot.")
        return

    try:
        y_log_transformed = np.log1p(y_plot_sample_non_neg + epsilon)
        if y_log_transformed.isnull().all() or not np.isfinite(y_log_transformed).any():
             logger.warning("Log transformation resulted in all NaN or infinite values, skipping log plot.")
             return
        plot_numerical_distribution(y_log_transformed, 'log1p(y + epsilon)',
                                    'demand_y_distribution_log1p_scale', plots_dir,
                                    title_prefix="Demand ", kde=True, showfliers=True)
    except Exception as log_plot_e:
         logger.exception(f"Error plotting log-transformed 'y' distribution: {log_plot_e}")

    logger.info("Demand 'y' distribution plotting complete.")


# --- 迁移后的 analyze_demand_timeseries_sample ---
def analyze_demand_timeseries_sample(sdf_demand: DataFrame, n_samples=5, plots_dir=None, random_state=42):
    """使用 Spark 抽取 n_samples 个 unique_id 的完整时间序列数据到 Pandas，然后绘制图形并分析频率。"""
    logger.info(f"--- Starting Spark analysis of Demand time series (sampling {n_samples} unique_id) ---")
    if plots_dir is None:
        logger.error("plots_dir not provided, cannot save time series plots.")
        return
    if 'unique_id' not in sdf_demand.columns or 'timestamp' not in sdf_demand.columns or 'y' not in sdf_demand.columns:
         logger.error("Input Spark DataFrame missing required columns ('unique_id', 'timestamp', 'y').")
         return

    pdf_sample = None # Initialize
    try:
        # 1. Spark 获取并抽样 unique_id
        logger.info("使用 Spark 获取所有 unique_ids...")
        # distinct() 是一个转换操作，需要 action (如 .rdd 或 .collect) 触发
        all_unique_ids_sdf = sdf_demand.select('unique_id').distinct()
        # 使用 takeSample 获取随机样本 (action)
        # takeSample(withReplacement, num, [seed])
        num_distinct_ids = all_unique_ids_sdf.count() # 获取总 ID 数，以便调整 n_samples

        if num_distinct_ids == 0:
             logger.warning("数据中没有发现 unique_id。")
             return

        if num_distinct_ids < n_samples:
            logger.warning(f"Total unique_id count ({num_distinct_ids}) is less than requested sample size ({n_samples}), using all unique_ids.")
            sampled_ids_rows = all_unique_ids_sdf.collect() # Collect all IDs
            n_samples = num_distinct_ids
        else:
            logger.info(f"Randomly sampling {n_samples} unique_ids from {num_distinct_ids:,} using takeSample...")
            # takeSample returns a list of Row objects
            sampled_ids_rows = all_unique_ids_sdf.rdd.takeSample(False, n_samples, seed=random_state)

        # 从 Row 对象中提取 ID 字符串/值
        sampled_ids = [row.unique_id for row in sampled_ids_rows]
        logger.info(f"Selected unique_ids: {sampled_ids}")

        # 2. Spark 过滤数据
        logger.info("使用 Spark 过滤数据以获取样本 ID 的时间序列...")
        sdf_sample_filtered = sdf_demand.filter(F.col('unique_id').isin(sampled_ids))

        # 3. 收集到 Pandas
        logger.info("将过滤后的样本数据收集到 Pandas DataFrame (可能需要一些时间和内存)...")
        try:
            pdf_sample = sdf_sample_filtered.toPandas()
            if pdf_sample is None or pdf_sample.empty:
                logger.error("收集样本数据到 Pandas 失败或结果为空。")
                return
            logger.info(f"Pandas DataFrame created from Spark sample, containing {len(pdf_sample):,} rows.")
        except Exception as collect_e:
             logger.exception(f"收集 Spark 样本数据到 Pandas 时出错: {collect_e}")
             # 可以考虑增加重试或错误处理逻辑
             return

        # 确保 Pandas DataFrame 按 ID 和时间排序 (后续绘图和频率分析需要)
        pdf_sample = pdf_sample.sort_values(['unique_id', 'timestamp'])
        # 确保 timestamp 是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(pdf_sample['timestamp']):
            pdf_sample['timestamp'] = pd.to_datetime(pdf_sample['timestamp'])


        # 4. 在 Pandas 上进行绘图和频率分析 (这部分逻辑不变)
        logger.info("Starting to plot time series for each sample (in Pandas) and analyze frequency...")
        plt.style.use('seaborn-v0_8-whitegrid')

        for unique_id in tqdm(sampled_ids, desc="Processing samples (Pandas)"):
            df_id = pdf_sample[pdf_sample['unique_id'] == unique_id]
            if df_id.empty:
                logger.warning(f"No data found for unique_id '{unique_id}' in the Pandas sample, skipping.")
                continue

            # --- 绘图逻辑 (不变) ---
            try:
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(df_id['timestamp'], df_id['y'], label=f'ID: {unique_id}')
                ax.set_title(f'Demand (y) Time Series - unique_id: {unique_id}')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('Electricity Demand (y) in kWh')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                save_plot(fig, f'timeseries_sample_{unique_id}.png', plots_dir)
            except Exception as plot_e:
                 logger.exception(f"Error plotting time series for unique_id '{unique_id}': {plot_e}")
                 plt.close(fig) # Ensure plot is closed even on error

            # --- 频率分析逻辑 (不变) ---
            try:
                 # 确保已排序
                 time_diffs = df_id['timestamp'].diff()
                 if len(time_diffs) > 1:
                     freq_counts = time_diffs.dropna().value_counts()
                     if not freq_counts.empty:
                         logger.info(f"--- unique_id: {unique_id} Timestamp Interval Frequency (Pandas Sample) ---")
                         log_str = f"Frequency Stats (Top 5):\n{freq_counts.head().to_string()}"
                         if len(freq_counts) > 5: log_str += "\n..."
                         logger.info(log_str)
                         if len(freq_counts) > 1:
                             logger.warning(f" unique_id '{unique_id}' has multiple time intervals detected in sample.")
                     else:
                         logger.info(f" unique_id '{unique_id}' has only one timestamp after diff/dropna in sample.")
                 else:
                     logger.info(f" unique_id '{unique_id}' has only one or zero timestamps in sample.")
            except Exception as freq_e:
                logger.exception(f"Error analyzing frequency for unique_id '{unique_id}' in sample: {freq_e}")

        logger.info("Time series sample analysis (Spark filter -> Pandas plot/analyze) complete.")

    except Exception as e:
        logger.exception(f"Error analyzing Demand time series samples with Spark: {e}")

    # No explicit cleanup needed due to context manager 