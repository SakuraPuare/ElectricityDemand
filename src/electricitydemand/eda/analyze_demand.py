import numpy as np
import pandas as pd
# 移除 Dask 相关导入
# import dask.dataframe as dd
# 引入 Spark 相关库
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType  # 用于检查数值类型
import matplotlib.pyplot as plt
from tqdm import tqdm  # tqdm 仍然可以在 Pandas 部分使用
from loguru import logger

# 使用相对导入 (utils 保持不变，假设绘图函数仍用 Pandas)
# 移除 dask_compute_context
from ..utils.eda_utils import plot_numerical_distribution, save_plot

# --- 迁移后的 analyze_demand_y_distribution ---


def analyze_demand_y_distribution(sdf_demand: DataFrame, sample_frac=0.005, random_state=42, plots_dir=None):
    """使用 Spark 分析 Demand 数据 'y' 列的分布 (基于 Spark 计算和抽样)。"""
    logger.info(
        f"--- 开始使用 Spark 分析 Demand 'y' 列分布 (抽样比例: {sample_frac:.1%}) ---")
    if 'y' not in sdf_demand.columns:
        logger.error("输入 Spark DataFrame 缺少 'y' 列。")
        return None
    # 确保 'y' 是数值类型
    if not isinstance(sdf_demand.schema['y'].dataType, NumericType):
        logger.error(
            f"'y' 列不是数值类型 ({sdf_demand.schema['y'].dataType})。无法进行分析。")
        return None

    y_sample_pd = None
    try:
        # --- 1. Spark 计算描述性统计 ---
        logger.info("使用 Spark 计算 'y' 列的描述性统计...")
        # 使用 .agg() 获得更结构化的统计结果
        stats_agg = sdf_demand.agg(
            F.count(F.col('y')).alias("count"),
            F.mean(F.col('y')).alias("mean"),
            F.stddev(F.col('y')).alias("stddev"),
            F.min(F.col('y')).alias("min"),
            F.expr("percentile_approx(y, 0.25)").alias("25%"),
            F.expr("percentile_approx(y, 0.5)").alias("50%"),
            F.expr("percentile_approx(y, 0.75)").alias("75%"),
            F.max(F.col('y')).alias("max")
        ).first()  # .first() returns a Row object

        if stats_agg is None:
            logger.error("无法计算 'y' 列的描述性统计。")
            return None

        # 将 Row 转换为更易读的格式
        desc_stats_spark = pd.Series(stats_agg.asDict())
        logger.info(
            f"'y' 列 (Spark 全量统计) 描述性统计:\n{desc_stats_spark.to_string()}")

        # 获取总行数 (用于计算百分比)
        total_count = stats_agg['count']  # 从统计结果中获取 count
        if total_count == 0:
            logger.warning("DataFrame 为空，无法进行进一步分析。")
            return None

        # --- 2. Spark 计算非正值 ---
        logger.info("使用 Spark 检查 'y' 列中的非正值 (<= 0)...")
        # 包含 NaN 的情况也需要考虑，filter(y <= 0) 会自动忽略 null
        non_positive_count = sdf_demand.filter(F.col('y') <= 0).count()
        # 计算缺失值数量
        null_count = sdf_demand.filter(F.col('y').isNull()).count()
        logger.info(f"全量数据中 'y' 缺失的数量: {null_count:,}")
        non_positive_perc = (non_positive_count / (total_count - null_count)
                             ) * 100 if (total_count - null_count) > 0 else 0
        logger.info(
            f"全量数据中 'y' <= 0 的数量 (在非缺失值中): {non_positive_count:,} ({non_positive_perc:.2f}%)")

        # --- 3. Spark 抽样并收集到 Pandas (用于绘图) ---
        if plots_dir:
            logger.info(f"对 'y' 列进行 Spark 抽样 (比例: {sample_frac:.1%}) 以便绘图...")
            # 先过滤掉 NaN/Null 值再抽样
            y_col_sdf = sdf_demand.select('y').dropna()
            # 使用 sample 方法
            y_sample_sdf = y_col_sdf.sample(
                withReplacement=False, fraction=sample_frac, seed=random_state)

            logger.info("将抽样结果收集到 Pandas Series...")
            try:
                y_sample_pd = y_sample_sdf.toPandas()['y']  # 转换为 Pandas Series
                if y_sample_pd is None or y_sample_pd.empty:
                    logger.warning("Spark 抽样结果收集到 Pandas 后为空。")
                    y_sample_pd = None  # 确保是 None
                else:
                    num_samples = len(y_sample_pd)
                    logger.info(f"抽样并收集完成，得到 {num_samples:,} 个非空样本 (Pandas)。")
                    # 可以选择性地计算 Pandas 样本统计信息进行比较
                    # if num_samples > 0:
                    #     logger.info("计算 Pandas 抽样数据的描述性统计信息...")
                    #     desc_stats_pd = y_sample_pd.describe()
                    #     logger.info(f"'y' 列 (Pandas 抽样) 描述性统计:\n{desc_stats_pd.to_string()}")

                    # --- 4. 调用绘图函数 ---
                    plot_demand_y_distribution(
                        y_sample_pd, plots_dir, random_state=random_state)

            except Exception as collect_e:
                logger.exception(f"收集 Spark 抽样数据到 Pandas 或绘图时出错: {collect_e}")
                y_sample_pd = None  # 出错则返回 None
        else:
            logger.info("未提供 plots_dir，跳过 'y' 列分布的抽样和绘图。")

        # 函数现在主要执行分析和记录，绘图是可选的
        # 返回 Pandas 样本（如果生成了）或 None
        # return y_sample_pd # 或者可以不返回任何东西，因为分析结果已记录
        return  # 分析结果已记录，绘图已完成（如果提供了目录）

    except Exception as e:
        logger.exception(f"使用 Spark 分析 Demand 'y' 列分布时发生错误: {e}")
        # return None


# --- plot_demand_y_distribution ---
# 此函数接收 Pandas Series，无需修改
def plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000, random_state=42):
    """绘制 Demand 'y' 列 (抽样得到的 Pandas Series) 的分布图并保存。"""
    if y_sample_pd is None or y_sample_pd.empty:
        logger.warning(
            "Input 'y' sample (Pandas Series) is empty, skipping plotting.")
        return

    # --- Further sampling for plotting (logic remains the same) ---
    if len(y_sample_pd) > plot_sample_size:
        logger.info(
            f"Original sample size {len(y_sample_pd):,} is large, further sampling {plot_sample_size:,} points for plotting.")
        y_plot_sample = y_sample_pd.sample(
            n=plot_sample_size, random_state=random_state)
    else:
        y_plot_sample = y_sample_pd
    logger.info(
        f"--- Starting plotting for Demand 'y' distribution (Plot sample size: {len(y_plot_sample):,}) ---")

    # --- Plot original scale (logic remains the same) ---
    plot_numerical_distribution(y_plot_sample, 'y',
                                'demand_y_distribution_original_scale', plots_dir,
                                title_prefix="Demand ", kde=False, showfliers=False)

    # --- Plot log scale (logic remains the same) ---
    epsilon = 1e-6
    y_plot_sample_non_neg = y_plot_sample[y_plot_sample >= 0].copy()

    if y_plot_sample_non_neg.empty:
        logger.warning(
            "No non-negative values in plot sample, skipping log scale plot.")
        return

    try:
        # Ensure input is numeric before transformation
        y_plot_sample_non_neg = pd.to_numeric(
            y_plot_sample_non_neg, errors='coerce').dropna()
        if y_plot_sample_non_neg.empty:
            logger.warning(
                "No numeric non-negative values after coercion, skipping log plot.")
            return

        y_log_transformed = np.log1p(
            y_plot_sample_non_neg + epsilon)  # 使用 np.log1p
        if y_log_transformed.isnull().all() or not np.isfinite(y_log_transformed).any():
            logger.warning(
                "Log transformation resulted in all NaN or infinite values, skipping log plot.")
            return
        plot_numerical_distribution(y_log_transformed, 'log1p(y + epsilon)',
                                    'demand_y_distribution_log1p_scale', plots_dir,
                                    title_prefix="Demand ", kde=True, showfliers=True)
    except Exception as log_plot_e:
        logger.exception(
            f"Error plotting log-transformed 'y' distribution: {log_plot_e}")

    logger.info("Demand 'y' distribution plotting complete.")


# --- 迁移后的 analyze_demand_timeseries_sample ---
def analyze_demand_timeseries_sample(sdf_demand: DataFrame, n_samples=5, plots_dir=None, random_state=42):
    """使用 Spark 抽取 n_samples 个 unique_id 的完整时间序列数据到 Pandas，然后绘制图形并分析频率。"""
    logger.info(
        f"--- Starting Spark analysis of Demand time series (sampling {n_samples} unique_id) ---")
    if plots_dir is None:
        logger.error("plots_dir not provided, cannot save time series plots.")
        return
    required_cols = ['unique_id', 'timestamp', 'y']
    if not all(col in sdf_demand.columns for col in required_cols):
        logger.error(
            f"Input Spark DataFrame missing required columns ({', '.join(required_cols)}).")
        return

    pdf_sample = None  # Initialize
    try:
        # 1. Spark 获取并抽样 unique_id
        logger.info("使用 Spark 获取所有 distinct unique_ids...")
        # distinct() 是一个转换操作，需要 action (如 .rdd 或 .collect) 触发
        all_unique_ids_sdf = sdf_demand.select('unique_id').distinct()
        # 使用 .rdd.takeSample 获取随机样本 (action)
        # takeSample(withReplacement, num, [seed])
        num_distinct_ids = all_unique_ids_sdf.count()  # 获取总 ID 数，以便调整 n_samples

        if num_distinct_ids == 0:
            logger.warning("数据中没有发现 unique_id。")
            return

        if n_samples <= 0:
            logger.warning(
                f"Requested sample size ({n_samples}) is not positive, skipping.")
            return

        actual_n_samples = min(n_samples, num_distinct_ids)
        if num_distinct_ids < n_samples:
            logger.warning(
                f"Total unique_id count ({num_distinct_ids}) is less than requested sample size ({n_samples}), using all {num_distinct_ids} unique_ids.")

        logger.info(
            f"Randomly sampling {actual_n_samples} unique_ids from {num_distinct_ids:,} using Spark RDD takeSample...")
        # takeSample returns a list of Row objects
        sampled_ids_rows = all_unique_ids_sdf.rdd.takeSample(
            False, actual_n_samples, seed=random_state)

        # 从 Row 对象中提取 ID 字符串/值
        sampled_ids = [row.unique_id for row in sampled_ids_rows]
        if not sampled_ids:
            logger.error("未能成功抽样 unique_ids。")
            return
        logger.info(f"Selected unique_ids (first 5): {sampled_ids[:5]}")

        # 2. Spark 过滤数据
        logger.info("使用 Spark 过滤数据以获取样本 ID 的时间序列...")
        # 使用 join 比 filter(isin(...)) 通常在 Spark 中性能更好，特别是当 sampled_ids 列表很大时
        # 创建一个包含抽样 ID 的小 DataFrame
        sampled_ids_sdf = sdf_demand.sql_ctx.createDataFrame(
            [(id,) for id in sampled_ids], ['unique_id'])
        # 使用 broadcast join 提示，因为 sampled_ids_sdf 很小
        sdf_sample_filtered = sdf_demand.join(F.broadcast(
            sampled_ids_sdf), on='unique_id', how='inner')

        # 3. 收集到 Pandas (只收集过滤后的数据)
        logger.info("将过滤后的样本数据收集到 Pandas DataFrame (可能需要一些时间和内存)...")
        try:
            # 增加配置避免 Arrow 错误 (如果遇到)
            # sdf_demand.sql_ctx.setConf("spark.sql.execution.arrow.pyspark.enabled", "false")
            pdf_sample = sdf_sample_filtered.select(
                required_cols).toPandas()  # 只选择需要的列
            if pdf_sample is None or pdf_sample.empty:
                logger.error("收集样本数据到 Pandas 失败或结果为空。")
                return
            logger.info(
                f"Pandas DataFrame created from Spark sample, containing {len(pdf_sample):,} rows for {len(sampled_ids)} unique_ids.")
        except Exception as collect_e:
            logger.exception(
                f"收集 Spark 样本数据到 Pandas 时出错: {collect_e}. 尝试禁用 Arrow...")
            try:
                # 尝试禁用 Arrow 并重试
                sdf_demand.sql_ctx.setConf(
                    "spark.sql.execution.arrow.pyspark.enabled", "false")
                pdf_sample = sdf_sample_filtered.select(
                    required_cols).toPandas()
                if pdf_sample is None or pdf_sample.empty:
                    logger.error("禁用 Arrow 后收集样本数据到 Pandas 仍然失败或结果为空。")
                    return
                logger.info(
                    f"(Arrow 已禁用) Pandas DataFrame created, containing {len(pdf_sample):,} rows.")
            except Exception as collect_e_no_arrow:
                logger.exception(
                    f"禁用 Arrow 后收集 Spark 样本数据到 Pandas 时仍然出错: {collect_e_no_arrow}")
                return
            finally:
                # 恢复 Arrow 设置 (可选)
                sdf_demand.sql_ctx.setConf(
                    "spark.sql.execution.arrow.pyspark.enabled", "true")

        # 确保 Pandas DataFrame 按 ID 和时间排序 (后续绘图和频率分析需要)
        logger.info("Sorting Pandas DataFrame by unique_id and timestamp...")
        pdf_sample = pdf_sample.sort_values(['unique_id', 'timestamp'])
        # 确保 timestamp 是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(pdf_sample['timestamp']):
            logger.info(
                "Converting timestamp column to datetime objects in Pandas...")
            pdf_sample['timestamp'] = pd.to_datetime(
                pdf_sample['timestamp'], errors='coerce')
            # 检查是否有转换失败的值
            if pdf_sample['timestamp'].isnull().any():
                logger.warning(
                    "Some timestamp values failed to convert to datetime in Pandas.")
                pdf_sample = pdf_sample.dropna(
                    subset=['timestamp'])  # 移除转换失败的行

        # 4. 在 Pandas 上进行绘图和频率分析 (这部分逻辑不变)
        logger.info(
            "Starting to plot time series for each sample (in Pandas) and analyze frequency...")
        plt.style.use('seaborn-v0_8-whitegrid')

        grouped_data = pdf_sample.groupby('unique_id')  # 按 ID 分组以便处理

        for unique_id, df_id in tqdm(grouped_data, desc="Processing samples (Pandas)"):
            # for unique_id in tqdm(sampled_ids, desc="Processing samples (Pandas)"): # 旧方法
            # df_id = pdf_sample[pdf_sample['unique_id'] == unique_id] # 旧方法
            if df_id.empty:
                logger.warning(
                    f"No data found for unique_id '{unique_id}' in the Pandas sample, skipping.")
                continue

            # --- 绘图逻辑 (不变) ---
            try:
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(df_id['timestamp'], df_id['y'],
                        # 小点+线
                        label=f'ID: {unique_id}', marker='.', linestyle='-', markersize=2)
                ax.set_title(
                    f'Demand (y) Time Series - unique_id: {unique_id}')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('Electricity Demand (y) in kWh')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                # 使用 Path 对象处理路径
                save_plot(fig, f'timeseries_sample_{unique_id}.png', plots_dir)
            except Exception as plot_e:
                logger.exception(
                    f"Error plotting time series for unique_id '{unique_id}': {plot_e}")
                plt.close(fig)  # Ensure plot is closed even on error

            # --- 频率分析逻辑 (不变) ---
            try:
                # 确保已排序 (已在分组前排序)
                time_diffs = df_id['timestamp'].diff(
                ).dropna()  # 计算差值并移除第一个 NaN
                if not time_diffs.empty:
                    # 转换为 Timedelta 字符串以便计数
                    freq_counts = time_diffs.astype(str).value_counts()
                    if not freq_counts.empty:
                        logger.info(
                            f"--- unique_id: {unique_id} Timestamp Interval Frequency (Pandas Sample) ---")
                        log_str = f"Frequency Stats (Top 5):\n{freq_counts.head().to_string()}"
                        if len(freq_counts) > 5:
                            log_str += "\n..."
                        logger.info(log_str)
                        if len(freq_counts) > 1:
                            logger.warning(
                                f" unique_id '{unique_id}' has multiple time intervals detected in sample.")
                    else:
                        logger.info(
                            f" unique_id '{unique_id}' has only one unique time interval after diff/dropna in sample.")
                else:
                    logger.info(
                        f" unique_id '{unique_id}' has less than two timestamps in sample, cannot calculate intervals.")
            except Exception as freq_e:
                logger.exception(
                    f"Error analyzing frequency for unique_id '{unique_id}' in sample: {freq_e}")

        logger.info(
            "Time series sample analysis (Spark filter -> Pandas plot/analyze) complete.")

    except Exception as e:
        logger.exception(
            f"Error analyzing Demand time series samples with Spark: {e}")
