import pandas as pd
# 移除 Dask 导入
# import dask.dataframe as dd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# 使用相对导入
from ..utils.eda_utils import plot_numerical_distribution, log_value_counts, save_plot, plot_categorical_distribution
# 移除 dask_compute_context

def analyze_weather_numerical(sdf_weather: DataFrame, columns_to_analyze=None, plot_sample_frac=0.1, plots_dir=None, random_state=42):
    """分析 Weather Spark DataFrame 中指定数值列的分布并记录/绘图。"""
    if sdf_weather is None:
        logger.warning("输入的 Weather Spark DataFrame 为空，跳过数值特征分析。")
        return
    plot = plots_dir is not None
    if not plot:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Weather 数值特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'wind_speed_10m', 'cloud_cover']
    logger.info(f"--- 开始分析 Weather 数值特征分布 (Spark) ({', '.join(columns_to_analyze)}) ---")

    # 1. 计算描述性统计 (Spark)
    desc_stats_pd = None
    relevant_cols = [col for col in columns_to_analyze if col in sdf_weather.columns]
    # 过滤掉非数值列
    numerical_cols = []
    for col in relevant_cols:
        try:
            if isinstance(sdf_weather.schema[col].dataType, NumericType):
                numerical_cols.append(col)
            else:
                logger.warning(f"列 '{col}' 非数值类型 ({sdf_weather.schema[col].dataType})，将从统计中排除。")
        except KeyError:
            logger.warning(f"列 '{col}' 不在 DataFrame schema 中，跳过。")

    if numerical_cols:
        try:
            logger.info("使用 Spark 计算数值特征的描述性统计...")
            # Spark describe 计算的是字符串类型，需要后续处理
            desc_stats_spark_df = sdf_weather.select(numerical_cols).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
            # 转为 Pandas 方便打印和后续使用
            desc_stats_pd = desc_stats_spark_df.toPandas()
            desc_stats_pd.set_index('summary', inplace=True)
            logger.info(f"Spark 计算的描述性统计:\n{desc_stats_pd.to_string()}")
        except Exception as e:
            logger.exception(f"使用 Spark 计算描述性统计时出错: {e}")

    # 2. 检查负值 (Spark)
    precipitation_cols = ['precipitation', 'rain', 'snowfall']
    neg_check_cols = [col for col in precipitation_cols if col in numerical_cols] # 只检查存在的数值列
    if neg_check_cols:
        logger.info(f"使用 Spark 检查列 {neg_check_cols} 中的负值...")
        try:
            for col in neg_check_cols:
                negative_count = sdf_weather.filter(F.col(col) < 0).count() # Action
                if negative_count > 0:
                    logger.warning(f"列 '{col}' 检测到 {negative_count:,} 个负值！请检查数据源或处理。")
                else:
                    logger.info(f"列 '{col}' 未检测到负值。")
        except Exception as e:
            logger.exception(f"使用 Spark 检查负值时出错: {e}")


    # 3. 绘图 (Spark 抽样 -> Pandas 绘图)
    if plot and numerical_cols:
        logger.info(f"开始绘制 Weather 特征分布图 (抽样比例: {plot_sample_frac:.1%})...")
        for col in numerical_cols:
            # 检查统计信息是否可用 (可选，如果上面计算失败则跳过)
            # if desc_stats_pd is None or col not in desc_stats_pd.columns:
            #     logger.warning(f"列 '{col}' 的统计信息不可用，跳过绘图。")
            #     continue

            logger.info(f"绘制列: {col}")
            try:
                # Spark 抽样
                logger.debug(f"对列 '{col}' 进行 Spark 抽样...")
                # 先过滤 NaN/Null，然后抽样
                col_sample_sdf = sdf_weather.select(col).dropna().sample(False, plot_sample_frac, seed=random_state)

                # 收集到 Pandas
                logger.debug(f"将列 '{col}' 的抽样结果收集到 Pandas...")
                col_sample_pd = col_sample_sdf.toPandas()[col] # 获取 Pandas Series
                logger.debug(f"列 '{col}' 抽样完成，样本大小: {len(col_sample_pd):,}")

                if col_sample_pd is None or col_sample_pd.empty:
                    logger.warning(f"列 '{col}' 的抽样结果为空或收集失败，跳过绘图。")
                    continue

                # 使用辅助函数绘图 (接收 Pandas Series)
                plot_numerical_distribution(col_sample_pd, col,
                                             f'weather_distribution_{col}', plots_dir,
                                             title_prefix="Weather ", kde=True)
            except Exception as e:
                logger.exception(f"绘制列 '{col}' 的分布图时出错: {e}")

    logger.info("Weather 数值特征分析完成。")


def analyze_weather_categorical(sdf_weather: DataFrame, columns_to_analyze=None, top_n=20, plots_dir=None):
    """分析 Weather Spark DataFrame 中指定分类列的分布并记录/绘图。"""
    if sdf_weather is None:
        logger.warning("输入的 Weather Spark DataFrame 为空，跳过分类特征分析。")
        return
    plot = plots_dir is not None
    if not plot:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Weather 分类特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['weather_code', 'is_day'] # is_day 也是分类的
    logger.info(f"--- 开始分析 Weather 分类特征分布 (Spark) ({', '.join(columns_to_analyze)}) ---")

    relevant_cols = [col for col in columns_to_analyze if col in sdf_weather.columns]

    for col in relevant_cols:
        logger.info(f"--- 分析列: {col} ---")
        try:
            logger.info(f"使用 Spark 计算列 '{col}' 的值分布...")
            # Spark 实现 value_counts: groupBy + count + orderBy
            counts_sdf = sdf_weather.groupBy(col).count().orderBy(F.desc("count"))

            # 收集到 Pandas DataFrame
            logger.debug(f"将列 '{col}' 的值分布结果收集到 Pandas...")
            counts_pd = counts_sdf.toPandas() # DataFrame with columns [col, 'count']

            if counts_pd is None or counts_pd.empty:
                 logger.warning(f"未能计算或计算结果为空，跳过记录/绘制列 '{col}' 的值分布。")
                 continue

            # 记录日志 (log_value_counts 接收 Pandas DF)
            # 需要确保列名是 'value' 和 'count' 或传递正确的列名
            # 重命名 Pandas DF 列以匹配 log_value_counts 预期 (或修改 log_value_counts)
            counts_pd_renamed = counts_pd.rename(columns={col: "value", "count": "计数"})
            log_value_counts(counts_pd_renamed, col, top_n=top_n, is_already_counts=True)

            # 绘图 (plot_categorical_distribution 接收 Pandas Series 或 DF)
            if plot:
                logger.info(f"绘制列: {col}")
                # 将 Pandas DF 转换为 Series (value 作为 index, count 作为 values) 以便绘图
                counts_series = counts_pd.set_index(col)['count']
                plot_categorical_distribution(
                     counts_series, # 传递 Series
                     col,
                     f"weather_distribution_{col}",
                     plots_dir,
                     top_n=top_n,
                     title_prefix="Weather "
                 )

        except Exception as e:
            logger.exception(f"分析列 '{col}' 时出错: {e}")

    logger.info("Weather 分类特征分析完成。")

# 移除旧的 Dask 相关代码 