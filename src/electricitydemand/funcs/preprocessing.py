# 移除 dask.dataframe 的导入
# import dask.dataframe as dd
import pandas as pd
from loguru import logger
from pyspark.sql import DataFrame as SparkDataFrame  # 使用 Spark DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType


def resample_demand_to_hourly_spark(sdf_demand: SparkDataFrame) -> SparkDataFrame | None:
    """
    Resamples the demand Spark DataFrame to hourly frequency using window functions.

    Aggregates the 'y' column (demand) using sum within each hour
    for each unique_id.

    Args:
        sdf_demand: Input Spark DataFrame with 'unique_id', 'timestamp', 'y'.
                    'timestamp' column must be convertible to TimestampType.

    Returns:
        Resampled Spark DataFrame with 'unique_id', 'timestamp' (hourly start), 'y' (summed).
        Returns None if input is None or on error.
    """
    if sdf_demand is None:
        logger.warning(
            "Input demand Spark DataFrame is None. Skipping resampling.")
        return None
    required_cols = {'unique_id', 'timestamp', 'y'}
    if not required_cols.issubset(sdf_demand.columns):
        logger.error(
            f"Input DataFrame missing required columns (need {required_cols}, have {sdf_demand.columns}). Cannot resample.")
        return None

    logger.info("开始将 Demand 数据重采样至小时频率 (使用 Spark)...")
    # Spark 获取分区数的方式
    logger.info(f"原始分区数 (估计): {sdf_demand.rdd.getNumPartitions()}")

    try:
        # 确保 timestamp 是 Timestamp 类型
        # 如果已经是 TimestampType，转换操作是幂等的
        logger.info("确保 'timestamp' 列是 TimestampType...")
        sdf_demand = sdf_demand.withColumn(
            "timestamp", F.col("timestamp").cast(TimestampType()))

        # 检查转换后是否有 null 时间戳 (cast 失败可能产生 null)
        null_timestamps = sdf_demand.where(F.col("timestamp").isNull()).count()
        if null_timestamps > 0:
            logger.warning(
                f"发现 {null_timestamps} 条记录时间戳转换失败 (变为 null)。这些记录将被忽略。")
            sdf_demand = sdf_demand.dropna(subset=["timestamp"])

        # 使用窗口函数进行重采样
        # 按 unique_id 和 1 小时的时间窗口分组
        # F.window 参数: timeColumn, windowDuration, [slideDuration, startTime]
        # 这里我们只需要 windowDuration 为 "1 hour"
        logger.info("按 unique_id 和小时窗口进行分组聚合 (sum)...")
        sdf_resampled = sdf_demand.groupBy(
            F.col("unique_id"),
            F.window(F.col("timestamp"), "1 hour")
        ).agg(
            F.sum("y").alias("y")  # 对每个窗口内的 y 求和
        )

        # 提取窗口的开始时间作为新的 timestamp，并选择所需列
        logger.info("提取窗口开始时间并选择最终列...")
        sdf_resampled = sdf_resampled.select(
            F.col("unique_id"),
            F.col("window.start").alias("timestamp"),  # 窗口开始时间
            F.col("y")
        ).orderBy("unique_id", "timestamp")  # 保持排序一致性

        # 打印一些信息以供验证
        logger.info("重采样后的数据结构预览 (Schema):")
        sdf_resampled.printSchema()
        logger.info(f"重采样后的分区数 (估计): {sdf_resampled.rdd.getNumPartitions()}")
        logger.info("Demand 数据小时重采样 (Spark) 完成。")

        return sdf_resampled

    except Exception as e:
        logger.exception(f"Spark 重采样过程中发生错误：{e}")
        return None


def validate_resampling_spark(sdf_resampled: SparkDataFrame | None, n_check: int = 5):
    """Validates the resampling by checking timestamp frequency using Spark."""
    if sdf_resampled is None:
        logger.warning(
            "Resampled Spark DataFrame is None. Skipping validation.")
        return

    logger.info("--- 开始验证 Spark 重采样结果 ---")
    try:
        # Ensure required columns exist before proceeding
        required_cols = {'timestamp', 'unique_id', 'y'}
        if not required_cols.issubset(sdf_resampled.columns):
            logger.error(
                f"验证失败：重采样后的 DataFrame 缺少必需列。需要：{required_cols}, 实际：{sdf_resampled.columns}")
            logger.info("重采样后的数据结构预览 (Schema):")
            sdf_resampled.printSchema()
            return

        # 检查时间戳是否都是整点小时
        logger.info(f"获取前 {n_check} 条记录的时间戳进行检查...")
        # 使用 limit 获取少量数据，然后 collect 到 Driver 端进行检查
        # 注意: collect() 会将数据拉到 Driver 内存，只适用于小量数据检查
        sample_df_pd = sdf_resampled.select(
            "timestamp", "unique_id").limit(n_check).toPandas()

        if sample_df_pd.empty:
            logger.warning("无法获取样本时间戳进行验证 (可能数据为空？)")
            return

        # 在 Pandas DataFrame 上进行检查
        all_hourly = all((ts.minute == 0 and ts.second == 0 and ts.microsecond == 0)
                         for ts in sample_df_pd['timestamp'])

        if all_hourly:
            logger.success("样本时间戳验证成功：所有检查的时间戳都是整点小时。")
        else:
            logger.warning("样本时间戳验证失败：部分时间戳不是整点小时。")
            logger.warning(
                f"前 {n_check} 个时间戳和 ID (从 Spark 获取):\n{sample_df_pd}")

    except Exception as e:
        logger.error(f"验证 Spark 重采样时出错：{e}")
    logger.info("--- 完成验证 Spark 重采样结果 ---")
