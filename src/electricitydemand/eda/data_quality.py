import traceback

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, isnan, isnull
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    ShortType,
    StringType,
)


def check_missing_values_spark(sdf: SparkDataFrame, name: str):
    """
    使用 Spark DataFrame 计算并记录每个列的缺失值数量和百分比。

    Args:
        sdf (SparkDataFrame): 输入的 Spark DataFrame。
        name (str): DataFrame 的名称，用于日志记录。
    """
    total_count = sdf.count()
    logger.info(f"{name} 总行数: {total_count}")

    if total_count == 0:
        logger.info(f"{name} 是空的，跳过缺失值检查。")
        return

    try:
        # 构建缺失值计数的聚合表达式
        exprs = []
        for col_name in sdf.columns:
            col_type = sdf.schema[col_name].dataType
            # 对数值类型，检查 isnan 和 isull
            if isinstance(col_type, (FloatType, DoubleType)):
                expr = F.sum(
                    F.when(F.isnan(F.col(col_name)) | F.isnull(F.col(col_name)), 1).otherwise(0)
                ).alias(f"{col_name}_missing_count")
            # 对其他类型（包括 String, Timestamp, Integer等），只检查 isull
            else:
                expr = F.sum(F.when(F.isnull(F.col(col_name)), 1).otherwise(0)).alias(
                    f"{col_name}_missing_count"
                )
            exprs.append(expr)

        # 执行聚合计算
        missing_counts = sdf.agg(*exprs).first()

        # 记录结果
        has_missing = False
        for col_name in sdf.columns:
            count = missing_counts[f"{col_name}_missing_count"]
            if count > 0:
                percentage = (count / total_count) * 100
                logger.warning(
                    f"{name} - 列 '{col_name}' 缺失值数量: {count} ({percentage:.2f}%)"
                )
                has_missing = True
            else:
                logger.info(f"{name} - 列 '{col_name}' 无缺失值")

        if not has_missing:
            logger.info(f"{name} 中未发现缺失值。")

    except Exception as e:
        logger.error(f"检查 {name} 缺失值时发生错误: {e}")
        # 可以在这里重新抛出异常，或者根据需要处理
        raise e


def check_duplicates_spark(sdf: SparkDataFrame, subset_cols: list, df_name: str):
    """
    检查 Spark DataFrame 中基于指定列子集的重复行。

    Args:
        sdf (SparkDataFrame): 要检查的 Spark DataFrame。
        subset_cols (list): 用于判断重复的列名列表。
        df_name (str): DataFrame 的名称，用于日志输出。
    """
    logger.info(f"--- 开始检查 {df_name} DataFrame 中基于 {subset_cols} 的重复行 (Spark) ---")
    if not sdf:
        logger.warning(f"{df_name} DataFrame 为空，跳过重复行检查。")
        return
    if not subset_cols:
        logger.warning("未提供用于检查重复的列，跳过检查。")
        return

    # 检查指定的列是否存在
    missing_cols = [col for col in subset_cols if col not in sdf.columns]
    if missing_cols:
        logger.error(f"指定的列 {missing_cols} 在 {df_name} DataFrame 中不存在，无法检查重复。")
        return

    try:
        # 计算基于 subset_cols 的重复行数量
        # 1. 对指定列分组
        # 2. 计算每组的行数
        # 3. 筛选出行数大于 1 的组（这些是重复的组合）
        # 4. 计算这些重复组的总行数（即所有重复行，包括第一次出现的）
        # 5. 计算唯一重复组合的数量
        duplicates_df = sdf.groupBy(subset_cols).count()
        duplicate_groups = duplicates_df.where(F.col("count") > 1)

        # 计算重复组的数量
        num_duplicate_groups = duplicate_groups.count()

        if num_duplicate_groups > 0:
            # 计算涉及的总重复行数 (summing the counts for groups with count > 1)
            # 这会计算所有重复的行，例如，如果一组有3行，这会算作3行
            total_duplicate_rows = duplicate_groups.agg(F.sum("count")).first()[0]
            # 计算"额外"的重复行数 (total_duplicate_rows - num_duplicate_groups)
            extra_duplicate_rows = total_duplicate_rows - num_duplicate_groups

            logger.warning(f"在 {df_name} 中发现基于 {subset_cols} 的重复数据:")
            logger.warning(f"  - 唯一重复组合的数量: {num_duplicate_groups:,}")
            logger.warning(f"  - 涉及的总行数（包括首次出现）: {total_duplicate_rows:,}")
            logger.warning(f"  - '额外'的重复行数（需移除）: {extra_duplicate_rows:,}")

            # 可选：显示一些重复的例子 (如果需要，但可能很慢)
            # logger.info("重复组示例:")
            # duplicate_groups.show(5, truncate=False)

        else:
            logger.info(f"在 {df_name} 中未发现基于 {subset_cols} 的重复行。")

    except Exception as e:
        logger.exception(f"检查 {df_name} 重复行时发生错误: {e}")

    logger.info(f"--- 完成 {df_name} DataFrame 重复行检查 ---") 