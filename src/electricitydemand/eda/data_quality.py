import traceback
from pathlib import Path  # Import Path

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
    TimestampType,  # Add TimestampType
    DateType  # Add DateType
)


def check_missing_values_spark(sdf: SparkDataFrame, name: str):
    """
    使用 Spark DataFrame 计算并记录每个列的缺失值数量和百分比。

    Args:
        sdf (SparkDataFrame): 输入的 Spark DataFrame。
        name (str): DataFrame 的名称，用于日志记录。
    """
    if not isinstance(sdf, SparkDataFrame):
        logger.error(f"Input '{name}' is not a Spark DataFrame.")
        return
    if not sdf.columns:
        logger.warning(f"{name} DataFrame is empty (no columns).")
        return

    logger.info(f"--- 开始检查 {name} DataFrame 缺失值 (Spark) ---")
    total_count = sdf.count()
    logger.info(f"{name} 总行数: {total_count:,}")

    if total_count == 0:
        logger.info(f"{name} 是空的，跳过缺失值检查。")
        return

    try:
        # 构建缺失值计数的聚合表达式
        exprs = []
        numeric_float_types = (FloatType, DoubleType,
                               DecimalType)  # Include DecimalType
        other_types = (StringType, IntegerType, LongType, ShortType,
                       ByteType, TimestampType, DateType)  # Add time types

        for col_name in sdf.columns:
            col_type = sdf.schema[col_name].dataType
            # 对浮点数值类型，检查 isnan 和 isull
            if isinstance(col_type, numeric_float_types):
                expr = F.sum(
                    F.when(F.isnan(F.col(col_name)) | F.isnull(
                        F.col(col_name)), 1).otherwise(0)
                ).alias(f"{col_name}_missing_count")
            # 对其他类型（包括 String, Timestamp, Integer等），只检查 isull
            elif isinstance(col_type, other_types):
                expr = F.sum(F.when(F.isnull(F.col(col_name)), 1).otherwise(0)).alias(
                    f"{col_name}_missing_count"
                )
            else:
                logger.warning(
                    f"Column '{col_name}' has unhandled type {col_type} for missing value check. Skipping sum aggregation, will count nulls separately if needed.")
                # Fallback: count nulls directly (less efficient if many columns)
                # Or handle specific types if required
                # Count non-nulls and subtract? No, sum is better. Let's stick to the sum approach.
                expr = F.count(F.when(F.isnull(F.col(col_name)), col_name)).alias(
                    f"{col_name}_missing_count")
                # Revert to simple null check for safety with unhandled types
                expr = F.sum(F.when(F.isnull(F.col(col_name)), 1).otherwise(0)).alias(
                    f"{col_name}_missing_count"
                )

            exprs.append(expr)

        # 执行聚合计算
        if not exprs:
            logger.warning(
                f"No columns found to check for missing values in {name}.")
            return

        missing_counts = sdf.agg(*exprs).first()

        # 记录结果
        has_missing = False
        if missing_counts:  # Check if aggregation returned a result
            for col_name in sdf.columns:
                count_col_name = f"{col_name}_missing_count"
                # Check if the aggregated column exists in the result Row
                if count_col_name in missing_counts.asDict():
                    count = missing_counts[count_col_name]
                    if count is not None and count > 0:  # Ensure count is not None
                        percentage = (count / total_count) * 100
                        logger.warning(
                            f"{name} - 列 '{col_name}' 缺失值数量: {count:,} ({percentage:.2f}%)"
                        )
                        has_missing = True
                    elif count is not None:
                        logger.info(f"{name} - 列 '{col_name}' 无缺失值")
                    else:
                        # This case might happen if the aggregation failed for a specific column type implicitly
                        logger.warning(
                            f"{name} - 列 '{col_name}' 缺失值计数结果为 None，请检查。")
                else:
                    logger.warning(
                        f"Could not find missing count result for column '{col_name}'.")

            if not has_missing:
                logger.info(f"{name} 中未发现缺失值。")
        else:
            logger.error(f"未能从 {name} 计算缺失值统计。")

    except Exception as e:
        logger.exception(f"检查 {name} 缺失值时发生错误: {e}")
        # Removed raise e to allow script continuation


def check_duplicates_spark(sdf: SparkDataFrame, subset_cols: list, df_name: str):
    """
    检查 Spark DataFrame 中基于指定列子集的重复行。

    Args:
        sdf (SparkDataFrame): 要检查的 Spark DataFrame。
        subset_cols (list): 用于判断重复的列名列表。
        df_name (str): DataFrame 的名称，用于日志输出。
    """
    logger.info(
        f"--- 开始检查 {df_name} DataFrame 中基于 {subset_cols} 的重复行 (Spark) ---")
    if not isinstance(sdf, SparkDataFrame):
        logger.error(f"Input '{df_name}' is not a Spark DataFrame.")
        return
    if not sdf.columns:
        logger.warning(f"{df_name} DataFrame 为空 (no columns)，跳过重复行检查。")
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
        duplicates_df = sdf.groupBy(subset_cols).count()
        duplicate_groups = duplicates_df.where(F.col("count") > 1)

        # 缓存重复组以避免重复计算 (如果后续需要多次使用)
        duplicate_groups.persist()  # Optional: persist if used multiple times below

        # 计算重复组的数量 (Action)
        num_duplicate_groups = duplicate_groups.count()

        if num_duplicate_groups > 0:
            # 计算涉及的总重复行数 (Action)
            # Ensure the aggregation result is not None
            agg_result = duplicate_groups.agg(F.sum("count")).first()
            total_duplicate_rows = agg_result[0] if agg_result else 0

            if total_duplicate_rows is not None:
                # 计算"额外"的重复行数
                extra_duplicate_rows = total_duplicate_rows - num_duplicate_groups

                logger.warning(f"在 {df_name} 中发现基于 {subset_cols} 的重复数据:")
                logger.warning(f"  - 唯一重复组合的数量: {num_duplicate_groups:,}")
                logger.warning(f"  - 涉及的总行数（包括首次出现）: {total_duplicate_rows:,}")
                logger.warning(f"  - '额外'的重复行数（需移除）: {extra_duplicate_rows:,}")

                # 可选：显示一些重复的例子 (限制数量以防过多输出)
                # logger.info("重复组示例 (前 5 个):")
                # duplicate_groups.show(5, truncate=False)
            else:
                logger.error(f"无法计算 {df_name} 中的总重复行数。")

        else:
            logger.info(f"在 {df_name} 中未发现基于 {subset_cols} 的重复行。")

        duplicate_groups.unpersist()  # Unpersist the DataFrame

    except Exception as e:
        logger.exception(f"检查 {df_name} 重复行时发生错误: {e}")
    finally:
        # Ensure unpersist happens even if errors occur above (might need specific context)
        try:
            if duplicate_groups and duplicate_groups.is_cached:
                duplicate_groups.unpersist()
        except Exception as unpersist_e:
            logger.warning(
                f"Error unpersisting duplicate_groups: {unpersist_e}")

    logger.info(f"--- 完成 {df_name} DataFrame 重复行检查 ---")
