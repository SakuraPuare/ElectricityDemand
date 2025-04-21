import sys
import os
# 移除 dask 和 pandas 导入 (如果不再需要)
# import dask.dataframe as dd
# import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F # 引入 Spark SQL 函数
from pyspark.sql import types as T # 引入 Spark SQL 类型
from loguru import logger
from pathlib import Path # 使用 Path 处理路径
import time
from typing import Optional, Tuple
from pyspark.sql.utils import AnalysisException

# --- 辅助函数：打印 Schema 和行数 ---
def log_sdf_info(sdf: DataFrame, name: str):
    """Logs schema and count for a Spark DataFrame."""
    if sdf is None:
        logger.warning(f"DataFrame '{name}' is None, skipping info logging.")
        return
    try:
        logger.info(f"{name} DataFrame - Schema:")
        sdf.printSchema() # Action to print schema
        logger.info(f"{name} DataFrame - Count: {sdf.count():,}") # Action to count rows
    except Exception as e:
        logger.error(f"Error getting info for DataFrame '{name}': {e}")
        # Log traceback for debugging
        logger.exception("Traceback:")


# --- 修改后的加载函数 ---
def load_datasets(spark: SparkSession, data_dir: Path) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None]:
    """使用 Spark 加载 Demand, Metadata, 和 Weather 数据集."""
    logger.info("--- 开始使用 Spark 加载数据集 ---")
    sdf_demand, sdf_metadata, sdf_weather = None, None, None # Initialize

    # --- 确定文件路径 ---
    # 数据文件现在包含 '_converted' 后缀
    demand_path = str(data_dir / "demand_converted.parquet")
    metadata_path = str(data_dir / "metadata.parquet") # metadata 文件名没有后缀
    weather_path = str(data_dir / "weather_converted.parquet")
    logger.info(f"Demand path: {demand_path}")
    logger.info(f"Metadata path: {metadata_path}")
    logger.info(f"Weather path: {weather_path}")

    # 加载 Demand 数据
    logger.info(f"加载 Demand 数据: {demand_path}")
    try:
        sdf_demand = spark.read.parquet(demand_path)
        # --- 时间戳转换 ---
        # 确保 timestamp 列是 TimestampType
        if 'timestamp' in sdf_demand.columns and not isinstance(sdf_demand.schema['timestamp'].dataType, T.TimestampType):
             logger.warning("Demand 'timestamp' 列不是 TimestampType，尝试转换...")
             sdf_demand = sdf_demand.withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
             # 可以在转换后再次检查类型
             logger.info("Demand 'timestamp' 转换后类型: " + str(sdf_demand.schema['timestamp'].dataType))
        log_sdf_info(sdf_demand, "Demand")
    except Exception as e:
        # 使用 f-string 格式化异常信息
        logger.exception(f"加载 Demand 数据时发生错误: {e}")
        logger.error("无法加载 Demand 数据，后续分析可能受影响。")

    # 加载 Metadata 数据
    logger.info(f"加载 Metadata 数据: {metadata_path}")
    try:
        # Metadata 通常较小，可以直接加载为 Spark DF，后续按需转 Pandas
        sdf_metadata = spark.read.parquet(metadata_path)
        log_sdf_info(sdf_metadata, "Metadata")
    except Exception as e:
        logger.exception(f"加载 Metadata 数据时发生错误: {e}")
        logger.error("无法加载 Metadata 数据，后续分析可能受影响。")

    # 加载 Weather 数据
    logger.info(f"加载 Weather 数据: {weather_path}")
    try:
        sdf_weather = spark.read.parquet(weather_path)
         # --- 时间戳转换 ---
         # 确保 timestamp 列是 TimestampType
        if 'timestamp' in sdf_weather.columns and not isinstance(sdf_weather.schema['timestamp'].dataType, T.TimestampType):
            logger.warning("Weather 'timestamp' 列不是 TimestampType，尝试转换...")
            sdf_weather = sdf_weather.withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
            logger.info("Weather 'timestamp' 转换后类型: " + str(sdf_weather.schema['timestamp'].dataType))
        log_sdf_info(sdf_weather, "Weather")
    except Exception as e:
        logger.exception(f"加载 Weather 数据时发生错误: {e}")
        logger.error("无法加载 Weather 数据，后续分析可能受影响。")

    logger.info("--- Spark 数据集加载完成 ---")
    return sdf_demand, sdf_metadata, sdf_weather

# --- 移除旧的 Dask/Pandas 加载函数 ---
# def load_demand_data(demand_path): ...
# def load_metadata(metadata_path): ...
# def load_weather_data(weather_path): ... 