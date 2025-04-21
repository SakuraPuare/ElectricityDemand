import sys
import os
# 移除 dask 和 pandas 导入 (如果不再需要)
# import dask.dataframe as dd
# import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from loguru import logger

# --- 辅助函数：打印 Schema 和行数 ---
def log_sdf_info(sdf: DataFrame, name: str):
    """记录 Spark DataFrame 的基本信息 (Schema 和行数)"""
    if sdf is None:
        logger.warning(f"DataFrame '{name}' is None.")
        return
    try:
        count = sdf.count() # Action to get count
        logger.info(f"{name} DataFrame - 行数: {count:,}")
        logger.info(f"{name} DataFrame - Schema:")
        sdf.printSchema() # Action to print schema
    except Exception as e:
        logger.exception(f"Error getting info for DataFrame '{name}': {e}")


# --- 修改后的加载函数 ---
def load_datasets(spark: SparkSession, demand_path: str, metadata_path: str, weather_path: str) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None]:
    """使用 Spark 加载 Demand, Metadata, 和 Weather 数据集."""
    logger.info("--- 开始使用 Spark 加载数据集 ---")
    ddf_demand, ddf_metadata, ddf_weather = None, None, None # Initialize

    # 加载 Demand 数据
    logger.info(f"加载 Demand 数据: {demand_path}")
    try:
        ddf_demand = spark.read.parquet(demand_path)
        log_sdf_info(ddf_demand, "Demand")
    except Exception as e:
        logger.exception(f"加载 Demand 数据时发生错误: {e}")
        logger.error("无法加载 Demand 数据，后续分析可能受影响。")
        # We might want to exit, but let's try loading others first

    # 加载 Metadata 数据
    logger.info(f"加载 Metadata 数据: {metadata_path}")
    try:
        # Metadata 通常较小，可以直接加载为 Spark DF，后续按需转 Pandas
        ddf_metadata = spark.read.parquet(metadata_path)
        log_sdf_info(ddf_metadata, "Metadata")
    except Exception as e:
        logger.exception(f"加载 Metadata 数据时发生错误: {e}")
        logger.error("无法加载 Metadata 数据，后续分析可能受影响。")

    # 加载 Weather 数据
    logger.info(f"加载 Weather 数据: {weather_path}")
    try:
        ddf_weather = spark.read.parquet(weather_path)
        log_sdf_info(ddf_weather, "Weather")
    except Exception as e:
        logger.exception(f"加载 Weather 数据时发生错误: {e}")
        logger.error("无法加载 Weather 数据，后续分析可能受影响。")

    logger.info("--- Spark 数据集加载完成 ---")
    return ddf_demand, ddf_metadata, ddf_weather

# --- 移除旧的 Dask/Pandas 加载函数 ---
# def load_demand_data(demand_path): ...
# def load_metadata(metadata_path): ...
# def load_weather_data(weather_path): ... 