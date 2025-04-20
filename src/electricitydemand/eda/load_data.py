import sys
import os
import dask.dataframe as dd
import pandas as pd
from loguru import logger

def load_demand_data(demand_path):
    """加载 Demand 数据集."""
    logger.info("开始加载 Demand 数据集...")
    try:
        ddf_demand = dd.read_parquet(demand_path)
        logger.info(f"成功加载 Demand 数据: {demand_path}")
        logger.info(f"Demand Dask DataFrame 分区数: {ddf_demand.npartitions}, 列: {ddf_demand.columns.tolist()}")
        return ddf_demand
    except FileNotFoundError as e:
        logger.error(f"Demand 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Demand 数据集时发生错误: {e}")
        sys.exit(1)

def load_metadata(metadata_path):
    """加载 Metadata 数据集 (Pandas)."""
    logger.info("开始加载 Metadata 数据集...")
    try:
        pdf_metadata = pd.read_parquet(metadata_path)
        logger.info(f"成功加载 Metadata 数据: {metadata_path}")
        logger.info(f"Metadata Pandas DataFrame 形状: {pdf_metadata.shape}, 列: {pdf_metadata.columns.tolist()}")
        logger.info(f"Metadata 头部信息:\n{pdf_metadata.head().to_string()}")
        logger.info(f"Metadata 数据类型:\n{pdf_metadata.dtypes.to_string()}")
        return pdf_metadata
    except FileNotFoundError as e:
        logger.error(f"Metadata 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Metadata 数据集时发生错误: {e}")
        sys.exit(1)

def load_weather_data(weather_path):
    """加载 Weather 数据集 (Dask)."""
    logger.info("开始加载 Weather 数据集...")
    try:
        ddf_weather = dd.read_parquet(weather_path)
        logger.info(f"成功加载 Weather 数据: {weather_path}")
        logger.info(f"Weather Dask DataFrame 分区数: {ddf_weather.npartitions}, 列: {ddf_weather.columns.tolist()}")
        logger.info(f"Weather 数据类型:\n{ddf_weather.dtypes.to_string()}")
        return ddf_weather
    except FileNotFoundError as e:
        logger.error(f"Weather 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Weather 数据集时发生错误: {e}")
        sys.exit(1) 