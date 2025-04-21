import os
import sys
import time
from pathlib import Path  # Use pathlib for paths

# 移除 dask 和 pandas 的导入 (如果不再直接使用它们加载)
# import dask.dataframe as dd
# import pandas as pd
import pandas as pd  # Pandas 仍然需要用于 Metadata 分析和部分绘图
from loguru import logger

# 引入 Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # 常用函数别名

# --- 项目设置 ---
# Determine project root dynamically
try:
    _script_path = os.path.abspath(__file__)
    project_root = Path(_script_path).parent.parent.parent  # Use Path object
except NameError:  # If __file__ is not defined (e.g., interactive)
    project_root = Path.cwd()
    # Add project root to path if running interactively might be needed
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Add src directory to sys.path to allow absolute imports from src
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- 配置日志 ---
# Assuming log_utils is in src/electricitydemand/utils
try:
    from electricitydemand.utils.log_utils import setup_logger
except ImportError:
    print("Error: Could not import setup_logger. Ensure src is in PYTHONPATH or script is run correctly.", file=sys.stderr)
    # Fallback basic logging if setup fails
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

log_prefix = Path(__file__).stem  # Use run_eda as prefix
logs_dir = project_root / 'logs'
plots_dir = project_root / 'plots'  # 图表保存目录 (Path object)
data_dir = project_root / "data"  # 数据目录 (Path object)

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)  # 确保目录存在

# 设置日志级别为 INFO (only if setup_logger was imported)
if 'setup_logger' in globals():
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")

logger.info(f"项目根目录: {project_root}")
logger.info(f"数据目录: {data_dir}")
logger.info(f"日志目录: {logs_dir}")
logger.info(f"图表目录: {plots_dir}")

# --- 导入分析函数 (Spark 版本) ---
try:
    # 假设 load_datasets 已更新为返回 Spark DataFrames
    # Use the correct path if it moved
    from electricitydemand.eda.analyze_demand import (  # plot_demand_y_distribution is called internally by analyze_demand_y_distribution
        analyze_demand_timeseries_sample,
        analyze_demand_y_distribution,
    )

    # 需要导入 relationship analysis 和 metadata 相关函数，因为 weather analysis 依赖它们
    # Metadata functions still work on Pandas DF
    from electricitydemand.eda.analyze_metadata import (
        analyze_metadata_categorical,
        analyze_metadata_numerical,
        analyze_missing_locations,
        plot_metadata_categorical,
    )
    from electricitydemand.eda.analyze_relationships import (
        analyze_demand_vs_location,
        analyze_demand_vs_metadata,
        analyze_demand_vs_weather,
    )
    from electricitydemand.eda.analyze_time import (
        analyze_datetime_features_spark,
        analyze_timestamp_consistency,
    )
    from electricitydemand.eda.analyze_weather import (
        analyze_weather_categorical,
        analyze_weather_correlation,
        analyze_weather_numerical,
        analyze_weather_timeseries_sample,
    )
    from electricitydemand.eda.data_quality import (
        check_duplicates_spark,
        check_missing_values_spark,
    )
    from electricitydemand.eda.load_data import load_datasets
except ImportError as e:
    logger.exception(f"Failed to import necessary EDA/ETL modules: {e}")
    sys.exit(1)


def run_all_eda():
    """Runs the entire EDA pipeline using Spark."""
    logger.info("=========================================")
    logger.info("===     开始执行 Spark EDA 脚本       ===")
    logger.info("=========================================")
    start_run_time = time.time()

    spark = None  # Initialize spark session variable

    try:
        # --- Spark Session ---
        logger.info("创建 SparkSession...")
        # 根据系统可用内存动态设置 Spark 内存配置
        import psutil

        # 获取系统当前剩余内存
        total_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        # 计算合适的驱动器和执行器内存（剩余内存的40%和40%）
        driver_memory = max(2, int(total_memory_gb * 0.4 + 0.5))
        executor_memory = max(2, int(total_memory_gb * 0.4 + 0.5))

        logger.info(
            f"系统总内存: {total_memory_gb:.2f}GB, 设置驱动器内存: {driver_memory}g, 执行器内存: {executor_memory}g")

        spark = SparkSession.builder \
            .appName("ElectricityDemandEDA") \
            .config("spark.driver.memory", f"{driver_memory}g") \
            .config("spark.executor.memory", f"{executor_memory}g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 步骤 1: 加载数据 ---
        logger.info("--- 步骤 1: 加载数据 (Spark) ---")
        sdf_demand, sdf_metadata, sdf_weather = load_datasets(spark, data_dir)

        if sdf_demand is None or sdf_metadata is None or sdf_weather is None:
            raise ValueError("一个或多个数据集加载失败，请检查路径和文件。")
        logger.info("数据集加载成功。")
        # Optional: Log schemas and counts 
        logger.info("Demand Schema:")
        sdf_demand.printSchema()
        logger.info(f"Demand Count: {sdf_demand.count():,}")
        logger.info("Metadata Schema:")
        sdf_metadata.printSchema()
        logger.info(f"Metadata Count: {sdf_metadata.count():,}")
        logger.info("Weather Schema:")
        sdf_weather.printSchema()
        logger.info(f"Weather Count: {sdf_weather.count():,}")

        # --- 步骤 2: 单变量分析  ---
        logger.info("--- 步骤 2: 单变量分析 ---")
        logger.info("--- 开始 Demand 分析 (Spark) ---")
        # 假设这些函数已适配 Spark 或能处理 Spark DF
        analyze_demand_y_distribution(sdf_demand, plots_dir=plots_dir, sample_frac=0.005)
        analyze_demand_timeseries_sample(sdf_demand, plots_dir=plots_dir, n_samples=3)
        logger.info("--- 完成 Demand 分析 (Spark) ---")

        # --- Metadata 分析 (仍然需要运行，因为 analyze_demand_vs_weather 需要 pdf_metadata) ---
        logger.info("--- 开始 Metadata 分析 (Spark DF -> Pandas DF) ---")
        logger.info("将 Metadata Spark DataFrame 转换为 Pandas DataFrame...")
        pdf_metadata = None  # 初始化
        try:
            pdf_metadata = sdf_metadata.toPandas()
            logger.info(
                f"Metadata Pandas DataFrame 转换成功，形状: {pdf_metadata.shape}")
            if 'unique_id' in pdf_metadata.columns:
                pdf_metadata = pdf_metadata.set_index('unique_id')
            else:
                logger.error("Metadata DataFrame 中缺少 'unique_id' 列，无法设置为索引。")
                raise ValueError("Metadata missing 'unique_id' column.")

        except Exception as meta_collect_e:
            logger.exception("将 Metadata 转换为 Pandas 时出错。无法继续 Metadata 分析。")
            # Skip metadata analysis if conversion fails
            # pdf_metadata = None # Ensure it's None # Redundant
            raise ValueError("Failed to convert Metadata to Pandas.") # Raise error if conversion fails

        if pdf_metadata is not None:
            # 这部分分析取消注释
            analyze_metadata_categorical(pdf_metadata) # Log counts
            plot_metadata_categorical(pdf_metadata, plots_dir=plots_dir) # Plot distributions
            analyze_metadata_numerical(pdf_metadata, plots_dir=plots_dir) # Analyze/Plot numerical
            analyze_missing_locations(pdf_metadata) # Analyze missing location info
            logger.info("--- 完成 Metadata 分析 ---")
        else:
            # This block should not be reached if the raise above works
            logger.error("Metadata 转换为 Pandas DataFrame 失败，无法进行后续分析。")
            raise ValueError("Failed to convert Metadata to Pandas.")

        # --- Weather 分析  ---
        logger.info("--- 开始 Weather 分析 (Spark) ---")
        # 假设这些函数已适配 Spark 或能处理 Spark DF
        analyze_weather_numerical(sdf_weather, plots_dir=plots_dir, plot_sample_frac=0.05)
        analyze_weather_categorical(sdf_weather, plots_dir=plots_dir)
        analyze_weather_timeseries_sample(sdf_weather, plots_dir=plots_dir, n_samples=3)
        analyze_weather_correlation(sdf_weather, plots_dir=plots_dir)
        logger.info("--- 完成 Weather 分析 (Spark) ---")

        # --- 步骤 3: 关系分析 (全部运行) ---
        logger.info("--- 步骤 3: 关系分析 ---")
        if pdf_metadata is not None:  # Check again, though we raise error if it fails now
            # --- Demand vs Metadata (Building Class)  ---
            logger.info("--- 开始 Demand vs building_class 分析 (Spark->Pandas) ---")
            analyze_demand_vs_metadata(sdf_demand, pdf_metadata, target_col='building_class', plots_dir=plots_dir, sample_frac=0.001)
            logger.info("--- 完成 Demand vs building_class 分析 ---")

            # --- Demand vs Metadata (Location)  ---
            logger.info("--- 开始 Demand vs location 分析 (Spark->Pandas) ---")
            analyze_demand_vs_location(sdf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001, top_n=5)
            logger.info("--- 完成 Demand vs location 分析 ---")

            # --- Demand vs Weather (保持运行) ---
            logger.info(
                "--- 开始 Demand vs Weather 分析 (Spark Join -> Pandas Collect) ---")
            analyze_demand_vs_weather(
                sdf_demand, pdf_metadata, sdf_weather, spark, plots_dir, n_sample_ids=50)
            logger.info("--- 完成 Demand vs Weather 分析 ---")
        else:
            logger.error("跳过关系分析，因为 Metadata Pandas DataFrame 不可用。") # Should not happen due to earlier check

        # --- 步骤 4: 时间特征分析 ---
        logger.info("--- 步骤 4: 时间特征分析 (Spark) ---")
        analyze_timestamp_consistency(sdf_demand, sdf_weather)
        analyze_datetime_features_spark(sdf_demand, plots_dir=plots_dir)
        logger.info("--- 完成 时间特征分析 ---")

        # --- 步骤 5: 数据质量检查总结 ---
        logger.info("--- 步骤 5: 数据质量检查总结 (Spark) ---")
        check_missing_values_spark(sdf_demand, "Demand")
        check_missing_values_spark(sdf_metadata, "Metadata")
        check_missing_values_spark(sdf_weather, "Weather")
        check_duplicates_spark(sdf_demand, ["unique_id", "timestamp"], "Demand")
        check_duplicates_spark(sdf_metadata, ["unique_id"], "Metadata")
        check_duplicates_spark(sdf_weather, ["location_id", "timestamp"], "Weather")
        logger.info("--- 完成 数据质量检查 ---")

        logger.info("=" * 40)
        logger.info("=== EDA (包含补充分析) 执行成功完成 ===") # Updated message
        logger.info("=" * 40)

    except Exception as e:
        logger.critical(f"执行过程中发生严重错误: {e}")
        logger.exception("Traceback:")  # Log full traceback
    finally:
        if spark:  # Check if spark session was initialized
            try:
                if not spark.sparkContext._jsc.sc().isStopped():
                    logger.info("正在停止 SparkSession...")
                    spark.stop()
                    logger.info("SparkSession 已停止。")
                else:
                    logger.info("SparkSession 已停止。")
            except Exception as stop_e:
                logger.error(f"停止 SparkSession 时发生错误: {stop_e}")
        else:
            logger.info("SparkSession 未成功初始化或已停止。")

        end_run_time = time.time()
        logger.info(
            f"--- EDA 脚本总执行时间: {end_run_time - start_run_time:.2f} 秒 ---")


if __name__ == "__main__":
    run_all_eda() # 取消注释以运行
