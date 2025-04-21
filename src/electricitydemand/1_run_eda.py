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
from pyspark.storagelevel import StorageLevel  # Import StorageLevel

# --- 项目设置 ---
# 使用工具函数
try:
    from electricitydemand.utils.project_utils import get_project_root, setup_project_paths, create_spark_session, stop_spark_session
    from electricitydemand.utils.log_utils import setup_logger  # 仍然直接导入 setup_logger
except ImportError as e:
    print(f"Error importing project utils: {e}", file=sys.stderr)
    sys.exit(1)

project_root = get_project_root()
src_path, data_dir, logs_dir, plots_dir = setup_project_paths(project_root)

# --- 配置日志 ---
log_prefix = Path(__file__).stem  # Use run_eda as prefix
try:
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
except NameError:  # If setup_logger wasn't imported successfully
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

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
        analyze_demand_vs_dataset
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
        spark = create_spark_session(
            app_name="ElectricityDemandEDA",
            driver_mem_ratio=0.4,  # 保持原比例
            executor_mem_ratio=0.4,
            default_mem_gb=2  # 保持原默认值
        )

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 步骤 1: 加载数据 ---
        logger.info("--- 步骤 1: 加载数据 (Spark) ---")
        sdf_demand, sdf_metadata, sdf_weather = load_datasets(spark, data_dir)

        if sdf_demand is None or sdf_metadata is None or sdf_weather is None:
            raise ValueError("一个或多个数据集加载失败，请检查路径和文件。")
        logger.info("数据集加载成功。")
        # Persist loaded dataframes as they are used multiple times
        sdf_demand.persist(StorageLevel.MEMORY_AND_DISK)
        sdf_metadata.persist(StorageLevel.MEMORY_AND_DISK)
        sdf_weather.persist(StorageLevel.MEMORY_AND_DISK)
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
        analyze_demand_y_distribution(
            sdf_demand, plots_dir=plots_dir, sample_frac=0.005)
        analyze_demand_timeseries_sample(
            sdf_demand, plots_dir=plots_dir, n_samples=3)
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
            # Raise error if conversion fails
            raise ValueError("Failed to convert Metadata to Pandas.")

        if pdf_metadata is not None:
            analyze_metadata_categorical(pdf_metadata)  # Log counts
            plot_metadata_categorical(
                pdf_metadata, plots_dir=plots_dir)  # Plot distributions
            # Analyze/Plot numerical
            analyze_metadata_numerical(pdf_metadata, plots_dir=plots_dir)
            # Analyze missing location info
            analyze_missing_locations(pdf_metadata)
            logger.info("--- 完成 Metadata 分析 ---")
        else:
            # This block should not be reached if the raise above works
            logger.error("Metadata 转换为 Pandas DataFrame 失败，无法进行后续分析。")
            raise ValueError("Failed to convert Metadata to Pandas.")

        # --- Weather 分析  ---
        logger.info("--- 开始 Weather 分析 (Spark) ---")
        # 假设这些函数已适配 Spark 或能处理 Spark DF
        analyze_weather_numerical(
            sdf_weather, plots_dir=plots_dir, plot_sample_frac=0.05)
        analyze_weather_categorical(sdf_weather, plots_dir=plots_dir)
        analyze_weather_timeseries_sample(
            sdf_weather, plots_dir=plots_dir, n_samples=3)
        analyze_weather_correlation(sdf_weather, plots_dir=plots_dir)
        logger.info("--- 完成 Weather 分析 (Spark) ---")

        # --- 步骤 3: 关系分析 (全部运行) ---
        logger.info("--- 步骤 3: 关系分析 ---")
        if pdf_metadata is not None:  # Check again, though we raise error if it fails now
            # --- Demand vs Metadata (Building Class)  ---
            logger.info(
                "--- 开始 Demand vs building_class 分析 (Spark->Pandas) ---")
            analyze_demand_vs_metadata(
                sdf_demand, pdf_metadata, target_col='building_class', plots_dir=plots_dir, sample_frac=0.001)
            logger.info("--- 完成 Demand vs building_class 分析 ---")

            # --- Demand vs Metadata (Location)  ---
            logger.info("--- 开始 Demand vs location 分析 (Spark->Pandas) ---")
            analyze_demand_vs_location(
                sdf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001, top_n=5)
            logger.info("--- 完成 Demand vs location 分析 ---")

            # --- Demand vs Metadata (Dataset) ---
            logger.info("--- 开始 Demand vs dataset 分析 (Spark->Pandas) ---")
            analyze_demand_vs_dataset(
                sdf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001)
            logger.info("--- 完成 Demand vs dataset 分析 ---")

            # --- Demand vs Weather (保持运行) ---
            logger.info(
                "--- 开始 Demand vs Weather 分析 (Spark Join -> Pandas Collect) ---")
            analyze_demand_vs_weather(
                sdf_demand,
                pdf_metadata,
                sdf_weather,
                spark,
                plots_dir=plots_dir,
                n_sample_ids=50)
            logger.info("--- 完成 Demand vs Weather 分析 ---")
        else:
            # Should not happen due to earlier check
            logger.error("跳过关系分析，因为 Metadata Pandas DataFrame 不可用。")

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
        check_duplicates_spark(
            sdf_demand, ["unique_id", "timestamp"], "Demand")
        check_duplicates_spark(sdf_metadata, ["unique_id"], "Metadata")
        check_duplicates_spark(
            sdf_weather, ["location_id", "timestamp"], "Weather")
        logger.info("--- 完成 数据质量检查 ---")

        logger.info("=" * 40)
        logger.info("=== EDA (包含补充分析) 执行成功完成 ===")  # Updated message
        logger.info("=" * 40)

    except Exception as e:
        logger.critical(f"执行过程中发生严重错误: {e}")
        logger.exception("Traceback:")  # Log full traceback
    finally:
        # Unpersist dataframes
        if 'sdf_demand' in locals() and sdf_demand.is_cached:
            sdf_demand.unpersist()
        if 'sdf_metadata' in locals() and sdf_metadata.is_cached:
            sdf_metadata.unpersist()
        if 'sdf_weather' in locals() and sdf_weather.is_cached:
            sdf_weather.unpersist()
        stop_spark_session(spark)  # Use utility function to stop Spark

        end_run_time = time.time()
        logger.info(
            f"--- EDA 脚本总执行时间: {end_run_time - start_run_time:.2f} 秒 ---")


if __name__ == "__main__":
    run_all_eda()  # 取消注释以运行
