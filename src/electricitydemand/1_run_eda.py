import sys
import os
# 移除 dask 和 pandas 的导入 (如果不再直接使用它们加载)
# import dask.dataframe as dd
# import pandas as pd
import pandas as pd # Pandas 仍然需要用于 Metadata 分析和部分绘图
from loguru import logger
# 引入 Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F # 常用函数别名

# --- 项目设置 ---
# Determine project root dynamically
try:
    _script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_path)))
except NameError: # If __file__ is not defined (e.g., interactive)
    project_root = os.getcwd()
    # Add project root to path if running interactively might be needed
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Add src directory to sys.path to allow absolute imports from src
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 配置日志 ---
# Assuming log_utils is in src/electricitydemand/utils
try:
    from electricitydemand.utils.log_utils import setup_logger
except ImportError:
    print("Error: Could not import setup_logger. Ensure src is in PYTHONPATH or script is run correctly.", file=sys.stderr)
    # Fallback basic logging if setup fails
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

log_prefix = os.path.splitext(os.path.basename(__file__))[0] # Use run_eda as prefix
logs_dir = os.path.join(project_root, 'logs')
plots_dir = os.path.join(project_root, 'plots') # 图表保存目录
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True) # 确保目录存在

# 设置日志级别为 INFO (only if setup_logger was imported)
if 'setup_logger' in globals():
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")

logger.info(f"项目根目录: {project_root}")
logger.info(f"日志目录: {logs_dir}")
logger.info(f"图表目录: {plots_dir}")


# --- 数据文件路径 ---
data_dir = os.path.join(project_root, "data") # 使用 project_root 确保路径正确
demand_path = os.path.join(data_dir, "demand_converted.parquet")
metadata_path = os.path.join(data_dir, "metadata.parquet")
weather_path = os.path.join(data_dir, "weather_converted.parquet")
logger.info(f"数据目录：{data_dir}")
# --- 导入分析函数 ---
try:
    # 导入已修改/确认的函数
    from electricitydemand.eda.load_data import load_datasets
    from electricitydemand.eda.analyze_demand import (
        analyze_demand_y_distribution,
        plot_demand_y_distribution,
        analyze_demand_timeseries_sample,
    )
    from electricitydemand.eda.analyze_metadata import (
        analyze_metadata_categorical,
        plot_metadata_categorical,
        analyze_metadata_numerical,
        analyze_missing_locations,
    )
    from electricitydemand.eda.analyze_weather import ( # 已迁移
        analyze_weather_numerical,
        analyze_weather_categorical,
    )
    from electricitydemand.eda.analyze_relationships import ( # 已迁移
        analyze_demand_vs_metadata,
        analyze_demand_vs_location,
        analyze_demand_vs_weather,
    )
except ImportError as e:
    logger.exception(f"Failed to import necessary EDA modules: {e}")
    sys.exit(1)


def run_all_eda():
    """主执行函数，编排 Spark EDA 步骤。"""
    logger.info("=========================================")
    logger.info("===     开始执行 Spark EDA 脚本       ===")
    logger.info("=========================================")

    spark = None # Initialize SparkSession variable
    try:
        # --- 创建 SparkSession ---
        logger.info("创建 SparkSession...")
        # 增加一些常用配置，特别是针对 Parquet 时间戳的处理
        spark = SparkSession.builder \
            .appName("ElectricityDemand_EDA_Spark") \
            .master("local[*]") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 加载数据 (使用 Spark) ---
        logger.info("--- 步骤 1: 加载数据 (Spark) ---")
        # 调用更新后的 load_datasets 函数，传入 SparkSession 和路径
        sdf_demand, sdf_metadata, sdf_weather = load_datasets(spark, demand_path, metadata_path, weather_path)

        # Check if data loading was successful
        if sdf_demand is None or sdf_metadata is None or sdf_weather is None:
            logger.error("未能加载所有必需的数据文件。终止 EDA。")
            return # Exit the function

        # --- 单变量分析 ---
        logger.info("--- 步骤 2: 单变量分析 ---")

        # Demand 分析 (使用 Spark DF)
        logger.info("--- 开始 Demand 分析 (Spark) ---")
        # 调用 analyze_demand_y_distribution (已迁移, 接收 Spark DF, 返回 Pandas Series)
        y_sample_pd = analyze_demand_y_distribution(sdf_demand, sample_frac=0.005)
        if y_sample_pd is not None and not y_sample_pd.empty:
             plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000) # 绘图接收 Pandas Series
        else:
             logger.warning("未能获取用于绘图的 Demand 'y' 样本数据。")

        # 调用 analyze_demand_timeseries_sample (已迁移, 接收 Spark DF)
        analyze_demand_timeseries_sample(sdf_demand, n_samples=3, plots_dir=plots_dir)
        logger.info("--- 完成 Demand 分析 (Spark) ---")

        # Metadata 分析 (将 Spark DF 转为 Pandas DF)
        logger.info("--- 开始 Metadata 分析 (Spark DF -> Pandas DF) ---")
        pdf_metadata = None
        try:
            logger.info("将 Metadata Spark DataFrame 转换为 Pandas DataFrame...")
            pdf_metadata = sdf_metadata.toPandas() # Action: Collects to driver
            logger.info(f"Metadata Pandas DataFrame 转换成功，形状: {pdf_metadata.shape}")

            # 调用基于 Pandas 的 Metadata 分析函数
            analyze_metadata_categorical(pdf_metadata)
            plot_metadata_categorical(pdf_metadata, plots_dir=plots_dir, top_n=10)
            analyze_metadata_numerical(pdf_metadata, plots_dir=plots_dir)
            analyze_missing_locations(pdf_metadata)
        except Exception as e:
             logger.exception("将 Metadata 转换为 Pandas 或进行分析时出错。检查驱动程序内存。")
             logger.warning("Metadata 分析函数 (基于 Pandas) 可能未完全执行。")
        logger.info("--- 完成 Metadata 分析 ---")


        # Weather 分析 (调用已迁移的函数)
        logger.info("--- 开始 Weather 分析 (Spark) ---")
        analyze_weather_numerical(sdf_weather, plots_dir=plots_dir, plot_sample_frac=0.05)
        analyze_weather_categorical(sdf_weather, plots_dir=plots_dir, top_n=15)
        # logger.warning("Weather 分析函数待迁移，暂时跳过。") # 注释掉旧的警告
        logger.info("--- 完成 Weather 分析 (Spark) ---")


        # --- 关系分析 --- (调用已迁移的函数)
        logger.info("--- 步骤 3: 关系分析 ---")

        if pdf_metadata is not None: # 确保 Metadata Pandas DF 存在
             # Demand vs Metadata (building_class)
             logger.info("--- 开始 Demand vs building_class 分析 (Spark->Pandas) ---")
             analyze_demand_vs_metadata(sdf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001)
             # logger.warning("Demand vs Metadata (building_class) 分析函数待迁移，暂时跳过。") # 注释掉旧的警告
             logger.info("--- 完成 Demand vs building_class 分析 ---")

             # Demand vs Metadata (location)
             logger.info("--- 开始 Demand vs location 分析 (Spark->Pandas) ---")
             analyze_demand_vs_location(sdf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001, top_n=5)
             # logger.warning("Demand vs Metadata (location) 分析函数待迁移，暂时跳过。") # 注释掉旧的警告
             logger.info("--- 完成 Demand vs location 分析 ---")

             # Demand vs Weather (需要传入 SparkSession)
             logger.info("--- 开始 Demand vs Weather 分析 (Spark->Pandas Merge) ---")
             analyze_demand_vs_weather(sdf_demand, pdf_metadata, sdf_weather, spark, plots_dir=plots_dir, n_sample_ids=50) # 传入 spark
             # logger.warning("Demand vs Weather 分析函数待迁移，暂时跳过。") # 注释掉旧的警告
             logger.info("--- 完成 Demand vs Weather 分析 ---")
        else:
             logger.error("Metadata Pandas DataFrame 未成功创建，跳过所有关系分析。")


        logger.info("=========================================")
        logger.info("===       Spark EDA 脚本执行完毕       ===")
        # logger.info("(注意: Weather 和 Relationships 分析函数需要迁移)") # 注释掉旧的警告
        logger.info("=========================================")

    except Exception as e:
        logger.exception(f"执行过程中发生严重错误: {e}")
        # Ensure Spark is stopped even on error
        if spark:
            logger.info("正在停止 SparkSession (因错误)...")
            try:
                spark.stop()
            except Exception as stop_e:
                logger.error(f"停止 SparkSession 时也发生错误: {stop_e}")
        sys.exit(1) # Exit with error code
    finally:
        # Ensure Spark is stopped in normal execution or after handled exception in try block
        if spark:
            logger.info("正在停止 SparkSession...")
            try:
                spark.stop()
                logger.info("SparkSession 已停止。")
            except Exception as stop_e:
                 logger.error(f"停止 SparkSession 时发生错误: {stop_e}")


if __name__ == "__main__":
    # try-except block here is less necessary as run_all_eda handles its exceptions
    run_all_eda()
