import sys
import os
# 移除 dask 和 pandas 的导入 (如果不再直接使用它们加载)
# import dask.dataframe as dd
# import pandas as pd
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
data_dir = os.path.join(project_root, "data")
demand_path = os.path.join(data_dir, "demand.parquet")
metadata_path = os.path.join(data_dir, "metadata.parquet")
weather_path = os.path.join(data_dir, "weather.parquet")
logger.info(f"数据目录: {data_dir}")

# --- 导入分析函数 (需要后续迁移) ---
try:
    # 修改导入: load_datasets 现在需要 SparkSession
    from electricitydemand.eda.load_data import load_datasets
    # 其他分析函数暂时保持，但需要迁移才能使用 Spark DataFrame
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
    from electricitydemand.eda.analyze_weather import (
        analyze_weather_numerical,
        analyze_weather_categorical,
    )
    from electricitydemand.eda.analyze_relationships import (
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
        spark = SparkSession.builder \
            .appName("ElectricityDemand_EDA") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
            .config("spark.sql.parquet.int64AsTimestampNanos", "true") \
            .getOrCreate()
        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 加载数据 (使用 Spark) ---
        logger.info("--- 步骤 1: 加载数据 (Spark) ---")
        # 调用更新后的 load_datasets 函数，传入 SparkSession
        ddf_demand, ddf_metadata, ddf_weather = load_datasets(spark)

        # Check if data loading was successful
        if ddf_demand is None or ddf_metadata is None or ddf_weather is None:
            logger.error("未能加载所有必需的数据文件。终止 EDA。")
            return # Exit the function

        # --- 缓存常用的 DataFrame (可选，如果内存允许) ---
        # logger.info("缓存 Demand 和 Weather DataFrame 以加速后续操作...")
        # ddf_demand.cache()
        # ddf_weather.cache()
        # ddf_demand.count() # 触发缓存计算
        # ddf_weather.count() # 触发缓存计算
        # logger.info("DataFrame 缓存完成。")

        # --- 单变量分析 (需要迁移) ---
        logger.info("--- 步骤 2: 单变量分析 (需要迁移到 Spark) ---")

        # Demand 'y' 分析 (需要迁移)
        logger.info("--- 开始 Demand 'y' 分析 (待迁移) ---")
        # 以下函数需要重写以接受 Spark DataFrame
        # y_sample_pd = analyze_demand_y_distribution(ddf_demand, sample_frac=0.005)
        # if y_sample_pd is not None and not y_sample_pd.empty:
        #      plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000)
        # analyze_demand_timeseries_sample(ddf_demand, n_samples=3, plots_dir=plots_dir)
        logger.warning("Demand 'y' 分析函数尚未迁移到 Spark，将跳过。")
        logger.info("--- 完成 Demand 'y' 分析 (跳过) ---")

        # Metadata 分析 (需要迁移)
        # Metadata 数据量不大，可以考虑 collect 到 Pandas 进行分析
        logger.info("--- 开始 Metadata 分析 (部分可使用 Pandas) ---")
        try:
            logger.info("将 Metadata Spark DataFrame 转换为 Pandas DataFrame (如果内存允许)...")
            pdf_metadata = ddf_metadata.toPandas()
            logger.info(f"Metadata Pandas DataFrame 转换成功，形状: {pdf_metadata.shape}")

            # 现在可以调用原来的基于 Pandas 的 Metadata 分析函数
            analyze_metadata_categorical(pdf_metadata)
            plot_metadata_categorical(pdf_metadata, plots_dir=plots_dir, top_n=10)
            analyze_metadata_numerical(pdf_metadata, plots_dir=plots_dir)
            analyze_missing_locations(pdf_metadata)
        except Exception as e:
             logger.exception("将 Metadata 转换为 Pandas 或进行分析时出错。")
             logger.warning("Metadata 分析函数 (基于 Pandas) 可能未完全执行。")
        logger.info("--- 完成 Metadata 分析 ---")


        # Weather 分析 (需要迁移)
        logger.info("--- 开始 Weather 分析 (待迁移) ---")
        # 以下函数需要重写以接受 Spark DataFrame
        # analyze_weather_numerical(ddf_weather, plots_dir=plots_dir, plot_sample_frac=0.05)
        # analyze_weather_categorical(ddf_weather, plots_dir=plots_dir, top_n=15)
        logger.warning("Weather 分析函数尚未迁移到 Spark，将跳过。")
        logger.info("--- 完成 Weather 分析 (跳过) ---")


        # --- 关系分析 --- (需要迁移)
        logger.info("--- 步骤 3: 关系分析 (待迁移) ---")

        # Demand vs Metadata (building_class) (需要迁移)
        # logger.info("--- 开始 Demand vs building_class 分析 (待迁移) ---")
        # analyze_demand_vs_metadata(ddf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001) # pdf_metadata 已是 Pandas
        # logger.info("--- 完成 Demand vs building_class 分析 (跳过) ---")

        # Demand vs Metadata (location) (需要迁移)
        # logger.info("--- 开始 Demand vs location 分析 (待迁移) ---")
        # analyze_demand_vs_location(ddf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001, top_n=5) # pdf_metadata 已是 Pandas
        # logger.info("--- 完成 Demand vs location 分析 (跳过) ---")

        # Demand vs Weather (需要迁移)
        logger.info("--- 开始 Demand vs Weather 分析 (待迁移) ---")
        # try:
        #     analyze_demand_vs_weather(ddf_demand, pdf_metadata, ddf_weather, plots_dir=plots_dir, n_sample_ids=50) # pdf_metadata 已是 Pandas
        # except Exception as e:
        #      logger.error(f"Demand vs Weather 分析执行期间遇到问题: {e}")
        logger.warning("Demand vs Weather 分析函数尚未迁移到 Spark，将跳过。")
        logger.info("--- 完成 Demand vs Weather 分析 (跳过) ---")


        logger.info("=========================================")
        logger.info("===     Spark EDA 脚本初步执行完毕     ===")
        logger.info("(注意: 大部分分析函数需要迁移才能运行)")
        logger.info("=========================================")

    except Exception as e:
        logger.exception(f"执行过程中发生错误: {e}")
        if spark: # 确保即使出错也尝试停止 Spark
            logger.info("正在停止 SparkSession...")
            spark.stop()
        sys.exit(1)
    finally:
        if spark:
            logger.info("正在停止 SparkSession...")
            spark.stop()
            logger.info("SparkSession 已停止。")


if __name__ == "__main__":
    try:
        run_all_eda()
    except Exception as e:
        # logger 已在 run_all_eda 的 finally 中处理
        pass # 避免重复打印
