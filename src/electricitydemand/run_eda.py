import sys
import os
import dask.dataframe as dd
import pandas as pd
from loguru import logger

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

# --- 导入分析函数 ---
try:
    from electricitydemand.eda.load_data import load_demand_data, load_metadata, load_weather_data
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
    """主执行函数，编排 EDA 步骤。"""
    logger.info("=========================================")
    logger.info("===        开始执行 EDA 脚本          ===")
    logger.info("=========================================")

    # --- 加载数据 ---
    logger.info("--- 步骤 1: 加载数据 ---")
    ddf_demand = load_demand_data(demand_path)
    pdf_metadata = load_metadata(metadata_path)
    ddf_weather = load_weather_data(weather_path)

    # Check if data loading was successful
    if ddf_demand is None or pdf_metadata is None or ddf_weather is None:
         logger.error("未能加载所有必需的数据文件。终止 EDA。")
         return # Exit the function

    # --- 单变量分析 ---
    logger.info("--- 步骤 2: 单变量分析 ---")

    # Demand 'y' 分析
    logger.info("--- 开始 Demand 'y' 分析 ---")
    y_sample_pd = analyze_demand_y_distribution(ddf_demand, sample_frac=0.005)
    if y_sample_pd is not None and not y_sample_pd.empty:
         plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000)
    analyze_demand_timeseries_sample(ddf_demand, n_samples=3, plots_dir=plots_dir) # 减少样本量加快速度
    logger.info("--- 完成 Demand 'y' 分析 ---")


    # Metadata 分析
    logger.info("--- 开始 Metadata 分析 ---")
    analyze_metadata_categorical(pdf_metadata)
    plot_metadata_categorical(pdf_metadata, plots_dir=plots_dir, top_n=10)
    analyze_metadata_numerical(pdf_metadata, plots_dir=plots_dir)
    analyze_missing_locations(pdf_metadata)
    logger.info("--- 完成 Metadata 分析 ---")


    # Weather 分析
    logger.info("--- 开始 Weather 分析 ---")
    analyze_weather_numerical(ddf_weather, plots_dir=plots_dir, plot_sample_frac=0.05) # 减少抽样比例
    analyze_weather_categorical(ddf_weather, plots_dir=plots_dir, top_n=15)
    logger.info("--- 完成 Weather 分析 ---")


    # --- 关系分析 ---
    logger.info("--- 步骤 3: 关系分析 ---")

    # Demand vs Metadata (building_class)
    logger.info("--- 开始 Demand vs building_class 分析 ---")
    analyze_demand_vs_metadata(ddf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001)
    logger.info("--- 完成 Demand vs building_class 分析 ---")

    # Demand vs Metadata (location)
    logger.info("--- 开始 Demand vs location 分析 ---")
    analyze_demand_vs_location(ddf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001, top_n=5) # 减少 TopN
    logger.info("--- 完成 Demand vs location 分析 ---")

    # Demand vs Weather (可能跳过)
    logger.info("--- 开始 Demand vs Weather 分析 ---")
    try:
        analyze_demand_vs_weather(ddf_demand, pdf_metadata, ddf_weather, plots_dir=plots_dir, n_sample_ids=50)
    except Exception as e:
         # analyze_demand_vs_weather 内部已记录详细错误，这里只记录简要信息
         logger.error(f"Demand vs Weather 分析执行期间遇到问题: {e}")
    logger.info("--- 完成 Demand vs Weather 分析 ---")


    logger.info("=========================================")
    logger.info("===        EDA 脚本执行完毕           ===")
    logger.info("=========================================")


if __name__ == "__main__":
    run_all_eda()