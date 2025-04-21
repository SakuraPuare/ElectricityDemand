import sys
import os
import dask.dataframe as dd
from loguru import logger

# --- 项目设置 ---
# Determine project root dynamically
try:
    _script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(_script_path)))
except NameError:  # If __file__ is not defined (e.g., interactive)
    project_root = os.getcwd()
    # Add project root to path if running interactively might be needed
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Add src directory to sys.path to allow absolute imports from src
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- 配置日志 ---
try:
    from electricitydemand.utils.log_utils import setup_logger
except ImportError:
    print("Error: Could not import setup_logger. Ensure src is in PYTHONPATH or script is run correctly.", file=sys.stderr)
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

log_prefix = os.path.splitext(os.path.basename(__file__))[
    0]  # Use run_preprocessing as prefix
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

if 'setup_logger' in globals():
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")

logger.info(f"项目根目录：{project_root}")
logger.info(f"日志目录：{logs_dir}")

# --- 数据文件路径 ---
data_dir = os.path.join(project_root, "data")
demand_path = os.path.join(data_dir, "demand.parquet")
# metadata_path = os.path.join(data_dir, "metadata.parquet") # Not needed yet
# weather_path = os.path.join(data_dir, "weather.parquet") # Not needed yet
logger.info(f"数据目录：{data_dir}")

# --- 导入所需函数 ---
try:
    from electricitydemand.eda.load_data import load_demand_data  # Need the loader
    # Import preprocessing functions
    from electricitydemand.preprocessing import resample_demand_to_hourly, validate_resampling
except ImportError as e:
    logger.exception(f"Failed to import necessary modules: {e}")
    sys.exit(1)


def run_demand_resampling():
    """Loads demand data and resamples it to hourly frequency."""
    logger.info("=========================================")
    logger.info("===     开始执行 Demand 数据重采样脚本   ===")
    logger.info("=========================================")

    # --- 加载数据 ---
    logger.info("--- 步骤 1: 加载 Demand 数据 ---")
    ddf_demand = load_demand_data(demand_path)

    if ddf_demand is None:
        logger.error("未能加载 Demand 数据文件。终止处理。")
        return  # Exit the function

    # --- 数据预处理：重采样 ---
    logger.info("--- 步骤 2: Demand 数据重采样 ---")
    ddf_demand_hourly = resample_demand_to_hourly(ddf_demand)

    # --- 验证重采样结果 ---
    if ddf_demand_hourly is not None:
        logger.info("--- 步骤 3: 验证重采样结果 ---")
        # Check first 10 records
        validate_resampling(ddf_demand_hourly, n_check=10)

        # --- (可选) 保存重采样后的数据 ---
        output_path = os.path.join(data_dir, "demand_hourly.parquet")
        logger.info(f"--- 步骤 4: 保存重采样后的数据到 {output_path} ---")
        try:
            # Consider using overwrite=True or compute() before saving depending on workflow
            # ddf_demand_hourly.to_parquet(output_path, overwrite=True, engine='pyarrow')
            # logger.success(f"成功保存重采样数据到：{output_path}")
            logger.warning("保存重采样数据步骤已注释掉。如果需要，请取消注释。")
        except Exception as e:
            logger.exception(f"保存重采样数据时出错：{e}")

    else:
        logger.error("Demand 数据重采样失败，无法进行验证或保存。")

    logger.info("=========================================")
    logger.info("===     Demand 数据重采样脚本执行完毕   ===")
    logger.info("=========================================")


if __name__ == "__main__":
    try:
        run_demand_resampling()
    except Exception as e:
        logger.exception(f"执行过程中发生错误：{e}")
        sys.exit(1)
