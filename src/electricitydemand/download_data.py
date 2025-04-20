import sys
import os
from loguru import logger # 提前导入 logger

# --- 项目设置 (路径和日志) ---
project_root = None
try:
    # 尝试标准的相对导入 (当作为包运行时)
    if __package__ and __package__.startswith('src.'):
        _script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_path)))
        from .utils.log_utils import setup_logger # 相对导入
    else:
        raise ImportError("Not running as a package or package structure mismatch.")

except (ImportError, ValueError, AttributeError, NameError):
    # 直接运行脚本或环境特殊的 fallback 逻辑
    try:
        _script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_path)))
    except NameError: # 如果 __file__ 未定义 (例如, 交互式环境)
        project_root = os.getcwd()

    # 如果是直接运行，将项目根目录添加到 sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 现在绝对导入应该可用
    from src.electricitydemand.utils.log_utils import setup_logger

# --- 配置日志 ---
log_prefix = os.path.splitext(os.path.basename(__file__))[0] # 从文件名自动获取前缀
logs_dir = os.path.join(project_root, 'logs')
setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir)
logger.info(f"项目根目录: {project_root}")
logger.info(f"日志目录: {logs_dir}")

# --- 标准模块导入 ---
from datasets import load_dataset
import dask.dataframe as dd # Dask 用于验证
from datetime import datetime
import pyarrow as pa # PyArrow 用于捕获特定错误类型

# --- 配置 ---
DATASET_NAME = "EDS-lab/electricity-demand"
CONFIGS = ["demand", "metadata", "weather"]
# 使用 project_root 确保数据目录路径正确
DATA_DIR = os.path.join(project_root, "data")
logger.info(f"数据目录: {DATA_DIR}")


# --- Helper Functions ---

def validate_parquet_file(filepath: str) -> bool:
    """
    Tries to read the schema or head of a Parquet file to validate it.

    Args:
        filepath: Path to the Parquet file.

    Returns:
        True if the file seems valid, False otherwise.
    """
    try:
        # Attempt to read a small part or just metadata.
        # Reading head is a reasonable quick check for Dask compatibility.
        dd.read_parquet(filepath).head(1)
        logger.debug(f"File validation successful for: {filepath}")
        return True
    except (FileNotFoundError, pa.lib.ArrowInvalid, ValueError, Exception) as e:
        # Catch common errors indicating corruption or incompatibility
        logger.warning(f"Validation failed for existing file '{filepath}': {e}. Will re-download.")
        return False

def download_and_save_config(config_name: str, dataset_id: str, output_dir: str, overwrite: bool = False):
    """
    Downloads a specific configuration from Hugging Face Datasets,
    validates existing files, and saves it as a Parquet file.

    Args:
        config_name: The name of the dataset configuration (e.g., 'demand').
        dataset_id: The Hugging Face dataset identifier.
        output_dir: The directory to save the Parquet file.
        overwrite: If True, always download and overwrite existing files.
    """
    output_filename = os.path.join(output_dir, f"{config_name}.parquet")
    logger.info(f"--- 处理配置: {config_name} ---") # Changed comment to Chinese

    # 检查文件是否存在，是否需要验证或覆盖
    if not overwrite and os.path.exists(output_filename):
        logger.info(f"文件 '{output_filename}' 已存在。正在验证...")
        if validate_parquet_file(output_filename):
            logger.info(f"现有文件 '{output_filename}' 有效。跳过下载。")
            return # 如果有效则跳过下载
        else:
            # 文件存在但无效，继续下载/覆盖
            pass
    elif overwrite and os.path.exists(output_filename):
         logger.info(f"设置了覆盖标志。重新下载 '{output_filename}'。")


    # 下载逻辑
    try:
        logger.info(f"正在从 '{dataset_id}' 加载/下载配置 '{config_name}'...")
        # 使用 streaming=True 可能最初会节省内存，
        # 但我们需要完整数据来保存为 Parquet。
        dataset_split = load_dataset(dataset_id, config_name, trust_remote_code=True)

        # 提取实际的 Dataset 对象
        if config_name in dataset_split:
            actual_data = dataset_split[config_name]
        elif 'train' in dataset_split: # HuggingFace 通常使用 'train' 作为默认 split 名称
            actual_data = dataset_split['train']
        else: # 如果找不到明确的 split 名称，尝试第一个可用的
            keys = list(dataset_split.keys())
            logger.warning(f"无法确定配置 '{config_name}' 的 split 名称，可用 keys: {keys}。使用第一个 key '{keys[0]}'.")
            actual_data = dataset_split[keys[0]]

        logger.info(f"配置 '{config_name}' 已加载。正在保存到 '{output_filename}'...")
        # 确保目录存在
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        actual_data.to_parquet(output_filename)
        logger.info(f"配置 '{config_name}' 保存成功。")

    except Exception as e:
        logger.exception(f"下载或保存配置 '{config_name}' 失败")


# --- Main Execution ---

def main():
    """主函数，下载所有数据集配置。"""
    logger.info(f"开始数据集下载流程 '{DATASET_NAME}'")
    logger.info(f"目标目录: {DATA_DIR}")

    try:
        # 确保数据目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.info(f"确保数据目录存在: {DATA_DIR}")

        # 处理每个配置
        for config in CONFIGS:
            # 如果总是想重新下载，设置 overwrite=True
            download_and_save_config(config, DATASET_NAME, DATA_DIR, overwrite=False)

        logger.info("--- 数据集下载流程结束 ---")

    except Exception as e:
        logger.exception("主下载流程中发生错误")

if __name__ == "__main__":
    main() 