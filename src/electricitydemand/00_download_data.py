from pathlib import Path

import dask.dataframe as dd  # Dask 用于验证
import pyarrow as pa  # PyArrow 用于捕获特定错误类型
from datasets import load_dataset
from loguru import logger  # 提前导入 logger

from electricitydemand.utils.log_utils import setup_logger

# --- 项目设置 (使用工具函数) ---
# project_utils 应该可以被导入，因为它在 src 目录下，而 setup_project_paths 会将 src 加入 sys.path
from electricitydemand.utils.project_utils import get_project_root, setup_project_paths

project_root = get_project_root()
src_path, data_dir, logs_dir, plots_dir = setup_project_paths(
    project_root)  # plots_dir 在这里未使用

# --- 配置日志 ---
log_prefix = Path(__file__).stem  # 从文件名自动获取前缀
setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir)
logger.info(f"项目根目录：{project_root}")
logger.info(f"日志目录：{logs_dir}")

# --- 标准模块导入 ---

# --- 配置 ---
DATASET_NAME = "EDS-lab/electricity-demand"
CONFIGS = ["demand", "metadata", "weather"]
# 使用 project_root 确保数据目录路径正确
logger.info(f"数据目录：{data_dir}")


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
        logger.warning(
            f"Validation failed for existing file '{filepath}': {e}. Will re-download.")
        return False


def download_and_save_config(config_name: str, dataset_id: str, output_dir: Path, overwrite: bool = False):
    """
    Downloads a specific configuration from Hugging Face Datasets,
    validates existing files, and saves it as a Parquet file.

    Args:
        config_name: The name of the dataset configuration (e.g., 'demand').
        dataset_id: The Hugging Face dataset identifier.
        output_dir: The directory to save the Parquet file.
        overwrite: If True, always download and overwrite existing files.
    """
    output_filename = output_dir / f"{config_name}.parquet"
    logger.info(f"--- 处理配置：{config_name} ---")  # Changed comment to Chinese

    # 检查文件是否存在，是否需要验证或覆盖
    if not overwrite and output_filename.exists():
        logger.info(f"文件 '{output_filename}' 已存在。正在验证...")
        if validate_parquet_file(str(output_filename)):
            logger.info(f"现有文件 '{output_filename}' 有效。跳过下载。")
            return  # 如果有效则跳过下载
        else:
            # 文件存在但无效，继续下载/覆盖
            pass
    elif overwrite and output_filename.exists():
        logger.info(f"设置了覆盖标志。重新下载 '{output_filename}'。")

    # 下载逻辑
    try:
        logger.info(f"正在从 '{dataset_id}' 加载/下载配置 '{config_name}'...")
        # 使用 streaming=True 可能最初会节省内存，
        # 但我们需要完整数据来保存为 Parquet。
        dataset_split = load_dataset(
            dataset_id, config_name, trust_remote_code=True)

        # 提取实际的 Dataset 对象
        if config_name in dataset_split:
            actual_data = dataset_split[config_name]
        elif 'train' in dataset_split:  # HuggingFace 通常使用 'train' 作为默认 split 名称
            actual_data = dataset_split['train']
        else:  # 如果找不到明确的 split 名称，尝试第一个可用的
            keys = list(dataset_split.keys())
            logger.warning(
                f"无法确定配置 '{config_name}' 的 split 名称，可用 keys: {keys}。使用第一个 key '{keys[0]}'.")
            actual_data = dataset_split[keys[0]]

        logger.info(f"配置 '{config_name}' 已加载。正在保存到 '{output_filename}'...")
        # 确保目录存在
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        actual_data.to_parquet(output_filename)
        logger.info(f"配置 '{config_name}' 保存成功。")

    except Exception as e:
        logger.exception(f"下载或保存配置 '{config_name}' 失败 {e}")


# --- Main Execution ---

def main():
    """主函数，下载所有数据集配置。"""
    logger.info(f"开始数据集下载流程 '{DATASET_NAME}'")
    logger.info(f"目标目录：{data_dir}")

    try:
        # 确保数据目录存在
        data_dir.mkdir(parents=True, exist_ok=True)  # 使用 pathlib
        logger.info(f"确保数据目录存在：{data_dir}")

        # 处理每个配置
        for config in CONFIGS:
            # 如果总是想重新下载，设置 overwrite=True
            download_and_save_config(
                config, DATASET_NAME, data_dir, overwrite=False)

        logger.info("--- 数据集下载流程结束 ---")

    except Exception as e:
        logger.exception(f"主下载流程中发生错误 {e}")


if __name__ == "__main__":
    main()
