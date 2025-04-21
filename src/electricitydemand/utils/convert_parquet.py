import sys
import os
import pandas as pd
from loguru import logger

# --- 项目根目录和日志设置 ---
try:
    _script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_path)))
except NameError: # Fallback for interactive use
    project_root = os.getcwd()

# Add src directory to sys.path if necessary (if not running as module)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    # Also add project root for sibling imports if needed
    if project_root not in sys.path:
         sys.path.insert(0, project_root)


# 相对导入日志设置 (假设 log_utils 在 src/electricitydemand/utils)
try:
    from electricitydemand.utils.log_utils import setup_logger
except ImportError:
     # Basic fallback logger if setup_logger fails
    logger.add(sys.stderr, level="INFO")
    logger.warning("Failed to import setup_logger, using basic stderr logging.")

log_prefix = os.path.splitext(os.path.basename(__file__))[0]
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Set up logger (if imported)
if 'setup_logger' in globals():
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")

logger.info(f"项目根目录: {project_root}")
logger.info(f"日志目录: {logs_dir}")

# --- 文件路径定义 ---
data_dir = "data"
files_to_convert = [
    {"input": "demand.parquet", "output": "demand_converted.parquet", "timestamp_col": "timestamp"},
    {"input": "weather.parquet", "output": "weather_converted.parquet", "timestamp_col": "timestamp"},
    # {"input": "metadata.parquet", "output": "metadata_converted.parquet", "timestamp_col": None}, # Metadata 可能不需要转换
]
logger.info(f"数据目录: {data_dir}")

def convert_parquet_file(input_filename: str, output_filename: str, timestamp_col: str | None = 'timestamp', engine_preference: list = ['pyarrow', 'fastparquet']):
    """
    Reads a Parquet file using preferred engines, adjusts timestamp,
    and writes it back using pyarrow with compatible settings.

    Args:
        input_filename: Path to the input Parquet file.
        output_filename: Path to save the converted Parquet file.
        timestamp_col: Name of the timestamp column to convert, or None.
        engine_preference: List of engines to try for reading ('pyarrow', 'fastparquet').
    """
    logger.info(f"--- 开始转换: {os.path.basename(input_filename)} ---")
    df = None
    read_success = False

    # 1. 尝试读取 Parquet 文件
    for engine in engine_preference:
        logger.info(f"尝试使用引擎 '{engine}' 读取...")
        try:
            df = pd.read_parquet(input_filename, engine=engine)
            logger.success(f"成功使用引擎 '{engine}' 读取文件: {input_filename}")
            read_success = True
            break # 读取成功，跳出循环
        except Exception as e:
            logger.warning(f"使用引擎 '{engine}' 读取失败: {e}")
            continue # 尝试下一个引擎

    if not read_success or df is None:
        logger.error(f"无法使用任何指定引擎读取文件: {input_filename}")
        return False # 返回失败状态

    # 2. (可选) 调整时间戳类型
    if timestamp_col and timestamp_col in df.columns:
        logger.info(f"转换列 '{timestamp_col}' 为 datetime 对象...")
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            logger.info(f"列 '{timestamp_col}' 转换成功。")
        except Exception as e:
            logger.error(f"转换列 '{timestamp_col}' 时出错: {e}")
            logger.warning("将继续尝试写入，但时间戳可能不是预期类型。")
    elif timestamp_col:
         logger.warning(f"指定的时间戳列 '{timestamp_col}' 不在 DataFrame 中。")


    # 3. 写回 Parquet 文件 (使用 pyarrow)
    logger.info(f"尝试将转换后的数据写入: {output_filename}")
    try:
        df.to_parquet(
            output_filename,
            engine='pyarrow',
            index=False,
            coerce_timestamps='ms', # 写入毫秒级时间戳，提高兼容性
            allow_truncated_timestamps=False, # 不允许截断
            # use_deprecated_int96_timestamps=False # 明确不使用旧的int96 (如果需要写给旧系统可以设为True)
            # compression='snappy' # 可以指定压缩方式
        )
        logger.success(f"成功转换并保存文件: {output_filename}")
        return True # 返回成功状态
    except Exception as e:
        logger.exception(f"写入 Parquet 文件 '{output_filename}' 时发生错误")
        return False # 返回失败状态

def main():
    """主函数，循环处理需要转换的文件。"""
    logger.info("=========================================")
    logger.info("===    开始执行 Parquet 转换脚本      ===")
    logger.info("=========================================")

    all_successful = True
    for file_info in files_to_convert:
        input_path = os.path.join(data_dir, file_info["input"])
        output_path = os.path.join(data_dir, file_info["output"])
        timestamp_col = file_info.get("timestamp_col") # 使用 .get() 以防 key 不存在

        if not os.path.exists(input_path):
            logger.error(f"输入文件未找到，跳过: {input_path}")
            all_successful = False
            continue

        success = convert_parquet_file(input_path, output_path, timestamp_col)
        if not success:
            all_successful = False
            logger.error(f"文件 '{file_info['input']}' 转换失败。")

    logger.info("=========================================")
    if all_successful:
        logger.success("===    Parquet 转换脚本执行完毕 (全部成功) ===")
    else:
        logger.error("===    Parquet 转换脚本执行完毕 (部分失败) ===")
    logger.info("=========================================")

if __name__ == "__main__":
    main()