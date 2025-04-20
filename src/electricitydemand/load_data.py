import sys
import os
from loguru import logger # 提前导入 logger
import dask.dataframe as dd # Dask 用于加载数据

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
# 设置日志级别为 INFO
setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
logger.info(f"项目根目录: {project_root}")
logger.info(f"日志目录: {logs_dir}")

# --- 数据文件路径 ---
data_dir = os.path.join(project_root, "data") # 使用 project_root 确保路径正确
demand_path = os.path.join(data_dir, "demand.parquet")
metadata_path = os.path.join(data_dir, "metadata.parquet")
weather_path = os.path.join(data_dir, "weather.parquet")
logger.info(f"数据目录: {data_dir}")

# --- 加载数据 ---
logger.info("开始加载数据集...")

# 使用 Dask 加载数据
try:
    ddf_demand = dd.read_parquet(demand_path)
    logger.info(f"成功加载 Demand 数据: {demand_path}")

    ddf_metadata = dd.read_parquet(metadata_path)
    logger.info(f"成功加载 Metadata 数据: {metadata_path}")

    ddf_weather = dd.read_parquet(weather_path)
    logger.info(f"成功加载 Weather 数据: {weather_path}")

    logger.info("所有数据集加载完成。")

    # 打印各数据集的列名和分区数，初步了解结构
    logger.info(f"Demand Dask DataFrame npartitions: {ddf_demand.npartitions}")
    logger.info(f"Demand columns: {ddf_demand.columns}")

    logger.info(f"Metadata Dask DataFrame npartitions: {ddf_metadata.npartitions}")
    logger.info(f"Metadata columns: {ddf_metadata.columns}")

    logger.info(f"Weather Dask DataFrame npartitions: {ddf_weather.npartitions}")
    logger.info(f"Weather columns: {ddf_weather.columns}")

except FileNotFoundError as e:
    logger.error(f"数据文件未找到: {e}. 请先运行 download_data.py 或检查路径。") # 添加提示
except Exception as e:
    logger.exception(f"加载数据时发生未预期错误: {e}") # 使用 logger.exception 记录堆栈信息

# --- 1. 获取行数 ---
logger.info("--- 计算行数 ---")
# Demand (可能是近似值)
num_demand_rows = len(ddf_demand)
logger.info(f"Demand 数据行数 (估算): {num_demand_rows:,}")

# Metadata (可能是精确值，因为分区少)
num_metadata_rows = len(ddf_metadata)
logger.info(f"Metadata 数据行数: {num_metadata_rows:,}")

# Weather (可能是精确值，因为分区少)
num_weather_rows = len(ddf_weather)
logger.info(f"Weather 数据行数: {num_weather_rows:,}")


# --- 2. 查看数据样本 ---
logger.info("--- 查看数据样本 (前 5 行) ---")
# 使用 to_string() 避免日志截断或格式混乱
logger.info(f"Demand head:\n{ddf_demand.head().to_string()}")
logger.info(f"Metadata head:\n{ddf_metadata.head().to_string()}")
logger.info(f"Weather head:\n{ddf_weather.head().to_string()}")


# --- 3. 查看数据类型 ---
logger.info("--- 查看数据类型 ---")
# 转换为字符串以便完整记录
logger.info(f"Demand dtypes:\n{ddf_demand.dtypes.to_string()}")
logger.info(f"Metadata dtypes:\n{ddf_metadata.dtypes.to_string()}")
logger.info(f"Weather dtypes:\n{ddf_weather.dtypes.to_string()}")


# --- 4. 检查缺失值 ---
logger.info("--- 检查缺失值 ---")

# Demand 缺失值
logger.info("Demand 缺失值统计:")
missing_demand = ddf_demand.isnull().sum().compute()
logger.info(f"\n{missing_demand.to_string()}") # 转换为字符串
logger.info(f"Demand 总行数 (估算): {num_demand_rows:,}") # 参考总行数

# Metadata 缺失值
logger.info("Metadata 缺失值统计:")
missing_metadata = ddf_metadata.isnull().sum().compute()
logger.info(f"\n{missing_metadata.to_string()}")
logger.info(f"Metadata 总行数: {num_metadata_rows:,}")

# Weather 缺失值
logger.info("Weather 缺失值统计:")
missing_weather = ddf_weather.isnull().sum().compute()
logger.info(f"\n{missing_weather.to_string()}")
logger.info(f"Weather 总行数: {num_weather_rows:,}")

# 计算缺失比例
logger.info("--- 计算缺失值比例 ---")
if num_demand_rows > 0: # 使用之前计算的行数
    logger.info(f"Demand 缺失值比例:\n{(missing_demand / num_demand_rows).to_string()}")
if num_metadata_rows > 0:
    logger.info(f"Metadata 缺失值比例:\n{(missing_metadata / num_metadata_rows).to_string()}")
if num_weather_rows > 0:
    logger.info(f"Weather 缺失值比例:\n{(missing_weather / num_weather_rows).to_string()}")
