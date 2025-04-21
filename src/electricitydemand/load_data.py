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
    except NameError: # 如果 __file__ 未定义 (例如，交互式环境)
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
logger.info(f"项目根目录：{project_root}")
logger.info(f"日志目录：{logs_dir}")

# --- 数据文件路径 ---
data_dir = os.path.join(project_root, "data") # 使用 project_root 确保路径正确
demand_path = os.path.join(data_dir, "demand.parquet")
metadata_path = os.path.join(data_dir, "metadata.parquet")
weather_path = os.path.join(data_dir, "weather.parquet")
logger.info(f"数据目录：{data_dir}")

def load_datasets():
    """加载 Demand, Metadata, 和 Weather 数据集."""
    logger.info("开始加载数据集...")
    try:
        ddf_demand = dd.read_parquet(demand_path)
        logger.info(f"成功加载 Demand 数据：{demand_path}")

        ddf_metadata = dd.read_parquet(metadata_path)
        logger.info(f"成功加载 Metadata 数据：{metadata_path}")

        ddf_weather = dd.read_parquet(weather_path)
        logger.info(f"成功加载 Weather 数据：{weather_path}")

        logger.info("所有数据集加载完成。")
        return ddf_demand, ddf_metadata, ddf_weather
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到：{e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载数据集时发生错误：{e}")
        sys.exit(1)

def log_basic_info(ddf_demand, ddf_metadata, ddf_weather):
    """记录数据集的基本信息：分区数、列名、行数、样本数据和数据类型。"""
    logger.info("--- 记录数据集基本信息 ---")

    # 分区数和列名
    logger.info(f"Demand Dask DataFrame npartitions: {ddf_demand.npartitions}, columns: {ddf_demand.columns}")
    logger.info(f"Metadata Dask DataFrame npartitions: {ddf_metadata.npartitions}, columns: {ddf_metadata.columns}")
    logger.info(f"Weather Dask DataFrame npartitions: {ddf_weather.npartitions}, columns: {ddf_weather.columns}")

    # 计算行数 (Dask 的 len() 可能返回估算值)
    logger.info("--- 计算行数 ---")
    # 使用 persist() 可能有助于在后续计算中重用内存中的行数结果，但会增加内存使用
    # num_demand_rows = len(ddf_demand.persist()) # 示例：如果需要频繁使用 len
    num_demand_rows = len(ddf_demand)
    num_metadata_rows = len(ddf_metadata)
    num_weather_rows = len(ddf_weather)
    logger.info(f"Demand 数据行数 (估算): {num_demand_rows:,}")
    logger.info(f"Metadata 数据行数：{num_metadata_rows:,}")
    logger.info(f"Weather 数据行数：{num_weather_rows:,}")

    # 查看数据样本
    logger.info("--- 查看数据样本 (前 5 行) ---")
    # 使用 compute() 获取实际数据，head() 默认获取前 5 行
    logger.info(f"Demand head:\n{ddf_demand.head().to_string()}")
    logger.info(f"Metadata head:\n{ddf_metadata.head().to_string()}")
    logger.info(f"Weather head:\n{ddf_weather.head().to_string()}")

    # 查看数据类型
    logger.info("--- 查看数据类型 ---")
    logger.info(f"Demand dtypes:\n{ddf_demand.dtypes.to_string()}")
    logger.info(f"Metadata dtypes:\n{ddf_metadata.dtypes.to_string()}")
    logger.info(f"Weather dtypes:\n{ddf_weather.dtypes.to_string()}")

    return num_demand_rows, num_metadata_rows, num_weather_rows # 返回行数供后续使用

def check_missing_values(ddf_demand, ddf_metadata, ddf_weather, num_demand_rows, num_metadata_rows, num_weather_rows):
    """检查并记录每个数据帧中的缺失值及其比例。"""
    logger.info("--- 检查缺失值 ---")

    # Demand 缺失值
    logger.info("Demand 缺失值统计：")
    missing_demand = ddf_demand.isnull().sum().compute()
    logger.info(f"\n{missing_demand.to_string()}")
    if num_demand_rows > 0:
        logger.info(f"Demand 缺失值比例:\n{(missing_demand / num_demand_rows * 100).round(2).astype(str) + '%'}") # 显示百分比
    else:
        logger.warning("Demand 行数为 0 或未知，无法计算缺失比例。")


    # Metadata 缺失值
    logger.info("Metadata 缺失值统计：")
    missing_metadata = ddf_metadata.isnull().sum().compute()
    logger.info(f"\n{missing_metadata.to_string()}")
    if num_metadata_rows > 0:
        logger.info(f"Metadata 缺失值比例:\n{(missing_metadata / num_metadata_rows * 100).round(2).astype(str) + '%'}") # 显示百分比
    else:
        logger.warning("Metadata 行数为 0 或未知，无法计算缺失比例。")


    # Weather 缺失值
    logger.info("Weather 缺失值统计：")
    missing_weather = ddf_weather.isnull().sum().compute()
    logger.info(f"\n{missing_weather.to_string()}")
    if num_weather_rows > 0:
        logger.info(f"Weather 缺失值比例:\n{(missing_weather / num_weather_rows * 100).round(2).astype(str) + '%'}") # 显示百分比
    else:
        logger.warning("Weather 行数为 0 或未知，无法计算缺失比例。")


def check_duplicates(ddf_demand, ddf_metadata, ddf_weather):
    """检查并记录数据帧中的重复值。"""
    logger.info("--- 检查重复值 ---")

    # Demand 重复值 (基于 unique_id 和 timestamp)
    logger.info("检查 Demand 基于 ['unique_id', 'timestamp'] 的重复值...")
    # 检查每个分区内的重复，然后聚合看是否有任何分区存在重复
    has_duplicates_demand_partition = ddf_demand.reduction(
        lambda chunk: chunk.duplicated(subset=['unique_id', 'timestamp'], keep=False).any(), # 没有这个函数
        aggregate=lambda chunks: chunks.any(),
        meta=bool
    ).compute()
    logger.info(f"Demand 数据中 {'存在' if has_duplicates_demand_partition else '不存在'} 基于 ['unique_id', 'timestamp'] 的重复值 (分区内检查)。")
    if has_duplicates_demand_partition:
         logger.warning("Demand 数据中检测到分区内重复，如果需要精确全局计数，可能需要更复杂的操作（如 set_index）")
    # 尝试计算全局重复数 (如果数据不大或资源允许)
    # try:
    #     num_duplicates_demand = ddf_demand.duplicated(subset=['unique_id', 'timestamp']).sum().compute()
    #     logger.info(f"Demand 中基于 ['unique_id', 'timestamp'] 的全局重复行数估算：{num_duplicates_demand}")
    # except Exception as e:
    #      logger.warning(f"计算 Demand 全局重复数时出错 (可能因数据量大): {e}")


    # Metadata 重复值 (基于 unique_id)
    logger.info("检查 Metadata 基于 ['unique_id'] 的重复值...")
    # # 计算所有被标记为重复的行数（包括第一次出现的）
    # # 暂时注释掉以下行以避免 AttributeError
    # num_duplicates_metadata = ddf_metadata.duplicated(subset=['unique_id'], keep=False).sum().compute()
    # logger.info(f"Metadata 中基于 ['unique_id'] 的重复行数 (标记所有重复项): {num_duplicates_metadata}")
    # if num_duplicates_metadata > 0:
    #     logger.warning(f"发现 {num_duplicates_metadata} 行 Metadata 的 unique_id 重复，需要检查具体哪些 unique_id 重复了。")
    #     # 查找重复的 unique_id 本身可能更有用
    #     # duplicated_ids = ddf_metadata[ddf_metadata.duplicated(subset=['unique_id'], keep=False)]['unique_id'].unique().compute()
    #     # logger.warning(f"重复的 unique_id 样本：{duplicated_ids[:5]}") # 只显示前几个
    logger.warning("暂时跳过 Metadata 全局重复值精确计数。")


    # Weather 重复值 (基于 location_id 和 timestamp)
    logger.info("检查 Weather 基于 ['location_id', 'timestamp'] 的重复值...")
    # # 暂时注释掉以下行以避免潜在的类似错误
    # num_duplicates_weather = ddf_weather.duplicated(subset=['location_id', 'timestamp'], keep=False).sum().compute()
    # logger.info(f"Weather 中基于 ['location_id', 'timestamp'] 的重复行数 (标记所有重复项): {num_duplicates_weather}")
    # if num_duplicates_weather > 0:
    #     logger.warning(f"发现 {num_duplicates_weather} 行 Weather 的 location_id 和 timestamp 组合重复。")
        # logger.info(ddf_weather[ddf_weather.duplicated(subset=['location_id', 'timestamp'], keep=False)].head().to_string()) # 显示一些重复行
    logger.warning("暂时跳过 Weather 全局重复值精确计数。根据之前的分析已知存在少量重复。")


def log_time_ranges(ddf_demand, ddf_weather):
    """计算并记录 Demand 和 Weather 数据集的时间戳范围。"""
    logger.info("--- 计算时间范围 ---")
    try:
        min_demand_ts = ddf_demand['timestamp'].min().compute()
        max_demand_ts = ddf_demand['timestamp'].max().compute()
        logger.info(f"Demand 时间范围：从 {min_demand_ts} 到 {max_demand_ts}")
    except Exception as e:
        logger.exception(f"计算 Demand 时间范围时出错：{e}")

    try:
        min_weather_ts = ddf_weather['timestamp'].min().compute()
        max_weather_ts = ddf_weather['timestamp'].max().compute()
        logger.info(f"Weather 时间范围：从 {min_weather_ts} 到 {max_weather_ts}")
    except Exception as e:
        logger.exception(f"计算 Weather 时间范围时出错：{e}")


def main():
    """主执行函数，编排数据加载和分析步骤。"""
    try:
        # 步骤 1: 加载数据集
        ddf_demand, ddf_metadata, ddf_weather = load_datasets()

        # 步骤 2: 记录基本信息
        num_demand_rows, num_metadata_rows, num_weather_rows = log_basic_info(ddf_demand, ddf_metadata, ddf_weather)

        # 步骤 3: 检查缺失值
        check_missing_values(ddf_demand, ddf_metadata, ddf_weather, num_demand_rows, num_metadata_rows, num_weather_rows)

        # 步骤 4: 检查重复值
        # FIXME: https://github.com/dask/dask/issues/1854 not working
        # check_duplicates(ddf_demand, ddf_metadata, ddf_weather)

        # 步骤 5: 记录时间范围
        log_time_ranges(ddf_demand, ddf_weather)

        logger.info("数据加载和初步检查脚本执行完毕。")
    except Exception as e:
        logger.exception(f"在主执行流程中发生严重错误：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
