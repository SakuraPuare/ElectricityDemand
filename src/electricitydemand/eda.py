import sys
import os
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
plots_dir = os.path.join(project_root, 'plots') # 图表保存目录
os.makedirs(plots_dir, exist_ok=True) # 确保目录存在

# 设置日志级别为 INFO
setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
logger.info(f"项目根目录: {project_root}")
logger.info(f"日志目录: {logs_dir}")
logger.info(f"图表目录: {plots_dir}")

# --- 数据文件路径 ---
data_dir = os.path.join(project_root, "data")
demand_path = os.path.join(data_dir, "demand.parquet")
# metadata_path = os.path.join(data_dir, "metadata.parquet") # 暂时不需要
# weather_path = os.path.join(data_dir, "weather.parquet") # 暂时不需要
logger.info(f"数据目录: {data_dir}")

def load_demand_data():
    """仅加载 Demand 数据集."""
    logger.info("开始加载 Demand 数据集...")
    try:
        ddf_demand = dd.read_parquet(demand_path)
        logger.info(f"成功加载 Demand 数据: {demand_path}")
        logger.info(f"Demand Dask DataFrame npartitions: {ddf_demand.npartitions}, columns: {ddf_demand.columns}")
        # 立即计算行数（或者在需要时计算）
        # num_rows = len(ddf_demand)
        # logger.info(f"Demand 数据行数 (估算): {num_rows:,}")
        return ddf_demand
    except FileNotFoundError as e:
        logger.error(f"Demand 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Demand 数据集时发生错误: {e}")
        sys.exit(1)

def analyze_demand_y_distribution(ddf_demand, sample_frac=0.005, random_state=42):
    """分析 Demand 数据 'y' 列的分布 (基于抽样)。"""
    logger.info(f"--- 开始分析 Demand 'y' 列分布 (抽样比例: {sample_frac:.1%}) ---")

    try:
        # 1. 对 'y' 列进行抽样
        logger.info("对 'y' 列进行抽样...")
        # 使用 persist() 将抽样结果保留在内存中，加速后续计算
        y_sample = ddf_demand['y'].dropna().sample(frac=sample_frac, random_state=random_state).persist()
        num_samples = len(y_sample)
        logger.info(f"抽样完成，得到 {num_samples:,} 个非空样本。")

        if num_samples == 0:
            logger.warning("抽样结果为空，无法进行分布分析。")
            return None # 或者返回一个空的 Pandas Series

        # 2. 计算描述性统计信息
        logger.info("计算抽样数据的描述性统计信息...")
        desc_stats = y_sample.describe().compute()
        logger.info(f"'y' 列 (抽样) 描述性统计:\n{desc_stats.to_string()}")

        # 3. 检查非正值比例
        logger.info("检查抽样数据中的非正值 (<= 0)...")
        non_positive_count = (y_sample <= 0).sum().compute()
        non_positive_perc = (non_positive_count / num_samples) * 100 if num_samples > 0 else 0
        logger.info(f"抽样数据中 'y' <= 0 的数量: {non_positive_count:,} ({non_positive_perc:.2f}%)")

        # 将 Pandas Series 返回，以便后续绘图
        # 注意：compute() 会将 Dask Series 转换为 Pandas Series
        y_sample_pd = y_sample.compute()
        logger.info("已将抽样结果转换为 Pandas Series 用于后续绘图。")
        return y_sample_pd

    except Exception as e:
        logger.exception(f"分析 Demand 'y' 列分布时发生错误: {e}")
        return None
    finally:
        # 如果使用了 persist()，计算完成后可以考虑释放内存
        # Dask 的垃圾回收通常会自动处理，但显式调用也可以
        # from dask.distributed import Client
        # client = Client.current() # 获取当前的 Dask 客户端
        # client.cancel(y_sample) # 尝试取消计算并释放内存
        pass


def main():
    """主执行函数，编排 EDA 步骤。"""
    # 步骤 1: 加载数据
    ddf_demand = load_demand_data()

    # 步骤 2: 分析 Demand 'y' 列的分布
    y_sample_pd = analyze_demand_y_distribution(ddf_demand, sample_frac=0.005) # 使用 0.5% 抽样

    # 后续可以添加绘图等步骤
    if y_sample_pd is not None and not y_sample_pd.empty:
        logger.info("后续可以基于 y_sample_pd 进行绘图分析。")
        # plot_demand_y_distribution(y_sample_pd) # 例如调用绘图函数
    else:
        logger.warning("由于抽样数据为空或分析出错，跳过后续基于 'y' 的分析步骤。")

    logger.info("EDA 脚本初步执行完毕。")


if __name__ == "__main__":
    main() 