import pandas as pd
from loguru import logger
import sys
import os

# 确保可以导入 src 目录下的模块
# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设 eda.py 在 src/electricity_demand_analysis/analysis/ 下）
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)

# 从 data.loader 导入数据加载函数
try:
    from src.electricity_demand_analysis.data.loader import load_electricity_data
except ImportError as e:
    logger.error(f"Failed to import load_electricity_data: {e}")
    logger.error("Ensure the script is run from the project root or the path is correctly configured.")
    sys.exit(1) # 如果无法导入，则退出

# 配置 loguru
log_file_path = os.path.join(project_root, "logs", "eda_analysis.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # 确保日志目录存在
logger.add(log_file_path, rotation="10 MB", level="INFO") # 单独的 EDA 日志

def check_missing_values(df: pd.DataFrame, df_name: str):
    """Checks and logs missing value information for a DataFrame."""
    logger.info(f"--- Checking Missing Values for {df_name} ---")
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage (%)': missing_percentages
    })
    # 只显示有缺失值的列
    missing_info = missing_info[missing_info['Missing Count'] > 0]
    if not missing_info.empty:
        logger.info(f"Missing values found in {df_name}:\n{missing_info}")
    else:
        logger.info(f"No missing values found in {df_name}.")
    return missing_info

def check_duplicate_values(df: pd.DataFrame, df_name: str):
    """Checks and logs duplicate value information for a DataFrame."""
    logger.info(f"--- Checking Duplicate Values for {df_name} ---")
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        logger.info(f"Found {num_duplicates} duplicate rows in {df_name}.")
    else:
        logger.info(f"No duplicate rows found in {df_name}.")
    return num_duplicates

def run_initial_eda():
    """Loads data and performs initial EDA checks."""
    logger.info("Starting EDA process...")
    loaded_data = load_electricity_data()

    if not loaded_data:
        logger.error("Data loading failed. Exiting EDA.")
        return

    demand_df, metadata_df, weather_df = loaded_data

    dataframes = {
        "Demand": demand_df,
        "Metadata": metadata_df,
        "Weather": weather_df
    }

    # 检查每个 DataFrame 的缺失值和重复值
    for name, df in dataframes.items():
        logger.info(f"\n--- Analyzing {name} DataFrame ---")
        df.info(verbose=True, show_counts=True) # 显示基本信息
        check_missing_values(df, name)
        check_duplicate_values(df, name)
        logger.info(f"{name} DataFrame Head:\n{df.head()}")

    logger.success("Initial EDA checks (Missing Values, Duplicates) completed.")

if __name__ == "__main__":
    run_initial_eda() 