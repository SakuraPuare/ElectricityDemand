from datasets import load_dataset
import pandas as pd
from loguru import logger  # 使用 loguru 替代 print

# 配置 loguru
logger.add("data_loading.log", rotation="500 MB") # 将日志记录到文件

def load_electricity_data(dataset_name: str = "EDS-lab/electricity-demand") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Loads the demand, metadata, and weather data from the Hugging Face Hub.

    Args:
        dataset_name: The name of the dataset on Hugging Face Hub.

    Returns:
        A tuple containing three pandas DataFrames: (demand_df, metadata_df, weather_df),
        or None if loading fails.
    """
    try:
        # 加载特定子集和分割
        logger.info(f"Attempting to load data from '{dataset_name}'...")

        logger.info("Loading demand data (subset: demand, split: train)...")
        demand_ds = load_dataset(dataset_name, name="demand", split="train", trust_remote_code=True) # 添加 trust_remote_code=True
        logger.success("Demand data loaded successfully.")

        logger.info("Loading metadata (subset: metadata, split: train)...")
        metadata_ds = load_dataset(dataset_name, name="metadata", split="train", trust_remote_code=True) # 添加 trust_remote_code=True
        logger.success("Metadata loaded successfully.")

        logger.info("Loading weather data (subset: weather, split: train)...")
        weather_ds = load_dataset(dataset_name, name="weather", split="train", trust_remote_code=True) # 添加 trust_remote_code=True
        logger.success("Weather data loaded successfully.")

        # 转换为 Pandas DataFrame
        logger.info("Converting datasets to Pandas DataFrames...")
        demand_df = demand_ds.to_pandas()
        metadata_df = metadata_ds.to_pandas()
        weather_df = weather_ds.to_pandas()
        logger.success("Conversion to DataFrames complete.")

        return demand_df, metadata_df, weather_df

    except ImportError as ie:
         logger.error(f"Import error: {ie}. Please ensure 'datasets', 'pandas', and 'pyarrow' or 'fastparquet' are installed.")
         logger.error("Install using: pip install datasets pandas pyarrow fastparquet loguru")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        logger.error("Check your internet connection, dataset name, and necessary permissions.")
        return None

# --- 使用示例 ---
if __name__ == "__main__":
    logger.info("Starting data loading process...")
    loaded_data = load_electricity_data()

    if loaded_data:
        demand_df, metadata_df, weather_df = loaded_data

        # 显示每个 DataFrame 的基本信息和前几行，以验证加载
        logger.info("\n--- Demand DataFrame Info ---")
        demand_df.info(verbose=True, show_counts=True) # 更详细的信息
        logger.info("Demand DataFrame Head:\n{}", demand_df.head())

        logger.info("\n--- Metadata DataFrame Info ---")
        metadata_df.info(verbose=True, show_counts=True)
        logger.info("Metadata DataFrame Head:\n{}", metadata_df.head())

        logger.info("\n--- Weather DataFrame Info ---")
        weather_df.info(verbose=True, show_counts=True)
        logger.info("Weather DataFrame Head:\n{}", weather_df.head())

        logger.success("Dataset downloaded and loaded successfully into Pandas DataFrames!")
    else:
        logger.error("Failed to load dataset.")
