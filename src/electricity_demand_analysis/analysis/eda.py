import pandas as pd
import dask.dataframe as dd # Import Dask
from dask.diagnostics import ProgressBar
from loguru import logger
import sys
import os
import numpy as np # 导入 numpy 用于计算分位数等
from tqdm import tqdm # 导入 tqdm
import multiprocessing
import matplotlib.pyplot as plt # 导入 matplotlib
import seaborn as sns          # 导入 seaborn
from dask.distributed import Client, LocalCluster # Import Client/LocalCluster if using persist

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

# 确保 matplotlib 不会尝试使用 GUI 后端 (适用于服务器环境或非交互式运行)
import matplotlib
matplotlib.use('Agg')

# 配置 loguru
log_file_path = os.path.join(project_root, "logs", "eda_analysis_dask.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # 确保日志目录存在
logger.add(log_file_path, rotation="10 MB", level="INFO") # 单独的 EDA 日志

def compute_with_progress(dask_obj, desc="Computing"):
    """Helper function to compute Dask object with progress bar."""
    logger.info(f"Starting computation: {desc}")
    with ProgressBar(dt=1.0): # Update progress every 1 second
        result = dask_obj.compute()
    logger.success(f"Computation finished: {desc}")
    return result

def check_missing_values_dask(ddf: dd.DataFrame, df_name: str):
    """Checks and logs missing value information for a Dask DataFrame."""
    logger.info(f"--- Checking Missing Values for {df_name} (Dask) ---")
    # Compute length lazily first if possible
    try:
        total_rows = compute_with_progress(ddf.index.size, desc=f"Total row count for {df_name}")
        logger.info(f"Total rows (computed): {total_rows}")
    except Exception as e:
        logger.warning(f"Could not compute exact row count efficiently, using len(): {e}")
        total_rows = len(ddf) # Fallback, might compute
        logger.info(f"Total rows (fallback using len): {total_rows}")

    missing_counts = compute_with_progress(ddf.isnull().sum(), desc=f"Missing counts for {df_name}")
    if missing_counts.sum() == 0:
        logger.info(f"No missing values found in {df_name}.")
        return pd.DataFrame({'Missing Count': [], 'Missing Percentage (%)': []})

    missing_percentages = (missing_counts / total_rows) * 100 if total_rows > 0 else 0
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage (%)': missing_percentages
    }).reset_index().rename(columns={'index': 'Column'})

    missing_info = missing_info[missing_info['Missing Count'] > 0]
    if not missing_info.empty:
        logger.info(f"Missing values found in {df_name}:\n{missing_info.to_string()}")
    else:
        logger.info(f"No missing values found in {df_name}.")
    return missing_info


def check_duplicate_values_dask(ddf: dd.DataFrame, df_name: str, subset=None):
    """Checks and logs duplicate value information using map_partitions."""
    logger.info(f"--- Checking Duplicate Values for {df_name} (Dask map_partitions) ---")

    if subset:
         logger.info(f"Checking duplicates based on subset: {subset}")
         # Define the function to apply to each partition
         def count_duplicates_subset(df, subset_cols):
             return df.duplicated(subset=subset_cols).sum()
         # Apply the function using map_partitions
         # Need to provide metadata about the output type (it's a single integer)
         partition_dup_counts = ddf.map_partitions(
             count_duplicates_subset, subset_cols=subset, meta=('duplicates', 'int64')
         )
    else:
         logger.info("Checking duplicates based on all columns...")
         def count_duplicates_all(df):
             return df.duplicated().sum()
         partition_dup_counts = ddf.map_partitions(
             count_duplicates_all, meta=('duplicates', 'int64')
         )

    # Sum the duplicate counts from all partitions
    total_duplicates = compute_with_progress(partition_dup_counts.sum(), desc=f"Duplicate check for {df_name}")

    if total_duplicates > 0:
        logger.info(f"Found {total_duplicates} duplicate rows in {df_name}.")
    else:
        logger.info(f"No duplicate rows found in {df_name}.")
    return total_duplicates


def analyze_demand_distribution_dask(demand_ddf: dd.DataFrame, sample_frac: float = 0.01):
    """Analyzes and logs the distribution of 'y' using a sample from the Dask DataFrame."""
    logger.info(f"--- Analyzing Demand (y) Distribution (Sampled {sample_frac*100:.2f}% from Dask) ---")
    if 'y' not in demand_ddf.columns:
        logger.warning("Column 'y' not found in Demand Dask DataFrame.")
        return None

    # Sample the Dask DataFrame first
    logger.info(f"Sampling {sample_frac*100:.2f}% of Demand Dask DataFrame...")
    demand_sample_ddf = demand_ddf.sample(frac=sample_frac, random_state=42)

    # Compute the sample into a Pandas DataFrame
    demand_sample_df = compute_with_progress(demand_sample_ddf, desc="Computing demand sample")
    logger.info(f"Computed sample size: {len(demand_sample_df)} rows")

    if demand_sample_df.empty or 'y' not in demand_sample_df.columns:
         logger.warning("Computed sample is empty or 'y' column missing.")
         return None

    # Now perform analysis on the Pandas sample
    logger.info("Calculating descriptive statistics for 'y' (sampled)...")
    desc_stats = demand_sample_df['y'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    logger.info(f"Descriptive Statistics for 'y' (sampled):\n{desc_stats}")

    logger.info("Checking non-positive values for 'y' (sampled)...")
    y_non_missing_sample = demand_sample_df['y'].dropna()
    non_positive_count = (y_non_missing_sample <= 0).sum()
    denominator = len(y_non_missing_sample)
    non_positive_perc = (non_positive_count / denominator) * 100 if denominator > 0 else 0
    logger.info(f"Count of non-positive (<= 0) demand values (sampled): {non_positive_count} ({non_positive_perc:.2f}% of non-missing values in sample)")

    logger.info("Suggestion: Visualize the distribution using histograms or box plots (using the computed sample).")
    return demand_sample_df # Return the computed sample for plotting


def analyze_timestamp_info_dask(ddf: dd.DataFrame, df_name: str, time_col: str = 'timestamp'):
    """Analyzes and logs timestamp range using Dask."""
    logger.info(f"--- Analyzing Timestamp Info for {df_name} (Dask) ---")
    if time_col not in ddf.columns:
        logger.warning(f"Timestamp column '{time_col}' not found in {df_name}.")
        return

    # Compute min and max (efficient in Dask)
    min_ts = compute_with_progress(ddf[time_col].min(), desc=f"Min timestamp for {df_name}")
    max_ts = compute_with_progress(ddf[time_col].max(), desc=f"Max timestamp for {df_name}")
    logger.info(f"Timestamp range: {min_ts} to {max_ts}")

    # is_monotonic_increasing is very expensive and often impractical in Dask, skip it.
    # logger.info("Skipping global sorted check (impractical for large Dask DataFrames).")


def analyze_metadata_categorical(metadata_df: pd.DataFrame):
    """Analyzes and logs distribution of key categorical features in metadata (Pandas)."""
    logger.info("--- Analyzing Metadata Categorical Features (Pandas) ---")
    categorical_cols = ['building_class', 'location', 'freq', 'timezone', 'dataset']
    for col in categorical_cols:
        if col in metadata_df.columns:
            logger.info(f"\nValue Counts for '{col}':")
            value_counts = metadata_df[col].value_counts(dropna=False)
            num_unique = len(value_counts) # Use len() for Series value_counts
            logger.info(f"Number of unique values (including NaN): {num_unique}")
            if num_unique > 20:
                 logger.info(f"Top 20 values:\n{value_counts.head(20)}")
                 other_count = value_counts.iloc[20:].sum()
                 logger.info(f"... and {other_count} in other categories.")
            else:
                logger.info(f"Counts:\n{value_counts}")
        else:
            logger.warning(f"Categorical column '{col}' not found in Metadata DataFrame.")


def analyze_weather_numeric_dask(weather_ddf: dd.DataFrame):
    """Analyzes and logs distribution of key numerical weather features using Dask."""
    logger.info("--- Analyzing Weather Numeric Features (Dask) ---")
    numeric_cols = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'precipitation', 'rain', 'snowfall',
        'pressure_msl', 'cloud_cover', 'wind_speed_10m' # Removed shortwave_radiation if not present
        # Check weather_ddf.columns if you added prefixes like 'weather_'
    ]
    # Adjust column names if they have prefixes from merging later
    prefixed_numeric_cols = [f"weather_{col}" if f"weather_{col}" in weather_ddf.columns else col for col in numeric_cols]
    present_cols = [col for col in prefixed_numeric_cols if col in weather_ddf.columns]

    if not present_cols:
        logger.warning("None of the selected key numeric weather columns found in Weather Dask DataFrame.")
        return

    # Compute descriptive statistics (can be memory intensive for many columns)
    logger.info("Calculating descriptive statistics for key weather variables (Dask)...")
    desc_stats_dask = weather_ddf[present_cols].describe(percentiles=[.01, .25, .5, .75, .99])
    desc_stats = compute_with_progress(desc_stats_dask, desc="Weather descriptive stats")
    logger.info(f"Descriptive Statistics for Key Weather Variables:\n{desc_stats.transpose()}") # Transpose for better readability

    # Check for negative values (compute required for each check)
    for col in ['weather_precipitation', 'weather_rain', 'weather_snowfall']: # Use prefixed names if applicable
        if col in weather_ddf.columns:
            logger.info(f"Checking for negative values in '{col}' (Dask)...")
            negative_count = compute_with_progress((weather_ddf[col] < 0).sum(), desc=f"Negative check for {col}")
            if negative_count > 0:
                 logger.warning(f"Found {negative_count} negative values in '{col}', which might be unusual.")


# --- Plotting Functions (Mostly Unchanged, expect Pandas input) ---
# These functions expect Pandas DataFrames (e.g., the computed sample)

def setup_plotting_style():
    """Sets a consistent style for plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

def plot_demand_distribution(demand_df_sampled: pd.DataFrame, output_dir: str):
    """Plots the distribution of electricity demand (y) using sampled Pandas data."""
    logger.info("Generating demand distribution plots (from computed sample)...")
    if demand_df_sampled is None or 'y' not in demand_df_sampled.columns or demand_df_sampled['y'].isnull().all():
        logger.warning("Cannot plot demand distribution: Sample data is missing, empty, or lacks 'y' column.")
        return

    y_data = demand_df_sampled['y'].dropna()
    if y_data.empty:
         logger.warning("No non-missing 'y' data in the sample to plot.")
         return

    # Plot 1: Histogram (Log Scale Y)
    plt.figure()
    sns.histplot(y_data, kde=False, bins=100)
    plt.title('Distribution of Electricity Demand (y) - Sampled')
    plt.xlabel('Demand (y)')
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'demand_y_histogram_sampled_log_scale.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved demand histogram (log y-scale) to {plot_path}")

    # Plot 2: Histogram (Zoomed Linear Scale)
    q99 = y_data.quantile(0.99) if not y_data.empty else 0
    plt.figure()
    sns.histplot(y_data[y_data <= q99], kde=False, bins=50)
    plt.title(f'Distribution of Electricity Demand (y <= {q99:.2f}) - Sampled')
    plt.xlabel('Demand (y)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'demand_y_histogram_sampled_zoomed.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved zoomed demand histogram to {plot_path}")

    # Plot 3: Box plot (Log Scale X)
    plt.figure()
    sns.boxplot(x=y_data)
    plt.title('Box Plot of Electricity Demand (y) - Sampled (Log Scale X)')
    plt.xlabel('Demand (y)')
    try:
        # Only set log scale if there are positive values
        if (y_data > 0).any():
             plt.xscale('log')
        else:
             logger.warning("Cannot use log scale for boxplot: no positive y values in sample.")
    except Exception as e:
        logger.warning(f"Could not set log scale for boxplot: {e}")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'demand_y_boxplot_sampled_log_scale.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved demand boxplot (log x-scale) to {plot_path}")

def plot_metadata_distribution(metadata_df: pd.DataFrame, output_dir: str):
    """Plots the distribution of key categorical features in metadata (Pandas)."""
    logger.info("Generating metadata distribution plots...")
    categorical_cols = ['building_class', 'freq', 'location']

    for col in categorical_cols:
        if col in metadata_df.columns:
            plt.figure()
            top_n = 15
            value_counts = metadata_df[col].value_counts(dropna=False)
            if len(value_counts) > top_n:
                data_to_plot = value_counts.nlargest(top_n)
                plot_title = f'Distribution of Top {top_n} {col.replace("_", " ").title()}'
            else:
                data_to_plot = value_counts
                plot_title = f'Distribution of {col.replace("_", " ").title()}'

            sns.barplot(x=data_to_plot.index.astype(str), y=data_to_plot.values, palette="viridis")
            plt.title(plot_title)
            plt.xlabel(col.replace("_", " ").title())
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'metadata_{col}_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved {col} distribution plot to {plot_path}")
        else:
             logger.warning(f"Cannot plot metadata distribution: Column '{col}' not found.")

def plot_weather_distribution(weather_ddf: dd.DataFrame, output_dir: str):
    """Plots the distribution of key numerical weather features using Dask samples."""
    logger.info("Generating weather distribution plots (sampled from Dask)...")
    numeric_cols = [ # Use prefixed names if applicable from merging later
        'temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m'
    ]
    prefixed_numeric_cols = [f"weather_{col}" if f"weather_{col}" in weather_ddf.columns else col for col in numeric_cols]
    present_cols = [col for col in prefixed_numeric_cols if col in weather_ddf.columns]

    # Weather data is much smaller than demand, but still potentially large.
    # Sample it for plotting to be safe, or compute directly if confident it fits memory.
    sample_frac_weather = 0.1 # Use a larger sample for weather if needed
    logger.info(f"Sampling {sample_frac_weather*100:.1f}% of Weather Dask DataFrame for plotting...")
    weather_sample_ddf = weather_ddf[present_cols].sample(frac=sample_frac_weather, random_state=42)
    weather_sample_df = compute_with_progress(weather_sample_ddf, desc="Computing weather sample for plotting")

    for col in present_cols:
        if col in weather_sample_df.columns and pd.api.types.is_numeric_dtype(weather_sample_df[col]):
            plt.figure()
            data_to_plot = weather_sample_df[col].dropna()
            if data_to_plot.empty:
                logger.warning(f"No non-missing data for '{col}' in weather sample.")
                plt.close()
                continue

            sns.histplot(data_to_plot, kde=True, bins=50)
            plot_title = f'Distribution of {col.replace("_", " ").title()} (Sampled)'
            plt.title(plot_title)
            plt.xlabel(col.replace("_", " ").title())
            plt.ylabel('Frequency')

            # Special handling for precipitation (log scale)
            if 'precipitation' in col and (data_to_plot > 0).any():
                plt.figure() # Create a new figure for the log scale plot
                sns.histplot(data_to_plot[data_to_plot > 0], kde=True, bins=50)
                plt.yscale('log')
                plt.title(f'{plot_title} (Log Scale, >0)')
                plt.xlabel(col.replace("_", " ").title())
                plt.ylabel('Frequency (Log Scale)')
                plt.tight_layout()
                plot_path_log = os.path.join(output_dir, f'{col}_distribution_sampled_log.png')
                plt.savefig(plot_path_log)
                plt.close() # Close the log scale plot
                logger.info(f"Saved {col} log distribution plot to {plot_path_log}")
                # Continue to save the linear scale plot as well
                plt.figure() # Re-create the linear plot figure
                sns.histplot(data_to_plot, kde=True, bins=50)
                plt.title(plot_title)
                plt.xlabel(col.replace("_", " ").title())
                plt.ylabel('Frequency')


            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{col}_distribution_sampled.png')
            plt.savefig(plot_path)
            plt.close() # Close the linear scale plot figure
            logger.info(f"Saved {col} distribution plot to {plot_path}")
        elif col not in weather_sample_df.columns:
             logger.warning(f"Cannot plot weather distribution: Column '{col}' not found in sample.")
        else:
             logger.warning(f"Cannot plot weather distribution: Column '{col}' in sample is not numeric.")


# --- Main EDA Execution (Using Dask) ---

def run_eda_dask(demand_sample_frac: float = 0.01):
    """Loads data using Dask and performs EDA checks and plotting."""
    logger.info("--- Starting EDA process (Dask) ---")
    logger.info(f"Demand data analysis/plotting will use a {demand_sample_frac*100:.2f}% sample.")

    # --- Setup ---
    setup_plotting_style()
    plot_output_dir = os.path.join(project_root, "reports", "figures", "eda_dask") # New output dir
    os.makedirs(plot_output_dir, exist_ok=True)
    logger.info(f"Plots will be saved to: {plot_output_dir}")

    # --- Data Loading (Request Dask format) ---
    logger.info("Loading data as Dask DataFrames...")
    npartitions = os.cpu_count() * 2 if os.cpu_count() else 4
    loaded_data = load_electricity_data(return_format="dask", npartitions=npartitions)
    if not loaded_data:
        logger.error("Data loading failed. Exiting EDA.")
        return
    demand_ddf, metadata_df, weather_ddf = loaded_data
    logger.info("Data loaded successfully (Demand/Weather as Dask, Metadata as Pandas).")
    logger.info(f"Demand partitions: {demand_ddf.npartitions}, Weather partitions: {weather_ddf.npartitions}")

    # --- Initial Checks (using Dask compute) ---
    dataframes_info = {
        "Demand (Dask)": demand_ddf,
        "Metadata (Pandas)": metadata_df,
        "Weather (Dask)": weather_ddf # Use original loaded ddf
    }

    computed_sample = None # To store the computed demand sample

    for name, df_obj in dataframes_info.items():
        logger.info(f"\n--- Analyzing {name} ---")
        is_dask = isinstance(df_obj, dd.DataFrame)

        if is_dask:
            logger.info(f"Schema (dtypes):\n{df_obj.dtypes}")
            check_missing_values_dask(df_obj, name)
            if name == "Weather (Dask)":
                 # Use map_partitions based check
                 check_duplicate_values_dask(df_obj, name, subset=['location_id', 'timestamp'])
            else:
                 logger.info(f"Skipping full duplicate check for {name} due to potential cost.")

            if name == "Demand (Dask)":
                computed_sample = analyze_demand_distribution_dask(df_obj, sample_frac=demand_sample_frac)
                analyze_timestamp_info_dask(df_obj, name)
            elif name == "Weather (Dask)":
                analyze_weather_numeric_dask(df_obj)
                analyze_timestamp_info_dask(df_obj, name)

        else: # Pandas DataFrame (Metadata)
            df_obj.info(verbose=True, show_counts=True)
            check_missing_values(df_obj, name)
            check_duplicate_values(df_obj, name) # Use Pandas version here
            logger.info(f"{name} Head:\n{df_obj.head()}")
            analyze_metadata_categorical(df_obj)

    # --- Generating Plots ---
    logger.info("\n--- Generating Plots (using computed samples where necessary) ---")
    try:
        if computed_sample is not None:
            plot_demand_distribution(computed_sample, plot_output_dir)
        else:
            logger.warning("Skipping demand distribution plots as sample computation failed or was skipped.")

        plot_metadata_distribution(metadata_df, plot_output_dir)
        plot_weather_distribution(weather_ddf, plot_output_dir)

    except Exception as e:
        logger.error(f"An error occurred during plot generation: {e}", exc_info=True)

    logger.success("EDA checks and plot generation completed (Dask).")


# --- Helper functions for Pandas checks (if needed, copy from original eda.py) ---
# Add check_missing_values and check_duplicate_values for Pandas if they are not globally available
def check_missing_values(df: pd.DataFrame, df_name: str):
    """Checks and logs missing value information for a Pandas DataFrame."""
    logger.info(f"--- Checking Missing Values for {df_name} (Pandas) ---")
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage (%)': missing_percentages
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0]
    if not missing_info.empty:
        logger.info(f"Missing values found in {df_name}:\n{missing_info.to_string()}")
    else:
        logger.info(f"No missing values found in {df_name}.")
    return missing_info

def check_duplicate_values(df: pd.DataFrame, df_name: str, subset=None):
    """Checks and logs duplicate value information for a Pandas DataFrame."""
    logger.info(f"--- Checking Duplicate Values for {df_name} (Pandas) ---")
    num_duplicates = df.duplicated(subset=subset).sum()
    if num_duplicates > 0:
        logger.info(f"Found {num_duplicates} duplicate rows in {df_name}.")
    else:
        logger.info(f"No duplicate rows found in {df_name}.")
    return num_duplicates


if __name__ == "__main__":
    run_eda_dask(demand_sample_frac=0.01)