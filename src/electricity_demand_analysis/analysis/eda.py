import pandas as pd
from loguru import logger
import sys
import os
import numpy as np # 导入 numpy 用于计算分位数等
from tqdm import tqdm # 导入 tqdm
from tqdm.contrib.concurrent import process_map # 导入 process_map
import multiprocessing
import matplotlib.pyplot as plt # 导入 matplotlib
import seaborn as sns          # 导入 seaborn

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

def check_duplicate_values(df: pd.DataFrame, df_name: str, sample_mode: bool = False):
    """Checks and logs duplicate value information for a DataFrame."""
    mode_info = "(Sampled)" if sample_mode else "(Full)"
    logger.info(f"--- Checking Duplicate Values for {df_name} {mode_info} ---")
    # 对非常大的 demand_df，即使在样本上检查重复也可能慢，可以考虑抽更小的样本或跳过
    # if df_name == "Demand" and sample_mode and df.shape[0] > 1_000_000: # Example threshold
    #     logger.warning("Duplicate check on large sampled Demand DF might still be slow.")

    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        logger.info(f"Found {num_duplicates} duplicate rows in {df_name} {mode_info}.")
    else:
        logger.info(f"No duplicate rows found in {df_name} {mode_info}.")
    return num_duplicates

def analyze_demand_distribution(demand_df: pd.DataFrame, sample_mode: bool = False):
    """Analyzes and logs the distribution of the 'y' column (electricity demand)."""
    mode_info = "(Sampled)" if sample_mode else "(Full)"
    logger.info(f"--- Analyzing Demand (y) Distribution {mode_info} ---")
    if 'y' not in demand_df.columns:
        logger.warning("Column 'y' not found in Demand DataFrame.")
        return

    # 计算描述性统计量 (忽略 NaN)
    logger.info(f"Calculating descriptive statistics for 'y' {mode_info}...")
    desc_stats = demand_df['y'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    logger.info(f"Descriptive Statistics for 'y' {mode_info}:\n{desc_stats}")

    # 检查是否存在非正值
    logger.info(f"Checking non-positive values for 'y' {mode_info}...")
    non_positive_count = (demand_df['y'] <= 0).sum()
    # 计算百分比时，如果使用样本，基于样本大小计算
    denominator = len(demand_df['y'].dropna())
    non_positive_perc = (non_positive_count / denominator) * 100 if denominator > 0 else 0
    logger.info(f"Count of non-positive (<= 0) demand values {mode_info}: {non_positive_count} ({non_positive_perc:.2f}% of non-missing values in the analyzed set)")

    # 检查是否存在极端值 (示例：基于 IQR)
    # Q1 = demand_df['y'].quantile(0.25)
    # Q3 = demand_df['y'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # outliers = demand_df[(demand_df['y'] < lower_bound) | (demand_df['y'] > upper_bound)]
    # logger.info(f"Potential outliers based on 1.5*IQR rule: {len(outliers)} rows")
    # 注意：对于大数据集，计算 IQR 和查找离群值可能非常耗时，这里暂时注释掉或考虑抽样计算

    # 提示：后续可以使用 matplotlib/seaborn 进行可视化，例如绘制直方图或箱线图
    logger.info("Suggestion: Visualize the distribution using histograms or box plots for better understanding.")


def analyze_timestamp_info(df: pd.DataFrame, df_name: str, time_col: str = 'timestamp'):
    """Analyzes and logs timestamp range and frequency."""
    logger.info(f"--- Analyzing Timestamp Info for {df_name} ---")
    if time_col not in df.columns:
        logger.warning(f"Timestamp column '{time_col}' not found in {df_name}.")
        return

    # 获取时间范围通常很快，可以在完整数据集上执行
    min_ts = df[time_col].min()
    max_ts = df[time_col].max()
    logger.info(f"Timestamp range: {min_ts} to {max_ts}")

    # 移除全局排序检查，因为它慢且意义不大
    # is_sorted = df[time_col].is_monotonic_increasing
    # logger.info(f"Is timestamp column globally sorted? {is_sorted}")


def analyze_metadata_categorical(metadata_df: pd.DataFrame):
    """Analyzes and logs distribution of key categorical features in metadata."""
    logger.info("--- Analyzing Metadata Categorical Features ---")
    categorical_cols = ['building_class', 'location', 'freq', 'timezone', 'dataset']
    for col in categorical_cols:
        if col in metadata_df.columns:
            logger.info(f"\nValue Counts for '{col}':")
            # 计算并打印值计数，显示 top N 和 other (如果类别太多)
            value_counts = metadata_df[col].value_counts(dropna=False) # 包括 NaN
            num_unique = value_counts.nunique()
            logger.info(f"Number of unique values (including NaN): {num_unique}")
            if num_unique > 20: # 如果唯一值过多，只显示最常见的
                 logger.info(f"Top 20 values:\n{value_counts.head(20)}")
                 other_count = value_counts.iloc[20:].sum()
                 logger.info(f"... and {other_count} in other categories.")
            else:
                logger.info(f"Counts:\n{value_counts}")
        else:
            logger.warning(f"Categorical column '{col}' not found in Metadata DataFrame.")


def analyze_weather_numeric(weather_df: pd.DataFrame):
    """Analyzes and logs distribution of key numerical weather features."""
    logger.info("--- Analyzing Weather Numeric Features ---")
    # 选择一些关键的天气指标进行分析
    numeric_cols = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'precipitation', 'rain', 'snowfall',
        'pressure_msl', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation'
    ]
    present_cols = [col for col in numeric_cols if col in weather_df.columns]

    if not present_cols:
        logger.warning("None of the selected key numeric weather columns found.")
        return

    # 计算描述性统计量
    desc_stats = weather_df[present_cols].describe(percentiles=[.01, .25, .5, .75, .99]).transpose()
    logger.info(f"Descriptive Statistics for Key Weather Variables:\n{desc_stats}")

    # 检查降水、雨、雪等是否有非负值
    for col in ['precipitation', 'rain', 'snowfall']:
        if col in weather_df.columns:
            negative_count = (weather_df[col] < 0).sum()
            if negative_count > 0:
                 logger.warning(f"Found {negative_count} negative values in '{col}', which might be unusual.")


# 为 process_map 创建一个简单的包装器，因为它需要一个接受单个参数的函数
def check_duplicates_wrapper(args):
    """Wrapper for check_duplicate_values to be used with process_map."""
    try:
        df, df_name, sample_mode = args
        # 注意：直接在子进程中调用 loguru 可能导致日志重复或行为异常
        # 一个更健壮的方法是返回结果，在主进程中记录日志
        # 但为了简单起见，我们暂时保留直接调用
        check_duplicate_values(df, df_name, sample_mode=sample_mode)
        return f"{df_name}: OK" # 返回一个简单的状态
    except Exception as e:
        # 在子进程中记录错误可能不可靠，最好返回错误信息
        return f"{df_name}: Error - {e}"

# --- Plotting Functions ---

def setup_plotting_style():
    """Sets a consistent style for plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6) # 设置默认图形大小

def plot_demand_distribution(demand_df_sampled: pd.DataFrame, output_dir: str):
    """Plots the distribution of electricity demand (y) using sampled data."""
    logger.info("Generating demand distribution plots (sampled)...")
    if 'y' not in demand_df_sampled.columns or demand_df_sampled['y'].isnull().all():
        logger.warning("Cannot plot demand distribution: 'y' column missing or all null in sample.")
        return

    y_data = demand_df_sampled['y'].dropna()

    # Plot 1: Histogram
    plt.figure()
    sns.histplot(y_data, kde=False, bins=100) # 使用更多 bins 可能需要调整
    plt.title('Distribution of Electricity Demand (y) - Sampled')
    plt.xlabel('Demand (y)')
    plt.ylabel('Frequency')
    # 由于数据高度右偏，可能需要对数刻度或限制范围才能看清主体
    plt.yscale('log') # 使用对数刻度 Y 轴
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'demand_y_histogram_sampled_log_scale.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved demand histogram (log y-scale) to {plot_path}")

    # Plot 2: Histogram (zoomed in, linear scale) - 排除极端值以看清主体
    q99 = y_data.quantile(0.99) # 只绘制 99% 分位数以下的数据
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

    # Plot 3: Box plot (可能因离群值过多而难以阅读，尝试对数刻度)
    plt.figure()
    sns.boxplot(x=y_data)
    plt.title('Box Plot of Electricity Demand (y) - Sampled')
    plt.xlabel('Demand (y)')
    plt.xscale('log') # 使用对数刻度 X 轴可能更好
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'demand_y_boxplot_sampled_log_scale.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved demand boxplot (log x-scale) to {plot_path}")


def plot_metadata_distribution(metadata_df: pd.DataFrame, output_dir: str):
    """Plots the distribution of key categorical features in metadata."""
    logger.info("Generating metadata distribution plots...")
    categorical_cols = ['building_class', 'freq', 'location'] # 选择要可视化的列

    for col in categorical_cols:
        if col in metadata_df.columns:
            plt.figure()
            # 如果类别过多，只绘制 top N
            top_n = 15
            value_counts = metadata_df[col].value_counts(dropna=False)
            if len(value_counts) > top_n:
                data_to_plot = value_counts.nlargest(top_n)
                plot_title = f'Distribution of Top {top_n} {col.replace("_", " ").title()}'
            else:
                data_to_plot = value_counts
                plot_title = f'Distribution of {col.replace("_", " ").title()}'

            sns.barplot(x=data_to_plot.index.astype(str), y=data_to_plot.values, palette="viridis") # 转为 str 避免类型问题
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


def plot_weather_distribution(weather_df: pd.DataFrame, output_dir: str):
    """Plots the distribution of key numerical weather features."""
    logger.info("Generating weather distribution plots...")
    numeric_cols = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m'
    ] # 选择几个关键变量

    for col in numeric_cols:
        if col in weather_df.columns and pd.api.types.is_numeric_dtype(weather_df[col]):
            plt.figure()
            data_to_plot = weather_df[col].dropna()
            sns.histplot(data_to_plot, kde=True, bins=50) # 可以加 kde 核密度估计
            plt.title(f'Distribution of {col.replace("_", " ").title()}')
            plt.xlabel(col.replace("_", " ").title())
            plt.ylabel('Frequency')

            # 特别处理降水，它可能有很多 0 值
            if col == 'precipitation' and (data_to_plot > 0).any():
                plt.yscale('log') # 尝试对数刻度
                plt.title(f'Distribution of {col.replace("_", " ").title()} (Log Scale, >0)')
                # 只绘制大于 0 的部分，因为 log(0) 无定义
                sns.histplot(data_to_plot[data_to_plot > 0], kde=True, bins=50)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'weather_{col}_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved {col} distribution plot to {plot_path}")
        elif col not in weather_df.columns:
             logger.warning(f"Cannot plot weather distribution: Column '{col}' not found.")
        else:
             logger.warning(f"Cannot plot weather distribution: Column '{col}' is not numeric.")


def run_eda(demand_sample_frac: float | None = 0.01):
    """Loads data and performs EDA checks, with optional sampling, parallelism, and plotting."""
    logger.info("Starting EDA process...")
    if demand_sample_frac is not None and 0 < demand_sample_frac < 1:
        logger.info(f"Demand data analysis will use a {demand_sample_frac*100:.2f}% sample for intensive operations.")
    elif demand_sample_frac is not None:
        logger.warning("Invalid sample fraction provided. Using full dataset analysis.")
        demand_sample_frac = None
    else:
        logger.info("No sampling requested for Demand DataFrame. Analyzing full dataset.")

    # --- Setup ---
    setup_plotting_style() # 设置绘图风格
    # 定义并创建输出目录
    plot_output_dir = os.path.join(project_root, "reports", "figures", "eda")
    os.makedirs(plot_output_dir, exist_ok=True)
    logger.info(f"Plots will be saved to: {plot_output_dir}")

    # --- Data Loading ---
    # 数据加载保持串行
    loaded_data = load_electricity_data()
    if not loaded_data:
        logger.error("Data loading failed. Exiting EDA.")
        return
    demand_df, metadata_df, weather_df = loaded_data

    # --- Sampling ---
    demand_df_sampled = demand_df # 默认为完整 DF
    is_sampled = False
    if demand_sample_frac is not None:
        logger.info(f"Creating {demand_sample_frac*100:.2f}% sample of Demand DataFrame...")
        try:
            # 使用 tqdm 包装 sample 操作
            with tqdm(total=1, desc="Sampling Demand DF", unit="op") as pbar:
                demand_df_sampled = demand_df.sample(frac=demand_sample_frac, random_state=42)
                pbar.update(1)
            logger.info(f"Sample created with {len(demand_df_sampled)} rows.")
            is_sampled = True
        except Exception as e:
            logger.error(f"Failed to create sample for demand_df: {e}. Proceeding with full dataset analysis.")
            demand_df_sampled = demand_df # Fallback to full df
            is_sampled = False

    # --- Initial Checks (Sequential Part) ---
    dataframes_info = {
        "Demand": (demand_df, demand_df_sampled, is_sampled), # Pass both original and sampled
        "Metadata": (metadata_df, metadata_df, False),
        "Weather": (weather_df, weather_df, False)
    }

    for name, (df_full, df_analysis, _) in dataframes_info.items():
        logger.info(f"\n--- Analyzing {name} DataFrame ---")
        if name == "Demand":
            df_full.info(verbose=True) # info on full df (no counts)
        else:
            df_full.info(verbose=True, show_counts=True) # info on smaller dfs
        check_missing_values(df_full, name) # Missing values on full df is usually ok
        logger.info(f"{name} DataFrame Head:\n{df_full.head()}") # Head is fast

    # --- Initial Checks (Parallel Part: Duplicates) ---
    logger.info("\n--- Checking Duplicates (Parallel) ---")
    duplicate_check_args = [
        (dataframes_info["Demand"][1], "Demand", dataframes_info["Demand"][2]), # Use sampled demand
        (dataframes_info["Metadata"][1], "Metadata", dataframes_info["Metadata"][2]),
        (dataframes_info["Weather"][1], "Weather", dataframes_info["Weather"][2]),
    ]

    # 确定工作进程数 (保守一点，避免过多内存占用)
    # max_workers = max(1, multiprocessing.cpu_count() // 2)
    max_workers = 3 # 或者直接指定一个固定的数量，比如 3，因为我们只有 3 个任务
    logger.info(f"Using up to {max_workers} workers for duplicate checks.")

    # 使用 process_map 并行执行包装函数
    results = process_map(check_duplicates_wrapper, duplicate_check_args,
                          max_workers=max_workers,
                          desc="Duplicate Checks",
                          chunksize=1) # chunksize=1 适用于任务差异较大的情况

    # 可以在这里检查 results 中的状态/错误
    logger.info(f"Duplicate check results: {results}")


    # --- Deeper Dive (Sequential Analysis) ---
    logger.info("\n--- Deeper Dive Analysis ---")
    analyze_demand_distribution(demand_df_sampled, sample_mode=is_sampled)
    analyze_timestamp_info(demand_df, "Demand")
    analyze_metadata_categorical(metadata_df)
    analyze_weather_numeric(weather_df)
    analyze_timestamp_info(weather_df, "Weather")

    # --- Generating Plots ---
    logger.info("\n--- Generating Plots ---")
    try:
        # 基于抽样数据绘制 Demand 分布
        if is_sampled:
            plot_demand_distribution(demand_df_sampled, plot_output_dir)
        else:
            logger.warning("Skipping demand distribution plots as sampling was not performed or failed.")

        # 绘制 Metadata 分布
        plot_metadata_distribution(metadata_df, plot_output_dir)

        # 绘制 Weather 分布
        plot_weather_distribution(weather_df, plot_output_dir)

        # TODO: 添加更多图表，例如时间序列图 (可能需要更多数据处理)
        # plot_demand_timeseries(demand_df_sampled, plot_output_dir) # 示例

    except Exception as e:
        logger.error(f"An error occurred during plot generation: {e}")


    logger.success("EDA checks and plot generation completed.")

if __name__ == "__main__":
    run_eda(demand_sample_frac=0.01)