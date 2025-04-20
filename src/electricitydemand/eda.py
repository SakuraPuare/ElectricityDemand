import sys
import os
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger # 提前导入 logger
import numpy as np # 需要 numpy 来处理对数变换中的 0 或负值
from tqdm import tqdm # 用于显示进度条
import contextlib # 用于更优雅地处理 Dask 内存清理
from dask import persist # <<<--- 添加此导入

# --- 项目设置 (路径和日志) ---
project_root = None
try:
    # 尝试标准的相对导入 (当作为包运行时)
    if __package__ and __package__.startswith('src.'):
        _script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_path)))
        from .utils.log_utils import setup_logger # 相对导入
    else:
        raise ImportError("未作为包运行或包结构不匹配。")

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
metadata_path = os.path.join(data_dir, "metadata.parquet")
weather_path = os.path.join(data_dir, "weather.parquet")
logger.info(f"数据目录: {data_dir}")

# --- 辅助函数 ---

def _save_plot(fig, filename, plots_dir):
    """保存 matplotlib 图表到指定目录并记录日志。"""
    if not plots_dir:
        logger.error("未提供图表保存目录 (plots_dir)，无法保存图表。")
        plt.close(fig) # 关闭图形以释放内存
        return
    filepath = os.path.join(plots_dir, filename)
    try:
        fig.savefig(filepath)
        logger.info(f"图表已保存到: {filepath}")
        plt.close(fig) # 保存后关闭图形
    except Exception as e:
        logger.exception(f"保存图表 '{filename}' 时出错: {e}")
        plt.close(fig) # 出错也要关闭

def _plot_numerical_distribution(data_pd_series, col_name, filename_base, plots_dir, title_prefix="", kde=True, showfliers=True):
    """绘制数值型 Pandas Series 的分布图 (直方图和箱线图)。"""
    if data_pd_series is None or data_pd_series.empty:
        logger.warning(f"Column '{col_name}' data is empty, skipping plot.")
        return

    logger.info(f"Plotting distribution for column '{col_name}' (Sample size: {len(data_pd_series):,})...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    try:
        # Histogram
        sns.histplot(data_pd_series, kde=kde, ax=axes[0])
        axes[0].set_title(f'{title_prefix}{col_name} Distribution')
        axes[0].set_xlabel(col_name)
        axes[0].set_ylabel('Frequency')

        # Box Plot
        sns.boxplot(x=data_pd_series, ax=axes[1], showfliers=showfliers)
        axes[1].set_title(f'{title_prefix}{col_name} Box Plot')
        axes[1].set_xlabel(col_name)

        plt.tight_layout()
        _save_plot(fig, f"{filename_base}.png", plots_dir)
    except Exception as e:
        logger.exception(f"Error plotting distribution for column '{col_name}': {e}")
        plt.close(fig) # Ensure figure is closed

def _plot_categorical_distribution(data_pd_series, col_name, filename_base, plots_dir, top_n=10, title_prefix=""):
    """绘制分类 Pandas Series 的分布图 (条形图)。"""
    if data_pd_series is None or data_pd_series.empty:
        logger.warning(f"Column '{col_name}' data is empty, skipping plot.")
        return

    logger.info(f"Plotting categorical distribution for column '{col_name}' (Top {top_n})...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        value_counts = data_pd_series.value_counts(dropna=False) # Include NaN
        num_unique = len(value_counts)

        if num_unique > top_n:
            logger.info(f"Column '{col_name}' has {num_unique} unique values, showing Top {top_n} and 'Others'.")
            top_values = value_counts.head(top_n)
            other_count = value_counts.iloc[top_n:].sum()
            if other_count > 0:
                others_label = 'Others'
                others_series = pd.Series([other_count], index=[others_label])
                if others_label in top_values.index: # Defensive check
                    top_values[others_label] += other_count
                else:
                    top_values = pd.concat([top_values, others_series])
            data_to_plot = top_values
            title = f'{title_prefix}Top {top_n} {col_name} Distribution'
        else:
            data_to_plot = value_counts
            title = f'{title_prefix}{col_name} Distribution'

        # Handle NaN labels for plotting and sort index for consistent plots if possible
        plot_index = data_to_plot.index.astype(str)
        # Try sorting by original value (numeric or string), fallback to count order
        try:
            sort_order = data_to_plot.sort_index().index.astype(str)
            plot_index_ordered = [idx for idx in sort_order if idx in plot_index] # Keep filtered indices
        except TypeError: # Cannot sort mixed types
            sort_order = data_to_plot.index.astype(str) # Order by appearance (count order)
            plot_index_ordered = plot_index # Use original order

        sns.barplot(x=plot_index, y=data_to_plot.values, ax=ax, palette="viridis", order=plot_index_ordered)
        ax.set_title(title)
        ax.set_xlabel(col_name)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        _save_plot(fig, f"{filename_base}.png", plots_dir)

    except Exception as e:
        logger.exception(f"Error plotting distribution for column '{col_name}': {e}")
        plt.close(fig)

def _log_value_counts(data_pd_series, col_name, top_n=10):
    """记录 Pandas Series 的值计数和百分比。"""
    if data_pd_series is None or data_pd_series.empty:
        logger.warning(f"列 '{col_name}' 的数据为空，跳过值计数记录。")
        return

    logger.info(f"--- 分析列: {col_name} 值分布 ---")
    total_rows = len(data_pd_series)
    counts = data_pd_series.value_counts(dropna=False) # 包含 NaN
    percentage = (counts / total_rows * 100).round(2) if total_rows > 0 else 0
    dist_df = pd.DataFrame({'计数': counts, '百分比 (%)': percentage})

    log_str = f"值分布 (含 NaN):\n{dist_df.head(top_n).to_string()}"
    if len(dist_df) > top_n:
        log_str += "\n..."
    logger.info(log_str)

    num_unique = data_pd_series.nunique(dropna=False)
    logger.info(f"列 '{col_name}' 唯一值数量 (含 NaN): {num_unique}")
    if data_pd_series.isnull().any():
        logger.warning(f"列 '{col_name}' 存在缺失值 (NaN)。")


@contextlib.contextmanager
def _dask_compute_context(*dask_objects):
    """上下文管理器，用于触发 Dask persist/compute 并尝试清理。"""
    persisted_objects = []
    try:
        if dask_objects:
            logger.debug(f"尝试持久化 {len(dask_objects)} 个 Dask 对象...")
            # 注意: persist 返回新的对象列表
            # 使用导入的 dask.persist 函数
            persisted_objects = persist(*dask_objects)
            logger.debug("Dask 对象持久化完成。")
        yield persisted_objects # 返回持久化的对象供使用
    finally:
        if persisted_objects:
            logger.debug("尝试释放 Dask 持久化对象...")
            try:
                from dask.distributed import Client, futures_of
                from distributed.client import Future # 检查 future 类型
                # 检查是否有活动的客户端
                with contextlib.suppress(RuntimeError): # 忽略 'no client found' 错误
                     client = Client.current()
                     # 收集所有的 Future 对象
                     futures = []
                     for obj in persisted_objects:
                         # 对于 DataFrame 或 Series
                         if hasattr(obj, 'dask'):
                            futures.extend(futures_of(obj))
                         # 如果直接传入 Future
                         elif isinstance(obj, Future):
                            futures.append(obj)

                     if futures:
                         logger.debug(f"找到 {len(futures)} 个 futures，尝试取消...")
                         client.cancel(futures, force=True) # 强制取消
                         # 可选：等待 futures 结束或被取消
                         # client.close() 或 await client.close() 如果在 async 上下文
                         # 注意：直接 close client 可能影响其他并发任务
                         # 更好的做法是依赖 Dask 的垃圾回收，但 cancel 可以帮助更快释放
                         logger.debug("取消 Futures 尝试完成。")
                     else:
                         logger.debug("未找到 Dask Futures 进行取消。")

            except ImportError:
                logger.debug("未安装 dask.distributed，跳过显式内存清理。")
            except Exception as e:
                logger.warning(f"清理 Dask 内存时出现异常: {e}", exc_info=False)
            logger.debug("Dask 上下文退出。")

# --- 数据加载函数 (保持不变) ---
def load_demand_data():
    """仅加载 Demand 数据集."""
    logger.info("开始加载 Demand 数据集...")
    try:
        ddf_demand = dd.read_parquet(demand_path)
        logger.info(f"成功加载 Demand 数据: {demand_path}")
        logger.info(f"Demand Dask DataFrame 分区数: {ddf_demand.npartitions}, 列: {ddf_demand.columns}")
        return ddf_demand
    except FileNotFoundError as e:
        logger.error(f"Demand 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Demand 数据集时发生错误: {e}")
        sys.exit(1)

def load_metadata():
    """加载 Metadata 数据集 (Pandas)."""
    logger.info("开始加载 Metadata 数据集...")
    try:
        pdf_metadata = pd.read_parquet(metadata_path)
        logger.info(f"成功加载 Metadata 数据: {metadata_path}")
        logger.info(f"Metadata Pandas DataFrame 形状: {pdf_metadata.shape}, 列: {pdf_metadata.columns.tolist()}")
        logger.info(f"Metadata 头部信息:\n{pdf_metadata.head().to_string()}")
        logger.info(f"Metadata 数据类型:\n{pdf_metadata.dtypes.to_string()}")
        return pdf_metadata
    except FileNotFoundError as e:
        logger.error(f"Metadata 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Metadata 数据集时发生错误: {e}")
        sys.exit(1)

def load_weather_data():
    """加载 Weather 数据集 (Dask)."""
    logger.info("开始加载 Weather 数据集...")
    try:
        ddf_weather = dd.read_parquet(weather_path)
        logger.info(f"成功加载 Weather 数据: {weather_path}")
        logger.info(f"Weather Dask DataFrame 分区数: {ddf_weather.npartitions}, 列: {ddf_weather.columns}")
        logger.info(f"Weather 数据类型:\n{ddf_weather.dtypes.to_string()}")
        return ddf_weather
    except FileNotFoundError as e:
        logger.error(f"Weather 数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Weather 数据集时发生错误: {e}")
        sys.exit(1)


# --- 分析函数 (重构后) ---

def analyze_demand_y_distribution(ddf_demand, sample_frac=0.005, random_state=42):
    """分析 Demand 数据 'y' 列的分布 (基于抽样)。"""
    logger.info(f"--- 开始分析 Demand 'y' 列分布 (抽样比例: {sample_frac:.1%}) ---")
    y_sample_pd = None
    try:
        with _dask_compute_context(ddf_demand['y']) as persisted_data:
            y_col = persisted_data[0] # 获取持久化的 'y' 列
            logger.info("对 'y' 列进行抽样...")
            y_sample = y_col.dropna().sample(frac=sample_frac, random_state=random_state)

            with _dask_compute_context(y_sample) as persisted_sample:
                y_sample_persisted = persisted_sample[0]
                num_samples = len(y_sample_persisted) # len 在持久化后更快
                logger.info(f"抽样完成，得到 {num_samples:,} 个非空样本。")

                if num_samples == 0:
                    logger.warning("抽样结果为空，无法进行分布分析。")
                    return None

                logger.info("计算抽样数据的描述性统计信息...")
                desc_stats = y_sample_persisted.describe().compute()
                logger.info(f"'y' 列 (抽样) 描述性统计:\n{desc_stats.to_string()}")

                logger.info("检查抽样数据中的非正值 (<= 0)...")
                non_positive_count = (y_sample_persisted <= 0).sum().compute()
                non_positive_perc = (non_positive_count / num_samples) * 100 if num_samples > 0 else 0
                logger.info(f"抽样数据中 'y' <= 0 的数量: {non_positive_count:,} ({non_positive_perc:.2f}%)")

                logger.info("将抽样结果转换为 Pandas Series...")
                y_sample_pd = y_sample_persisted.compute()
                logger.info("转换为 Pandas Series 完成。")
                return y_sample_pd

    except Exception as e:
        logger.exception(f"分析 Demand 'y' 列分布时发生错误: {e}")
        return None

def plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000, random_state=42):
    """绘制 Demand 'y' 列 (抽样) 的分布图并保存。"""
    if y_sample_pd is None or y_sample_pd.empty:
        logger.warning("Input 'y' sample is empty, skipping plotting.")
        return

    # --- Further sampling for plotting ---
    if len(y_sample_pd) > plot_sample_size:
        logger.info(f"Original sample size {len(y_sample_pd):,} is large, further sampling {plot_sample_size:,} points for plotting.")
        y_plot_sample = y_sample_pd.sample(n=plot_sample_size, random_state=random_state)
    else:
        y_plot_sample = y_sample_pd
    logger.info(f"--- Starting plotting for Demand 'y' distribution (Plot sample size: {len(y_plot_sample):,}) ---")

    # --- Plot original scale ---
    _plot_numerical_distribution(y_plot_sample, 'y',
                                 'demand_y_distribution_original_scale', plots_dir,
                                 title_prefix="Demand ", kde=False, showfliers=False)

    # --- Plot log scale ---
    epsilon = 1e-6
    y_plot_sample_non_neg = y_plot_sample[y_plot_sample >= 0]
    if y_plot_sample_non_neg.empty:
        logger.warning("No non-negative values in plot sample, skipping log scale plot.")
        return

    y_log_transformed = np.log1p(y_plot_sample_non_neg + epsilon)
    _plot_numerical_distribution(y_log_transformed, 'log1p(y + epsilon)',
                                 'demand_y_distribution_log1p_scale', plots_dir,
                                 title_prefix="Demand ", kde=True, showfliers=True)

    logger.info("Demand 'y' distribution plotting complete.")


def analyze_demand_timeseries_sample(ddf_demand, n_samples=5, plots_dir=None, random_state=42):
    """抽取 n_samples 个 unique_id 的完整时间序列数据，绘制图形并分析频率。"""
    logger.info(f"--- Starting analysis of Demand time series characteristics (sampling {n_samples} unique_id) ---")
    if plots_dir is None:
        logger.error("plots_dir not provided, cannot save time series plots.")
        return

    pdf_sample = None # Initialize
    try:
        # 1. Get and sample unique_id
        logger.info("Getting all unique_ids...")
        with _dask_compute_context(ddf_demand['unique_id'].unique()) as persisted_ids:
            all_unique_ids = persisted_ids[0].compute() # Compute to get Pandas Series

        if len(all_unique_ids) < n_samples:
            logger.warning(f"Total unique_id count ({len(all_unique_ids)}) is less than requested sample size ({n_samples}), using all unique_ids.")
            sampled_ids = all_unique_ids.tolist()
            n_samples = len(sampled_ids)
        else:
            logger.info(f"Randomly sampling {n_samples} unique_ids from {len(all_unique_ids):,}...")
            np.random.seed(random_state)
            sampled_ids = np.random.choice(all_unique_ids, n_samples, replace=False).tolist()
        logger.info(f"Selected unique_ids: {sampled_ids}")

        # 2. Filter and compute Pandas DataFrame
        logger.info("Filtering Dask DataFrame to get sample data...")
        ddf_sample_filtered = ddf_demand[ddf_demand['unique_id'].isin(sampled_ids)]

        logger.info("Converting sample data to Pandas DataFrame (might take time and memory)...")
        with _dask_compute_context(ddf_sample_filtered) as persisted_sample:
             pdf_sample = persisted_sample[0].compute()
        logger.info(f"Pandas DataFrame created, containing {len(pdf_sample):,} rows.")

        pdf_sample = pdf_sample.sort_values(['unique_id', 'timestamp'])

        # 3. Plot and analyze
        logger.info("Starting to plot time series for each sample and analyze frequency...")
        plt.style.use('seaborn-v0_8-whitegrid')

        for unique_id in tqdm(sampled_ids, desc="Processing samples"):
            df_id = pdf_sample[pdf_sample['unique_id'] == unique_id]
            if df_id.empty:
                logger.warning(f"No data found for unique_id '{unique_id}', skipping.")
                continue

            # Plot
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(df_id['timestamp'], df_id['y'], label=f'ID: {unique_id}')
            ax.set_title(f'Demand (y) Time Series - unique_id: {unique_id}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Electricity Demand (y) in kWh')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            _save_plot(fig, f'timeseries_sample_{unique_id}.png', plots_dir) # Use helper

            # Analyze frequency
            time_diffs = df_id['timestamp'].diff()
            if len(time_diffs) > 1:
                freq_counts = time_diffs.value_counts()
                logger.info(f"--- unique_id: {unique_id} Timestamp Interval Frequency ---")
                log_str = f"Frequency Stats (Top 5):\n{freq_counts.head().to_string()}"
                if len(freq_counts) > 5: log_str += "\n..."
                logger.info(log_str)
                if len(freq_counts) > 1:
                    logger.warning(f" unique_id '{unique_id}' has multiple time intervals detected, possible missing data or frequency change.")
            else:
                logger.info(f" unique_id '{unique_id}' has only one timestamp, cannot calculate frequency.")

        logger.info("Time series sample analysis complete.")

    except Exception as e:
        logger.exception(f"Error analyzing Demand time series samples: {e}")
    # Finally block no longer needed for explicit cleanup

def analyze_metadata_categorical(pdf_metadata, columns_to_analyze=None):
    """分析 Metadata DataFrame 中指定分类列的分布并记录。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过分类特征分析。")
        return
    if columns_to_analyze is None:
        columns_to_analyze = ['building_class', 'location', 'freq', 'timezone', 'dataset']
    logger.info(f"--- 开始分析 Metadata 分类特征分布 ({', '.join(columns_to_analyze)}) ---")

    for col in columns_to_analyze:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过。")
            continue
        _log_value_counts(pdf_metadata[col], col) # 使用辅助函数记录

    logger.info("Metadata 分类特征分析完成。")


def plot_metadata_categorical(pdf_metadata, columns_to_plot=None, top_n=10, plots_dir=None):
    """可视化 Metadata DataFrame 中指定分类列的分布并保存。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过分类特征绘图。")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存 Metadata 分类特征图。")
        return
    if columns_to_plot is None:
        columns_to_plot = ['building_class', 'location', 'freq', 'timezone', 'dataset']
    logger.info(f"--- Starting plotting Metadata categorical feature distributions ({', '.join(columns_to_plot)}) ---")

    for col in columns_to_plot:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过绘图。")
            continue
        _plot_categorical_distribution(pdf_metadata[col], col,
                                       f'metadata_distribution_{col}', plots_dir,
                                       top_n=top_n, title_prefix="Metadata ") # Use helper

    logger.info("Metadata categorical feature plotting complete.")


def analyze_metadata_numerical(pdf_metadata, columns_to_analyze=None, plots_dir=None):
    """分析 Metadata DataFrame 中指定数值列的分布并记录/绘图。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过数值特征分析。")
        return
    plot = plots_dir is not None # Whether to plot
    if not plot:
         logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Metadata 数值特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['latitude', 'longitude', 'cluster_size']
    logger.info(f"--- Starting analysis of Metadata numerical feature distributions ({', '.join(columns_to_analyze)}) ---")

    for col in columns_to_analyze:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过。")
            continue
        if not pd.api.types.is_numeric_dtype(pdf_metadata[col]):
            logger.warning(f"列 '{col}' 不是数值类型，跳过数值分析。")
            continue

        logger.info(f"--- Analyzing column: {col} ---")
        desc_stats = pdf_metadata[col].describe()
        logger.info(f"Descriptive Statistics:\n{desc_stats.to_string()}")

        missing_count = pdf_metadata[col].isnull().sum()
        if missing_count > 0:
            missing_perc = (missing_count / len(pdf_metadata) * 100).round(2)
            logger.warning(f"列 '{col}' 存在 {missing_count} ({missing_perc}%) 个缺失值。")

        if plot:
            # Use helper function to plot
            _plot_numerical_distribution(pdf_metadata[col].dropna(), col, # dropna to prevent plot errors
                                         f'metadata_distribution_{col}', plots_dir,
                                         title_prefix="Metadata ", kde=True) # Default kde=True

    logger.info("Metadata numerical feature analysis complete.")


def analyze_missing_locations(pdf_metadata):
    """分析 Metadata 中位置信息缺失的行。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过缺失位置分析。")
        return

    logger.info("--- 开始分析 Metadata 中缺失的位置信息 ---")
    location_cols = ['location_id', 'latitude', 'longitude', 'location']
    existing_location_cols = [col for col in location_cols if col in pdf_metadata.columns]
    if not existing_location_cols:
        logger.warning("Metadata 中不包含任何位置信息列，跳过分析。")
        return

    missing_mask = pdf_metadata[existing_location_cols].isnull().all(axis=1)
    missing_rows_count = missing_mask.sum()
    logger.info(f"发现 {missing_rows_count} 行的所有位置信息 ({', '.join(existing_location_cols)}) 均为缺失。")

    if missing_rows_count > 0:
        missing_df = pdf_metadata[missing_mask]
        logger.info(f"缺失位置信息的行 (前 5 行):\n{missing_df.head().to_string()}")
        logger.info("分析缺失位置信息行的其他特征分布:")
        for col in ['dataset', 'building_class', 'freq']: # 分析示例列
             if col in missing_df.columns:
                _log_value_counts(missing_df[col], f"(缺失位置) {col}", top_n=None) # 显示所有

    logger.info("缺失位置信息分析完成。")


def analyze_weather_numerical(ddf_weather, columns_to_analyze=None, plot_sample_frac=0.1, plots_dir=None, random_state=42):
    """分析 Weather Dask DataFrame 中指定数值列的分布并记录/绘图。"""
    if ddf_weather is None:
        logger.warning("输入的 Weather Dask DataFrame 为空，跳过数值特征分析。")
        return
    plot = plots_dir is not None
    if not plot:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Weather 数值特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'wind_speed_10m', 'cloud_cover']
    logger.info(f"--- Starting analysis of Weather numerical feature distributions ({', '.join(columns_to_analyze)}) ---")

    # Compute descriptive statistics
    desc_stats_all = None
    relevant_cols = [col for col in columns_to_analyze if col in ddf_weather.columns]
    if relevant_cols:
        try:
            logger.info("Computing descriptive statistics for selected weather features (might take some time)...")
            with _dask_compute_context(ddf_weather[relevant_cols].describe()) as persisted_stats:
                desc_stats_all = persisted_stats[0].compute()
            logger.info(f"Descriptive Statistics:\n{desc_stats_all.to_string()}")
        except Exception as e:
            logger.exception(f"Error computing weather feature descriptive statistics: {e}")

    # Check for negative values
    precipitation_cols = ['precipitation', 'rain', 'snowfall']
    neg_check_cols = [col for col in precipitation_cols if col in relevant_cols]
    if neg_check_cols:
        logger.info(f"Checking columns {neg_check_cols} for negative values...")
        try:
            # Compute negative counts in parallel
            neg_counts_futures = [(col, (ddf_weather[col] < 0).sum()) for col in neg_check_cols]
            with _dask_compute_context(*[f[1] for f in neg_counts_futures]) as persisted_counts:
                 neg_counts_computed = [p.compute() for p in persisted_counts]

            for i, (col, _) in enumerate(neg_counts_futures):
                 negative_count = neg_counts_computed[i]
                 if negative_count > 0:
                     logger.warning(f"Column '{col}' detected {negative_count} negative values! Check data source or handle.")
                 else:
                     logger.info(f"Column '{col}' has no negative values detected.")
        except Exception as e:
            logger.exception(f"Error checking for negative values: {e}")


    # Plot column by column (if needed)
    if plot:
        logger.info(f"Starting to plot weather feature distributions (sample fraction: {plot_sample_frac:.1%}) ...")
        for col in relevant_cols:
            if desc_stats_all is not None and col not in desc_stats_all.columns:
                 logger.warning(f"Could not compute statistics for column '{col}', skipping plot.")
                 continue

            logger.info(f"Plotting column: {col}")
            try:
                # Sample and compute Pandas Series
                logger.debug(f"Sampling and computing Pandas Series for column '{col}'...")
                with _dask_compute_context(ddf_weather[col].dropna().sample(frac=plot_sample_frac, random_state=random_state)) as persisted_sample:
                     col_sample_pd = persisted_sample[0].compute()
                logger.debug(f"Column '{col}' sampling complete, sample size: {len(col_sample_pd):,}")

                if col_sample_pd.empty:
                    logger.warning(f"Sample result for column '{col}' is empty, skipping plot.")
                    continue

                # Use helper function to plot
                _plot_numerical_distribution(col_sample_pd, col,
                                             f'weather_distribution_{col}', plots_dir,
                                             title_prefix="Weather ", kde=True)
            except Exception as e:
                logger.exception(f"Error plotting distribution for column '{col}': {e}")

    logger.info("Weather numerical feature analysis complete.")


def analyze_weather_categorical(ddf_weather, columns_to_analyze=None, top_n=20, plots_dir=None):
    """分析 Weather Dask DataFrame 中指定分类列的分布并记录/绘图。"""
    if ddf_weather is None:
        logger.warning("输入的 Weather Dask DataFrame 为空，跳过分类特征分析。")
        return
    plot = plots_dir is not None
    if not plot:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Weather 分类特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['weather_code']
    logger.info(f"--- Starting analysis of Weather categorical feature distributions ({', '.join(columns_to_analyze)}) ---")

    relevant_cols = [col for col in columns_to_analyze if col in ddf_weather.columns]

    for col in relevant_cols:
        logger.info(f"--- Analyzing column: {col} ---")
        try:
            logger.info(f"Computing value counts for column '{col}'...")
            with _dask_compute_context(ddf_weather[col].value_counts()) as persisted_counts:
                 counts_pd = persisted_counts[0].compute() # Get Pandas Series

            # Log value counts (using Pandas Series)
            _log_value_counts(counts_pd, col, top_n=top_n) # Note: Passing Series, % recalculated inside

            # Plot (if needed)
            if plot and not counts_pd.empty:
                 # Pass counts_pd to plotting function (would need adjustment or plot manually here)
                 # Current _plot_categorical_distribution expects raw Series, so plot manually with counts
                logger.info(f"Plotting column: {col}")
                fig, ax = plt.subplots(figsize=(12, 6))
                data_to_plot = counts_pd.head(top_n)
                num_unique = len(counts_pd)
                title = f'Weather Top {top_n} {col} Distribution'
                if num_unique > top_n:
                    title += " (Others not shown)"

                # Try converting index to int then str, fallback to just str
                try:
                    plot_index = data_to_plot.index.astype(int).astype(str)
                    order = plot_index # Sort by WMO code
                except (ValueError, TypeError):
                    plot_index = data_to_plot.index.astype(str)
                    order = plot_index # Sort by count

                sns.barplot(x=plot_index, y=data_to_plot.values, ax=ax, palette="viridis", order=order)
                ax.set_title(title)
                ax.set_xlabel(f'{col} (WMO Code)' if col == 'weather_code' else col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                _save_plot(fig, f'weather_distribution_{col}.png', plots_dir)

        except Exception as e:
            logger.exception(f"Error analyzing column '{col}': {e}")

    logger.info("Weather categorical feature analysis complete.")


def analyze_demand_vs_metadata(ddf_demand, pdf_metadata, plots_dir=None, sample_frac=0.001, random_state=42):
    """分析 Demand (y) 与 Metadata 特征 (如 building_class) 的关系。"""
    if ddf_demand is None or pdf_metadata is None:
        logger.warning("Need Demand and Metadata data for relationship analysis.")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存关系分析图。")
        return

    target_col = 'building_class'
    logger.info(f"--- Starting analysis of Demand vs {target_col} relationship (sample fraction: {sample_frac:.1%}) ---")

    try:
        pdf_merged = _merge_demand_metadata_sample(ddf_demand, pdf_metadata, [target_col], sample_frac, random_state)

        if pdf_merged.empty:
             logger.warning(f"Failed to get merged data, cannot analyze Demand vs {target_col}.")
             return

        logger.info(f"Plotting Demand (y) vs {target_col} box plot...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- Original scale ---
        fig_orig, ax_orig = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=pdf_merged, x=target_col, y='y', showfliers=False, ax=ax_orig, palette="viridis")
        ax_orig.set_title(f'Demand (y) Distribution by {target_col} (Original Scale, No Outliers)')
        ax_orig.set_xlabel(target_col)
        ax_orig.set_ylabel('Electricity Demand (y) in kWh')
        plt.tight_layout()
        _save_plot(fig_orig, f'demand_vs_{target_col}_boxplot_orig.png', plots_dir)

        # --- Log scale ---
        epsilon = 1e-6
        pdf_merged['y_log1p'] = np.log1p(pdf_merged['y'][pdf_merged['y'] >= 0] + epsilon)

        if pdf_merged['y_log1p'].isnull().all():
             logger.warning("No valid log values to plot, skipping log scale box plot.")
        else:
            fig_log, ax_log = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=pdf_merged.dropna(subset=['y_log1p']), x=target_col, y='y_log1p', showfliers=True, ax=ax_log, palette="viridis")
            ax_log.set_title(f'Demand (y) Distribution by {target_col} (Log1p Scale)')
            ax_log.set_xlabel(target_col)
            ax_log.set_ylabel('log1p(Electricity Demand (y) + epsilon)')
            plt.tight_layout()
            _save_plot(fig_log, f'demand_vs_{target_col}_boxplot_log1p.png', plots_dir)

        logger.info(f"Demand vs {target_col} analysis complete.")

    except Exception as e:
        logger.exception(f"Error analyzing Demand vs {target_col} relationship: {e}")


def analyze_demand_vs_location(ddf_demand, pdf_metadata, plots_dir=None, sample_frac=0.001, top_n=10, random_state=42):
    """分析 Demand (y) 与 Top N location 的关系。"""
    if ddf_demand is None or pdf_metadata is None:
        logger.warning("Need Demand and Metadata data for relationship analysis.")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存关系分析图。")
        return

    target_col = 'location'
    logger.info(f"--- Starting analysis of Demand vs {target_col} relationship (Top {top_n}, sample fraction: {sample_frac:.1%}) ---")

    try:
        # Preprocess location column before merging
        pdf_metadata_processed = pdf_metadata.copy()
        if target_col in pdf_metadata_processed.columns:
             pdf_metadata_processed[target_col] = pdf_metadata_processed[target_col].fillna('Missing')
        else:
             logger.error(f"Metadata is missing '{target_col}' column, cannot proceed.")
             return

        pdf_merged = _merge_demand_metadata_sample(ddf_demand, pdf_metadata_processed, [target_col], sample_frac, random_state)

        if pdf_merged.empty:
             logger.warning(f"Failed to get merged data, cannot analyze Demand vs {target_col}.")
             return

        # Determine Top N locations
        location_counts = pdf_merged[target_col].value_counts()
        top_locations = location_counts.head(top_n).index.tolist()
        logger.info(f"Top {top_n} locations (by data points in sample): {top_locations}")

        pdf_merged_top_n = pdf_merged[pdf_merged[target_col].isin(top_locations)].copy()
        if pdf_merged_top_n.empty:
             logger.warning(f"Could not find data for Top {top_n} locations, cannot plot.")
             return

        logger.info(f"Plotting Demand (y) vs Top {top_n} {target_col} box plot...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- Original scale ---
        fig_orig, ax_orig = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=pdf_merged_top_n, x=target_col, y='y', showfliers=False, ax=ax_orig, palette="viridis", order=top_locations)
        ax_orig.set_title(f'Demand (y) Distribution by Top {top_n} {target_col} (Original Scale, No Outliers)')
        ax_orig.set_xlabel(target_col)
        ax_orig.set_ylabel('Electricity Demand (y) in kWh')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        _save_plot(fig_orig, f'demand_vs_top{top_n}_{target_col}_boxplot_orig.png', plots_dir)

        # --- Log scale ---
        epsilon = 1e-6
        pdf_merged_top_n['y_log1p'] = np.log1p(pdf_merged_top_n['y'][pdf_merged_top_n['y'] >= 0] + epsilon)

        if pdf_merged_top_n['y_log1p'].isnull().all():
             logger.warning("No valid log values to plot, skipping log scale box plot.")
        else:
            fig_log, ax_log = plt.subplots(figsize=(15, 7))
            sns.boxplot(data=pdf_merged_top_n.dropna(subset=['y_log1p']), x=target_col, y='y_log1p', showfliers=True, ax=ax_log, palette="viridis", order=top_locations)
            ax_log.set_title(f'Demand (y) Distribution by Top {top_n} {target_col} (Log1p Scale)')
            ax_log.set_xlabel(target_col)
            ax_log.set_ylabel('log1p(Electricity Demand (y) + epsilon)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            _save_plot(fig_log, f'demand_vs_top{top_n}_{target_col}_boxplot_log1p.png', plots_dir)

        logger.info(f"Demand vs {target_col} analysis complete.")

    except Exception as e:
        logger.exception(f"Error analyzing Demand vs {target_col} relationship: {e}")


def analyze_demand_vs_weather(ddf_demand, pdf_metadata, ddf_weather, plots_dir=None, n_sample_ids=50, plot_sample_frac=0.1, random_state=42):
    """分析 Demand (y) 与 Weather 特征的关系 (抽样 unique_id)。"""
    if ddf_demand is None or pdf_metadata is None or ddf_weather is None:
        logger.warning("Need Demand, Metadata, and Weather data for Demand vs Weather analysis.")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存 Demand vs Weather 分析图。")
        return

    logger.info(f"--- Starting analysis of Demand vs Weather relationship (sampling {n_sample_ids} unique_ids) ---")
    pdf_merged = pd.DataFrame() # Initialize

    try:
        # 1. Sample unique_id
        logger.info(f"Randomly sampling {n_sample_ids} unique_ids from Metadata...")
        valid_ids = pdf_metadata['unique_id'].unique()
        if len(valid_ids) < n_sample_ids:
            logger.warning(f"Number of unique_ids in Metadata ({len(valid_ids)}) is less than requested sample size ({n_sample_ids}), using all available IDs.")
            sampled_unique_ids = valid_ids
            n_sample_ids = len(sampled_unique_ids)
        else:
            np.random.seed(random_state)
            sampled_unique_ids = np.random.choice(valid_ids, n_sample_ids, replace=False)
        logger.info(f"Selected unique_ids (first 10): {sampled_unique_ids[:10].tolist()}...")

        # 2. Prepare Demand data (filter + get location_id)
        logger.info("Filtering Demand data and merging Metadata to get location_id...")
        pdf_metadata_sample = pdf_metadata[pdf_metadata['unique_id'].isin(sampled_unique_ids)][['unique_id', 'location_id']].drop_duplicates().dropna(subset=['location_id'])
        if pdf_metadata_sample.empty:
            logger.warning("Sampled unique_ids have no valid location_id in Metadata, cannot proceed.")
            return

        ddf_demand_filtered = ddf_demand[ddf_demand['unique_id'].isin(sampled_unique_ids)][['unique_id', 'timestamp', 'y']].dropna(subset=['y'])
        # Dask merge Dask DF with Pandas DF
        ddf_demand_with_loc = dd.merge(ddf_demand_filtered, pdf_metadata_sample, on='unique_id', how='inner')

        # 3. Prepare Weather data
        weather_cols = ['location_id', 'timestamp', 'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'precipitation', 'wind_speed_10m', 'cloud_cover']
        ddf_weather_subset = ddf_weather[[col for col in weather_cols if col in ddf_weather.columns]] # Ensure columns exist

        # 4. Merge (convert to Pandas for merge_asof, as Dask's is still unstable)
        logger.info("Computing sampled Demand (with location_id) to Pandas DataFrame...")
        with _dask_compute_context(ddf_demand_with_loc) as p_demand:
            pdf_demand_with_loc = p_demand[0].compute()
        num_demand_rows = len(pdf_demand_with_loc)
        logger.info(f"Demand Pandas DataFrame computed, containing {num_demand_rows:,} rows.")
        if num_demand_rows == 0:
             logger.warning("Failed to merge location_id into Demand data, cannot proceed.")
             return

        relevant_location_ids = pdf_demand_with_loc['location_id'].unique()
        logger.info(f"Filtering Weather data, keeping only {len(relevant_location_ids)} relevant location_ids...")
        ddf_weather_filtered = ddf_weather_subset[ddf_weather_subset['location_id'].isin(relevant_location_ids)]

        logger.info("Computing filtered Weather data to Pandas DataFrame...")
        with _dask_compute_context(ddf_weather_filtered) as p_weather:
             pdf_weather_filtered = p_weather[0].compute()
        logger.info(f"Weather Pandas DataFrame computed, containing {len(pdf_weather_filtered):,} rows.")


        logger.info("Preparing data for merge (sorting, deduplicating, type conversion)...")
        # Drop NaN, sort, deduplicate
        for df, name in [(pdf_demand_with_loc, "Demand"), (pdf_weather_filtered, "Weather")]:
             rows_before = len(df)
             df.dropna(subset=['location_id', 'timestamp'], inplace=True)
             rows_after_na = len(df)
             df.sort_values(by=['location_id', 'timestamp'], inplace=True)
             df.drop_duplicates(subset=['location_id', 'timestamp'], keep='first', inplace=True)
             rows_after_dedup = len(df)
             df.reset_index(drop=True, inplace=True)
             if rows_before > rows_after_na: logger.warning(f"Removed {rows_before - rows_after_na} rows from {name} data (due to NaN location_id or timestamp)")
             if rows_after_na > rows_after_dedup: logger.warning(f"Removed {rows_after_na - rows_after_dedup} duplicate (location_id, timestamp) rows from {name} data")
             # Try converting location_id type
             try:
                 df['location_id'] = df['location_id'].astype(str)
             except Exception as e:
                 logger.error(f"Failed to convert {name}'s location_id type: {e}")
                 return # Critical error, cannot proceed

        # Double-check sorting (merge_asof requires it strictly)
        is_left_sorted = pdf_demand_with_loc.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing).all()
        is_right_sorted = pdf_weather_filtered.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing).all()

        if not is_left_sorted or not is_right_sorted:
             logger.error(f"Data sorting check failed! Left sorted: {is_left_sorted}, Right sorted: {is_right_sorted}. Cannot safely execute merge_asof.")
             # Add more detailed diagnostics here, find which group failed
             if not is_left_sorted:
                 bad_groups = pdf_demand_with_loc.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing)
                 logger.error(f"Left side sorting failed for location_ids: {bad_groups[~bad_groups].index.tolist()}")
             if not is_right_sorted:
                 bad_groups = pdf_weather_filtered.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing)
                 logger.error(f"Right side sorting failed for location_ids: {bad_groups[~bad_groups].index.tolist()}")
             # === IMPORTANT: Mark error and skip merge here ===
             logger.error("Skipping merge_asof step due to sorting issues.")

             return # Return directly, do not proceed

        else:
            logger.info("Data sorting check passed.")
            logger.info("Merging data using Pandas merge_asof...")
            try:
                pdf_merged = pd.merge_asof(
                    pdf_demand_with_loc,
                    pdf_weather_filtered,
                    on='timestamp',
                    by='location_id',
                    direction='backward',
                    tolerance=pd.Timedelta('1hour')
                )
                logger.info(f"Final merge complete, resulting in {len(pdf_merged):,} rows.")
            except ValueError as ve:
                 # Can still fail sometimes due to edge cases or type issues even if check passes
                 logger.error(f"Pandas merge_asof failed unexpectedly (even after sort check): {ve}")
                 logger.exception("Error during merge_asof execution")
                 return # Return on failure
            except Exception as merge_exc:
                 logger.exception("Unknown error during data merge")
                 return # Return on failure

        if pdf_merged.empty:
             logger.warning("Final merged data is empty, cannot perform relationship analysis.")
             return

        # 5. Calculate correlation
        logger.info("Calculating correlations between Demand (y) and weather features...")
        correlation_cols = ['y'] + [col for col in weather_cols if col in pdf_merged.columns and col not in ['location_id', 'timestamp']]
        # Ensure all correlation columns are numeric
        numeric_correlation_cols = [col for col in correlation_cols if pd.api.types.is_numeric_dtype(pdf_merged[col])]
        if len(numeric_correlation_cols) < len(correlation_cols):
             skipped_cols = set(correlation_cols) - set(numeric_correlation_cols)
             logger.warning(f"The following columns are not numeric and will be excluded from correlation calculation: {skipped_cols}")

        if 'y' not in numeric_correlation_cols or len(numeric_correlation_cols) <= 1:
             logger.warning("Not enough numeric columns (including 'y') to calculate correlations.")
        else:
            correlation_matrix = pdf_merged[numeric_correlation_cols].corr()
            logger.info(f"Correlation Matrix ('y' vs Weather):\n{correlation_matrix['y'].to_string()}")

            # 6. Visualize
            # --- Correlation heatmap ---
            logger.info("Plotting correlation heatmap...")
            plt.style.use('seaborn-v0_8-whitegrid')
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
            ax_corr.set_title(f'Correlation Matrix: Demand (y) vs Weather Features (Sampled {n_sample_ids} IDs)')
            plt.tight_layout()
            _save_plot(fig_corr, 'demand_vs_weather_correlation_heatmap.png', plots_dir)

            # --- Scatter plots ---
            scatter_cols = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature']
            scatter_cols = [col for col in scatter_cols if col in numeric_correlation_cols] # Only plot existing numeric columns
            logger.info(f"Plotting scatter plots for Demand (y) vs {', '.join(scatter_cols)} (plot sample fraction: {plot_sample_frac:.1%})")

            plot_sample = pdf_merged.sample(frac=plot_sample_frac, random_state=random_state) if len(pdf_merged) * plot_sample_frac >= 1 else pdf_merged

            for col in scatter_cols:
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=plot_sample, x=col, y='y', alpha=0.5, s=10, ax=ax_scatter)
                ax_scatter.set_title(f'Demand (y) vs {col} (Sampled Data)')
                ax_scatter.set_xlabel(col)
                ax_scatter.set_ylabel('Electricity Demand (y) in kWh')
                corr_val = correlation_matrix.loc['y', col]
                ax_scatter.text(0.05, 0.95, f'Correlation: {corr_val:.3f}', transform=ax_scatter.transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                plt.tight_layout()
                _save_plot(fig_scatter, f'demand_vs_weather_{col}_scatterplot.png', plots_dir)

        logger.info("Demand vs Weather analysis complete.")

    except Exception as e:
        logger.exception(f"Error analyzing Demand vs Weather relationship: {e}")
    # Finally block no longer needed for explicit cleanup


def main():
    """主执行函数，编排 EDA 步骤。"""
    logger.info("=========================================")
    logger.info("===        开始执行 EDA 脚本          ===")
    logger.info("=========================================")

    # --- 加载数据 ---
    logger.info("--- 步骤 1: 加载数据 ---")
    ddf_demand = load_demand_data()
    pdf_metadata = load_metadata()
    ddf_weather = load_weather_data()

    # --- 单变量分析 ---
    logger.info("--- 步骤 2: 单变量分析 ---")

    # Demand 'y' 分析
    logger.info("--- 开始 Demand 'y' 分析 ---")
    y_sample_pd = analyze_demand_y_distribution(ddf_demand, sample_frac=0.005)
    if y_sample_pd is not None:
         plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000)
    # analyze_demand_timeseries_sample(ddf_demand, n_samples=3, plots_dir=plots_dir) # 减少样本量加快速度
    logger.info("--- 完成 Demand 'y' 分布分析 ---")


    # Metadata 分析
    logger.info("--- 开始 Metadata 分析 ---")
    # analyze_metadata_categorical(pdf_metadata)
    # plot_metadata_categorical(pdf_metadata, plots_dir=plots_dir, top_n=10)
    # analyze_metadata_numerical(pdf_metadata, plots_dir=plots_dir)
    # analyze_missing_locations(pdf_metadata)
    logger.info("--- 完成 Metadata 分析 (或已注释) ---")


    # Weather 分析
    logger.info("--- 开始 Weather 分析 ---")
    # analyze_weather_numerical(ddf_weather, plots_dir=plots_dir, plot_sample_frac=0.05) # 减少抽样比例
    # analyze_weather_categorical(ddf_weather, plots_dir=plots_dir, top_n=15)
    # analyze_weather_timestamp_frequency(ddf_weather, sample_frac=0.005) # 减少抽样比例
    logger.info("--- 完成 Weather 分析 (或已注释) ---")


    # --- 关系分析 ---
    logger.info("--- 步骤 3: 关系分析 ---")

    # Demand vs Metadata (building_class)
    # logger.info("--- 开始 Demand vs building_class 分析 ---")
    # analyze_demand_vs_metadata(ddf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001)
    # logger.info("--- 完成 Demand vs building_class 分析 (或已注释) ---")

    # Demand vs Metadata (location)
    # logger.info("--- 开始 Demand vs location 分析 ---")
    # analyze_demand_vs_location(ddf_demand, pdf_metadata, plots_dir=plots_dir, sample_frac=0.001, top_n=5) # 减少 TopN
    # logger.info("--- 完成 Demand vs location 分析 (或已注释) ---")

    # Demand vs Weather (可能跳过)
    # logger.info("--- 开始 Demand vs Weather 分析 ---")
    # try:
    #     analyze_demand_vs_weather(ddf_demand, pdf_metadata, ddf_weather, plots_dir=plots_dir, n_sample_ids=50)
    # except Exception as e:
    #      # analyze_demand_vs_weather 内部已记录详细错误，这里只记录简要信息
    #      logger.error(f"Demand vs Weather 分析执行期间遇到问题: {e}")
    # logger.info("--- 完成 Demand vs Weather 分析 (或已跳过) ---")


    logger.info("=========================================")
    logger.info("===        EDA 脚本执行完毕           ===")
    logger.info("=========================================")


if __name__ == "__main__":
    # --- 执行入口点 ---
    # 注释掉之前运行成功的函数入口，只保留当前需要分析的
    main()

    # --- 示例：只运行特定分析 ---
    # logger.info("--- 单独运行 Demand vs Weather 分析 ---")
    # ddf_demand = load_demand_data()
    # pdf_metadata = load_metadata()
    # ddf_weather = load_weather_data()
    # analyze_demand_vs_weather(ddf_demand, pdf_metadata, ddf_weather, plots_dir=plots_dir, n_sample_ids=50)
    # logger.info("--- 单独分析完成 ---")