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
        logger.warning(f"列 '{col_name}' 的数据为空，跳过绘图。")
        return

    logger.info(f"绘制列 '{col_name}' 的分布图 (样本量: {len(data_pd_series):,})...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    try:
        # 直方图
        sns.histplot(data_pd_series, kde=kde, ax=axes[0])
        axes[0].set_title(f'{title_prefix}{col_name} 分布')
        axes[0].set_xlabel(col_name)
        axes[0].set_ylabel('频数')

        # 箱线图
        sns.boxplot(x=data_pd_series, ax=axes[1], showfliers=showfliers)
        axes[1].set_title(f'{title_prefix}{col_name} 箱线图')
        axes[1].set_xlabel(col_name)

        plt.tight_layout()
        _save_plot(fig, f"{filename_base}.png", plots_dir)
    except Exception as e:
        logger.exception(f"绘制列 '{col_name}' 分布图时出错: {e}")
        plt.close(fig) # 确保关闭图形

def _plot_categorical_distribution(data_pd_series, col_name, filename_base, plots_dir, top_n=10, title_prefix=""):
    """绘制分类 Pandas Series 的分布图 (条形图)。"""
    if data_pd_series is None or data_pd_series.empty:
        logger.warning(f"列 '{col_name}' 的数据为空，跳过绘图。")
        return

    logger.info(f"绘制列 '{col_name}' 的分类分布图 (Top {top_n})...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        value_counts = data_pd_series.value_counts(dropna=False) # 包含 NaN
        num_unique = len(value_counts)

        if num_unique > top_n:
            logger.info(f"列 '{col_name}' 唯一值数量 ({num_unique}) 超过 {top_n}，将仅显示 Top {top_n} 和 '其他'。")
            top_values = value_counts.head(top_n)
            other_count = value_counts.iloc[top_n:].sum()
            if other_count > 0:
                others_label = '其他 (Others)'
                others_series = pd.Series([other_count], index=[others_label])
                if others_label in top_values.index: # 防御性检查
                    top_values[others_label] += other_count
                else:
                    top_values = pd.concat([top_values, others_series])
            data_to_plot = top_values
            title = f'{title_prefix}Top {top_n} {col_name} 分布'
        else:
            data_to_plot = value_counts
            title = f'{title_prefix}{col_name} 分布'

        # 处理 NaN 标签以便绘图，并将索引排序以获得一致的绘图顺序（如果适用）
        plot_index = data_to_plot.index.astype(str)
        # 尝试按原值排序（数值或字符串），如果失败则按计数排序
        try:
            sort_order = data_to_plot.sort_index().index.astype(str)
            plot_index_ordered = [idx for idx in sort_order if idx in plot_index] # 保持过滤后的索引
        except TypeError: # 不能混合排序
            sort_order = data_to_plot.index.astype(str) # 按出现顺序（计数排序）
            plot_index_ordered = plot_index # 使用原始顺序

        sns.barplot(x=plot_index, y=data_to_plot.values, ax=ax, palette="viridis", order=plot_index_ordered)
        ax.set_title(title)
        ax.set_xlabel(col_name)
        ax.set_ylabel('计数')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        _save_plot(fig, f"{filename_base}.png", plots_dir)

    except Exception as e:
        logger.exception(f"绘制列 '{col_name}' 分布图时出错: {e}")
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
        logger.warning("输入的 'y' 样本为空，跳过绘图。")
        return

    # --- 进一步抽样用于绘图 ---
    if len(y_sample_pd) > plot_sample_size:
        logger.info(f"原始样本量 {len(y_sample_pd):,} 较大，进一步抽样 {plot_sample_size:,} 个点用于绘图。")
        y_plot_sample = y_sample_pd.sample(n=plot_sample_size, random_state=random_state)
    else:
        y_plot_sample = y_sample_pd
    logger.info(f"--- 开始绘制 Demand 'y' 列分布图 (绘图样本量: {len(y_plot_sample):,}) ---")

    # --- 绘制原始尺度 ---
    _plot_numerical_distribution(y_plot_sample, 'y',
                                 'demand_y_distribution_original_scale', plots_dir,
                                 title_prefix="Demand ", kde=False, showfliers=False)

    # --- 绘制对数尺度 ---
    epsilon = 1e-6
    y_plot_sample_non_neg = y_plot_sample[y_plot_sample >= 0]
    if y_plot_sample_non_neg.empty:
        logger.warning("绘图样本中没有非负值，跳过对数尺度绘图。")
        return

    y_log_transformed = np.log1p(y_plot_sample_non_neg + epsilon)
    _plot_numerical_distribution(y_log_transformed, 'log1p(y + epsilon)',
                                 'demand_y_distribution_log1p_scale', plots_dir,
                                 title_prefix="Demand ", kde=True, showfliers=True)

    logger.info("Demand 'y' 列分布图绘制完成。")


def analyze_demand_timeseries_sample(ddf_demand, n_samples=5, plots_dir=None, random_state=42):
    """抽取 n_samples 个 unique_id 的完整时间序列数据，绘制图形并分析频率。"""
    logger.info(f"--- 开始分析 Demand 时间序列特性 (抽样 {n_samples} 个 unique_id) ---")
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存时间序列图。")
        return

    pdf_sample = None # 初始化
    try:
        # 1. 获取并抽样 unique_id
        logger.info("获取所有 unique_id...")
        with _dask_compute_context(ddf_demand['unique_id'].unique()) as persisted_ids:
            all_unique_ids = persisted_ids[0].compute() # 计算得到 Pandas Series

        if len(all_unique_ids) < n_samples:
            logger.warning(f"总 unique_id 数量 ({len(all_unique_ids)}) 少于请求的样本数量 ({n_samples})，将使用所有 unique_id。")
            sampled_ids = all_unique_ids.tolist()
            n_samples = len(sampled_ids)
        else:
            logger.info(f"从 {len(all_unique_ids):,} 个 unique_id 中随机抽取 {n_samples} 个...")
            np.random.seed(random_state)
            sampled_ids = np.random.choice(all_unique_ids, n_samples, replace=False).tolist()
        logger.info(f"选取的 unique_id: {sampled_ids}")

        # 2. 筛选并计算 Pandas DataFrame
        logger.info("筛选 Dask DataFrame 以获取样本数据...")
        ddf_sample_filtered = ddf_demand[ddf_demand['unique_id'].isin(sampled_ids)]

        logger.info("将样本数据转换为 Pandas DataFrame (可能需要较长时间和较多内存)...")
        with _dask_compute_context(ddf_sample_filtered) as persisted_sample:
             pdf_sample = persisted_sample[0].compute()
        logger.info(f"Pandas DataFrame 创建完成，包含 {len(pdf_sample):,} 行数据。")

        pdf_sample = pdf_sample.sort_values(['unique_id', 'timestamp'])

        # 3. 绘制并分析
        logger.info("开始为每个样本绘制时间序列图并分析频率...")
        plt.style.use('seaborn-v0_8-whitegrid')

        for unique_id in tqdm(sampled_ids, desc="处理样本中"):
            df_id = pdf_sample[pdf_sample['unique_id'] == unique_id]
            if df_id.empty:
                logger.warning(f"未找到 unique_id '{unique_id}' 的数据，跳过。")
                continue

            # 绘图
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(df_id['timestamp'], df_id['y'], label=f'ID: {unique_id}')
            ax.set_title(f'Demand (y) 时间序列 - unique_id: {unique_id}')
            ax.set_xlabel('时间戳')
            ax.set_ylabel('电力需求 (y) in kWh')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            _save_plot(fig, f'timeseries_sample_{unique_id}.png', plots_dir) # 使用辅助函数

            # 分析频率
            time_diffs = df_id['timestamp'].diff()
            if len(time_diffs) > 1:
                freq_counts = time_diffs.value_counts()
                logger.info(f"--- unique_id: {unique_id} 时间戳间隔频率 ---")
                log_str = f"频率统计 (Top 5):\n{freq_counts.head().to_string()}"
                if len(freq_counts) > 5: log_str += "\n..."
                logger.info(log_str)
                if len(freq_counts) > 1:
                    logger.warning(f" unique_id '{unique_id}' 检测到多种时间间隔，可能存在数据缺失或频率变化。")
            else:
                logger.info(f" unique_id '{unique_id}' 只有一个时间点，无法计算频率。")

        logger.info("时间序列样本分析完成。")

    except Exception as e:
        logger.exception(f"分析 Demand 时间序列样本时发生错误: {e}")
    # finally 中不再需要显式清理，由 _dask_compute_context 处理

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
    logger.info(f"--- 开始绘制 Metadata 分类特征分布图 ({', '.join(columns_to_plot)}) ---")

    for col in columns_to_plot:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过绘图。")
            continue
        _plot_categorical_distribution(pdf_metadata[col], col,
                                       f'metadata_distribution_{col}', plots_dir,
                                       top_n=top_n, title_prefix="Metadata ") # 使用辅助函数

    logger.info("Metadata 分类特征绘图完成。")


def analyze_metadata_numerical(pdf_metadata, columns_to_analyze=None, plots_dir=None):
    """分析 Metadata DataFrame 中指定数值列的分布并记录/绘图。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过数值特征分析。")
        return
    plot = plots_dir is not None # 是否绘图
    if not plot:
         logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Metadata 数值特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['latitude', 'longitude', 'cluster_size']
    logger.info(f"--- 开始分析 Metadata 数值特征分布 ({', '.join(columns_to_analyze)}) ---")

    for col in columns_to_analyze:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过。")
            continue
        if not pd.api.types.is_numeric_dtype(pdf_metadata[col]):
            logger.warning(f"列 '{col}' 不是数值类型，跳过数值分析。")
            continue

        logger.info(f"--- 分析列: {col} ---")
        desc_stats = pdf_metadata[col].describe()
        logger.info(f"描述性统计:\n{desc_stats.to_string()}")

        missing_count = pdf_metadata[col].isnull().sum()
        if missing_count > 0:
            missing_perc = (missing_count / len(pdf_metadata) * 100).round(2)
            logger.warning(f"列 '{col}' 存在 {missing_count} ({missing_perc}%) 个缺失值。")

        if plot:
            # 使用辅助函数绘图
            _plot_numerical_distribution(pdf_metadata[col].dropna(), col, # dropna 以防绘图错误
                                         f'metadata_distribution_{col}', plots_dir,
                                         title_prefix="Metadata ", kde=True) # 默认 kde=True

    logger.info("Metadata 数值特征分析完成。")


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
    logger.info(f"--- 开始分析 Weather 数值特征分布 ({', '.join(columns_to_analyze)}) ---")

    # 计算描述性统计
    desc_stats_all = None
    relevant_cols = [col for col in columns_to_analyze if col in ddf_weather.columns]
    if relevant_cols:
        try:
            logger.info("计算选中天气特征的描述性统计信息 (可能需要一些时间)...")
            with _dask_compute_context(ddf_weather[relevant_cols].describe()) as persisted_stats:
                desc_stats_all = persisted_stats[0].compute()
            logger.info(f"描述性统计:\n{desc_stats_all.to_string()}")
        except Exception as e:
            logger.exception(f"计算天气特征描述性统计时出错: {e}")

    # 检查负值
    precipitation_cols = ['precipitation', 'rain', 'snowfall']
    neg_check_cols = [col for col in precipitation_cols if col in relevant_cols]
    if neg_check_cols:
        logger.info(f"检查列 {neg_check_cols} 是否存在负值...")
        try:
            # 并行计算负值计数
            neg_counts_futures = [(col, (ddf_weather[col] < 0).sum()) for col in neg_check_cols]
            with _dask_compute_context(*[f[1] for f in neg_counts_futures]) as persisted_counts:
                 neg_counts_computed = [p.compute() for p in persisted_counts]

            for i, (col, _) in enumerate(neg_counts_futures):
                 negative_count = neg_counts_computed[i]
                 if negative_count > 0:
                     logger.warning(f"列 '{col}' 检测到 {negative_count} 个负值！需要检查数据源或处理。")
                 else:
                     logger.info(f"列 '{col}' 未检测到负值。")
        except Exception as e:
            logger.exception(f"检查负值时出错: {e}")


    # 逐列绘图 (如果需要)
    if plot:
        logger.info(f"开始绘制天气特征分布图 (抽样比例: {plot_sample_frac:.1%}) ...")
        for col in relevant_cols:
            if desc_stats_all is not None and col not in desc_stats_all.columns:
                 logger.warning(f"列 '{col}' 未能计算统计信息，跳过绘图。")
                 continue

            logger.info(f"绘制列: {col}")
            try:
                # 抽样并计算 Pandas Series
                logger.debug(f"为列 '{col}' 抽样并计算 Pandas Series...")
                with _dask_compute_context(ddf_weather[col].dropna().sample(frac=plot_sample_frac, random_state=random_state)) as persisted_sample:
                     col_sample_pd = persisted_sample[0].compute()
                logger.debug(f"列 '{col}' 抽样完成，样本量: {len(col_sample_pd):,}")

                if col_sample_pd.empty:
                    logger.warning(f"列 '{col}' 抽样结果为空，跳过绘图。")
                    continue

                # 使用辅助函数绘图
                _plot_numerical_distribution(col_sample_pd, col,
                                             f'weather_distribution_{col}', plots_dir,
                                             title_prefix="Weather ", kde=True)
            except Exception as e:
                logger.exception(f"绘制列 '{col}' 的分布图时出错: {e}")

    logger.info("Weather 数值特征分析完成。")


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
    logger.info(f"--- 开始分析 Weather 分类特征分布 ({', '.join(columns_to_analyze)}) ---")

    relevant_cols = [col for col in columns_to_analyze if col in ddf_weather.columns]

    for col in relevant_cols:
        logger.info(f"--- 分析列: {col} ---")
        try:
            logger.info(f"计算列 '{col}' 的值计数...")
            with _dask_compute_context(ddf_weather[col].value_counts()) as persisted_counts:
                 counts_pd = persisted_counts[0].compute() # 得到 Pandas Series

            # 记录值计数 (使用 Pandas Series)
            _log_value_counts(counts_pd, col, top_n=top_n) # 注意：这里传 Series，内部会重算百分比

            # 绘图 (如果需要)
            if plot and not counts_pd.empty:
                 # 将 counts_pd 传给绘图函数（需要调整绘图函数以接受 counts 或原始数据）
                 # 当前 _plot_categorical_distribution 需要原始 Series，所以我们传入计算好的 counts 来手动绘图
                logger.info(f"绘制列: {col}")
                fig, ax = plt.subplots(figsize=(12, 6))
                data_to_plot = counts_pd.head(top_n)
                num_unique = len(counts_pd)
                title = f'Weather Top {top_n} {col} 分布'
                if num_unique > top_n:
                    title += " (其他未显示)"

                # 尝试将索引转为整数再转字符串，失败则直接转字符串
                try:
                    plot_index = data_to_plot.index.astype(int).astype(str)
                    order = plot_index # 按 WMO 代码排序
                except (ValueError, TypeError):
                    plot_index = data_to_plot.index.astype(str)
                    order = plot_index # 按计数排序

                sns.barplot(x=plot_index, y=data_to_plot.values, ax=ax, palette="viridis", order=order)
                ax.set_title(title)
                ax.set_xlabel(f'{col} (WMO Code)' if col == 'weather_code' else col)
                ax.set_ylabel('计数')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                _save_plot(fig, f'weather_distribution_{col}.png', plots_dir)

        except Exception as e:
            logger.exception(f"分析列 '{col}' 时出错: {e}")

    logger.info("Weather 分类特征分析完成。")


def analyze_weather_timestamp_frequency(ddf_weather, sample_frac=0.01, random_state=42):
    """分析 Weather Dask DataFrame 时间戳频率 (基于抽样)。"""
    if ddf_weather is None:
        logger.warning("输入的 Weather Dask DataFrame 为空，跳过时间戳频率分析。")
        return
    logger.info(f"--- 开始分析 Weather 时间戳频率 (抽样比例: {sample_frac:.1%}) ---")
    pdf_sample = None
    try:
        logger.info("抽取样本数据进行频率分析...")
        ddf_sample_cols = ddf_weather[['location_id', 'timestamp']]
        with _dask_compute_context(ddf_sample_cols.sample(frac=sample_frac, random_state=random_state)) as persisted_sample:
             pdf_sample = persisted_sample[0].compute()
        logger.info(f"样本 Pandas DataFrame 创建完成，包含 {len(pdf_sample):,} 行数据。")

        if pdf_sample.empty:
             logger.warning("样本数据为空，无法分析频率。")
             return

        logger.info("按 location_id 分组计算时间戳间隔...")
        pdf_sample = pdf_sample.sort_values(['location_id', 'timestamp'])
        # 使用 transform 在组内计算 diff，结果可以直接附加回原 DataFrame
        # Dask 中可以用 map_partitions + groupby().diff()，Pandas 更直接
        pdf_sample['time_diff'] = pdf_sample.groupby('location_id')['timestamp'].transform(lambda x: x.diff())

        logger.info("统计时间戳间隔频率...")
        freq_counts = pdf_sample['time_diff'].value_counts(dropna=False) # 包含 NaN (每个 group 的第一个)
        total_diffs = len(pdf_sample['time_diff'].dropna())

        if total_diffs > 0:
            percentage = (freq_counts / total_diffs * 100).round(2)
            dist_df = pd.DataFrame({'计数': freq_counts, '百分比 (%)': percentage})
            log_str = f"时间戳间隔频率统计 (Top 10):\n{dist_df.head(10).to_string()}"
            if len(freq_counts) > 10: log_str += "\n..."
            logger.info(log_str)
            most_common_freq = freq_counts.idxmax()
            logger.info(f"最常见的时间间隔: {most_common_freq}")
            # 检查除 NaT 外是否还有其他频率
            if len(freq_counts.drop(index=pd.NaT, errors='ignore')) > 1:
                logger.warning("在抽样数据中检测到多种时间间隔。")
        else:
            logger.warning("样本中未能计算出有效的时间间隔。")

        logger.info("Weather 时间戳频率分析完成。")

    except Exception as e:
        logger.exception(f"分析 Weather 时间戳频率时发生错误: {e}")


# --- 关系分析函数 (重构后) ---

def _merge_demand_metadata_sample(ddf_demand, pdf_metadata, metadata_cols, sample_frac, random_state):
    """辅助函数：抽样 Demand，合并 Metadata，返回计算后的 Pandas DF。"""
    logger.info("对 Demand 数据进行抽样并选择列 ('unique_id', 'y')...")
    pdf_merged = pd.DataFrame() # 默认空 DF
    required_metadata_cols = ['unique_id'] + [col for col in metadata_cols if col != 'unique_id']

    # 检查 Metadata 是否包含所需列
    if not all(col in pdf_metadata.columns for col in required_metadata_cols):
        missing_cols = [col for col in required_metadata_cols if col not in pdf_metadata.columns]
        logger.error(f"Metadata 缺少必要列: {missing_cols}，无法进行合并。")
        return pdf_merged

    pdf_metadata_subset = pdf_metadata[required_metadata_cols].copy()

    with _dask_compute_context(ddf_demand[['unique_id', 'y']]) as persisted_demand:
         ddf_demand_subset = persisted_demand[0]
         ddf_demand_sample = ddf_demand_subset.dropna(subset=['y']).sample(frac=sample_frac, random_state=random_state)

         logger.info("合并抽样的 Demand 数据与 Metadata 数据...")
         # Dask merge Dask DF with Pandas DF
         ddf_merged = dd.merge(ddf_demand_sample, pdf_metadata_subset, on='unique_id', how='inner')

         with _dask_compute_context(ddf_merged) as persisted_merged:
              num_merged = len(persisted_merged[0]) # 计算长度
              logger.info(f"合并完成，持久化得到 {num_merged:,} 行匹配数据。")
              if num_merged == 0:
                   logger.warning("合并后数据为空。")
                   return pdf_merged # 返回空 DF

              logger.info("计算合并后的 Pandas DataFrame...")
              pdf_merged = persisted_merged[0].compute()
              logger.info("合并后的 Pandas DataFrame 计算完成。")
              return pdf_merged


def analyze_demand_vs_metadata(ddf_demand, pdf_metadata, plots_dir=None, sample_frac=0.001, random_state=42):
    """分析 Demand (y) 与 Metadata 特征 (如 building_class) 的关系。"""
    if ddf_demand is None or pdf_metadata is None:
        logger.warning("需要 Demand 和 Metadata 数据才能进行关系分析。")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存关系分析图。")
        return

    target_col = 'building_class'
    logger.info(f"--- 开始分析 Demand vs {target_col} 关系 (抽样比例: {sample_frac:.1%}) ---")

    try:
        pdf_merged = _merge_demand_metadata_sample(ddf_demand, pdf_metadata, [target_col], sample_frac, random_state)

        if pdf_merged.empty:
             logger.warning(f"未能获取合并数据，无法分析 Demand vs {target_col}。")
             return

        logger.info(f"绘制 Demand (y) vs {target_col} 箱线图...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- 原始尺度 ---
        fig_orig, ax_orig = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=pdf_merged, x=target_col, y='y', showfliers=False, ax=ax_orig, palette="viridis")
        ax_orig.set_title(f'Demand (y) 按 {target_col} 分布 (原始尺度, 无离群点)')
        ax_orig.set_xlabel(target_col)
        ax_orig.set_ylabel('电力需求 (y) in kWh')
        plt.tight_layout()
        _save_plot(fig_orig, f'demand_vs_{target_col}_boxplot_orig.png', plots_dir)

        # --- 对数尺度 ---
        epsilon = 1e-6
        pdf_merged['y_log1p'] = np.log1p(pdf_merged['y'][pdf_merged['y'] >= 0] + epsilon)

        if pdf_merged['y_log1p'].isnull().all():
             logger.warning("没有有效的对数值进行绘图，跳过对数尺度箱线图。")
        else:
            fig_log, ax_log = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=pdf_merged.dropna(subset=['y_log1p']), x=target_col, y='y_log1p', showfliers=True, ax=ax_log, palette="viridis")
            ax_log.set_title(f'Demand (y) 按 {target_col} 分布 (Log1p 尺度)')
            ax_log.set_xlabel(target_col)
            ax_log.set_ylabel('log1p(电力需求 (y) + epsilon)')
            plt.tight_layout()
            _save_plot(fig_log, f'demand_vs_{target_col}_boxplot_log1p.png', plots_dir)

        logger.info(f"Demand vs {target_col} 分析完成。")

    except Exception as e:
        logger.exception(f"分析 Demand vs {target_col} 关系时发生错误: {e}")


def analyze_demand_vs_location(ddf_demand, pdf_metadata, plots_dir=None, sample_frac=0.001, top_n=10, random_state=42):
    """分析 Demand (y) 与 Top N location 的关系。"""
    if ddf_demand is None or pdf_metadata is None:
        logger.warning("需要 Demand 和 Metadata 数据才能进行关系分析。")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存关系分析图。")
        return

    target_col = 'location'
    logger.info(f"--- 开始分析 Demand vs {target_col} 关系 (Top {top_n}, 抽样比例: {sample_frac:.1%}) ---")

    try:
        # 合并前预处理 location 列
        pdf_metadata_processed = pdf_metadata.copy()
        if target_col in pdf_metadata_processed.columns:
             pdf_metadata_processed[target_col] = pdf_metadata_processed[target_col].fillna('缺失 (Missing)')
        else:
             logger.error(f"Metadata 中缺少 '{target_col}' 列，无法继续。")
             return

        pdf_merged = _merge_demand_metadata_sample(ddf_demand, pdf_metadata_processed, [target_col], sample_frac, random_state)

        if pdf_merged.empty:
             logger.warning(f"未能获取合并数据，无法分析 Demand vs {target_col}。")
             return

        # 确定 Top N locations
        location_counts = pdf_merged[target_col].value_counts()
        top_locations = location_counts.head(top_n).index.tolist()
        logger.info(f"Top {top_n} 地点 (按样本中的数据点数): {top_locations}")

        pdf_merged_top_n = pdf_merged[pdf_merged[target_col].isin(top_locations)].copy()
        if pdf_merged_top_n.empty:
             logger.warning(f"未能找到 Top {top_n} 地点的数据，无法绘图。")
             return

        logger.info(f"绘制 Demand (y) vs Top {top_n} {target_col} 箱线图...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- 原始尺度 ---
        fig_orig, ax_orig = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=pdf_merged_top_n, x=target_col, y='y', showfliers=False, ax=ax_orig, palette="viridis", order=top_locations)
        ax_orig.set_title(f'Demand (y) 按 Top {top_n} {target_col} 分布 (原始尺度, 无离群点)')
        ax_orig.set_xlabel(target_col)
        ax_orig.set_ylabel('电力需求 (y) in kWh')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        _save_plot(fig_orig, f'demand_vs_top{top_n}_{target_col}_boxplot_orig.png', plots_dir)

        # --- 对数尺度 ---
        epsilon = 1e-6
        pdf_merged_top_n['y_log1p'] = np.log1p(pdf_merged_top_n['y'][pdf_merged_top_n['y'] >= 0] + epsilon)

        if pdf_merged_top_n['y_log1p'].isnull().all():
             logger.warning("没有有效的对数值进行绘图，跳过对数尺度箱线图。")
        else:
            fig_log, ax_log = plt.subplots(figsize=(15, 7))
            sns.boxplot(data=pdf_merged_top_n.dropna(subset=['y_log1p']), x=target_col, y='y_log1p', showfliers=True, ax=ax_log, palette="viridis", order=top_locations)
            ax_log.set_title(f'Demand (y) 按 Top {top_n} {target_col} 分布 (Log1p 尺度)')
            ax_log.set_xlabel(target_col)
            ax_log.set_ylabel('log1p(电力需求 (y) + epsilon)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            _save_plot(fig_log, f'demand_vs_top{top_n}_{target_col}_boxplot_log1p.png', plots_dir)

        logger.info(f"Demand vs {target_col} 分析完成。")

    except Exception as e:
        logger.exception(f"分析 Demand vs {target_col} 关系时发生错误: {e}")


def analyze_demand_vs_weather(ddf_demand, pdf_metadata, ddf_weather, plots_dir=None, n_sample_ids=50, plot_sample_frac=0.1, random_state=42):
    """分析 Demand (y) 与 Weather 特征的关系 (抽样 unique_id)。"""
    if ddf_demand is None or pdf_metadata is None or ddf_weather is None:
        logger.warning("需要 Demand, Metadata 和 Weather 数据才能进行 Demand vs Weather 分析。")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存 Demand vs Weather 分析图。")
        return

    logger.info(f"--- 开始分析 Demand vs Weather 关系 (抽样 {n_sample_ids} 个 unique_id) ---")
    pdf_merged = pd.DataFrame() # 初始化

    try:
        # 1. 抽样 unique_id
        logger.info(f"从 Metadata 中随机抽取 {n_sample_ids} 个 unique_id...")
        valid_ids = pdf_metadata['unique_id'].unique()
        if len(valid_ids) < n_sample_ids:
            logger.warning(f"Metadata 中 unique_id 数量 ({len(valid_ids)}) 少于请求的样本数量 ({n_sample_ids})，将使用所有可用 ID。")
            sampled_unique_ids = valid_ids
            n_sample_ids = len(sampled_unique_ids)
        else:
            np.random.seed(random_state)
            sampled_unique_ids = np.random.choice(valid_ids, n_sample_ids, replace=False)
        logger.info(f"选取的 unique_id (前 10): {sampled_unique_ids[:10].tolist()}...")

        # 2. 准备 Demand 数据 (筛选 + 获取 location_id)
        logger.info("筛选 Demand 数据并合并 Metadata 获取 location_id...")
        pdf_metadata_sample = pdf_metadata[pdf_metadata['unique_id'].isin(sampled_unique_ids)][['unique_id', 'location_id']].drop_duplicates().dropna(subset=['location_id'])
        if pdf_metadata_sample.empty:
            logger.warning("抽样的 unique_id 在 Metadata 中没有找到有效的 location_id，无法继续。")
            return

        ddf_demand_filtered = ddf_demand[ddf_demand['unique_id'].isin(sampled_unique_ids)][['unique_id', 'timestamp', 'y']].dropna(subset=['y'])
        # Dask merge Dask DF with Pandas DF
        ddf_demand_with_loc = dd.merge(ddf_demand_filtered, pdf_metadata_sample, on='unique_id', how='inner')

        # 3. 准备 Weather 数据
        weather_cols = ['location_id', 'timestamp', 'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'precipitation', 'wind_speed_10m', 'cloud_cover']
        ddf_weather_subset = ddf_weather[[col for col in weather_cols if col in ddf_weather.columns]] # 确保列存在

        # 4. 合并 (转换为 Pandas 进行 merge_asof，因为 Dask 的 merge_asof 仍然不稳定)
        logger.info("将抽样 Demand (带 location_id) 数据计算为 Pandas DataFrame...")
        with _dask_compute_context(ddf_demand_with_loc) as p_demand:
            pdf_demand_with_loc = p_demand[0].compute()
        num_demand_rows = len(pdf_demand_with_loc)
        logger.info(f"Demand Pandas DataFrame 计算完成，包含 {num_demand_rows:,} 行。")
        if num_demand_rows == 0:
             logger.warning("未能将 location_id 合并到 Demand 数据，无法继续。")
             return

        relevant_location_ids = pdf_demand_with_loc['location_id'].unique()
        logger.info(f"筛选 Weather 数据，只保留 {len(relevant_location_ids)} 个相关的 location_id...")
        ddf_weather_filtered = ddf_weather_subset[ddf_weather_subset['location_id'].isin(relevant_location_ids)]

        logger.info("计算筛选后的 Weather 数据为 Pandas DataFrame...")
        with _dask_compute_context(ddf_weather_filtered) as p_weather:
             pdf_weather_filtered = p_weather[0].compute()
        logger.info(f"Weather Pandas DataFrame 计算完成，包含 {len(pdf_weather_filtered):,} 行。")


        logger.info("准备合并数据 (排序, 去重, 类型转换)...")
        # 删除 NaN, 排序, 去重
        for df, name in [(pdf_demand_with_loc, "Demand"), (pdf_weather_filtered, "Weather")]:
             rows_before = len(df)
             df.dropna(subset=['location_id', 'timestamp'], inplace=True)
             rows_after_na = len(df)
             df.sort_values(by=['location_id', 'timestamp'], inplace=True)
             df.drop_duplicates(subset=['location_id', 'timestamp'], keep='first', inplace=True)
             rows_after_dedup = len(df)
             df.reset_index(drop=True, inplace=True)
             if rows_before > rows_after_na: logger.warning(f"从 {name} 数据中删除 {rows_before - rows_after_na} 行 (因 location_id 或 timestamp 为 NaN)")
             if rows_after_na > rows_after_dedup: logger.warning(f"从 {name} 数据中删除 {rows_after_na - rows_after_dedup} 行重复的 (location_id, timestamp)")
             # 尝试转换 location_id 类型
             try:
                 df['location_id'] = df['location_id'].astype(str)
             except Exception as e:
                 logger.error(f"转换 {name} 的 location_id 类型失败: {e}")
                 return # 严重错误，无法继续

        # 再次检查排序 （merge_asof 严格要求）
        is_left_sorted = pdf_demand_with_loc.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing).all()
        is_right_sorted = pdf_weather_filtered.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing).all()

        if not is_left_sorted or not is_right_sorted:
             logger.error(f"数据排序检查失败! 左侧排序: {is_left_sorted}, 右侧排序: {is_right_sorted}. merge_asof 无法安全执行。")
             # 这里可以添加更详细的诊断，找出哪个组有问题
             if not is_left_sorted:
                 bad_groups = pdf_demand_with_loc.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing)
                 logger.error(f"左侧排序失败的 location_id: {bad_groups[~bad_groups].index.tolist()}")
             if not is_right_sorted:
                 bad_groups = pdf_weather_filtered.groupby('location_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing)
                 logger.error(f"右侧排序失败的 location_id: {bad_groups[~bad_groups].index.tolist()}")
             # === 重要：在此处标记错误并跳过合并 ===
             logger.error("由于排序问题，跳过 merge_asof 步骤。")

             return # 直接返回，不执行后续步骤

        else:
            logger.info("数据排序检查通过。")
            logger.info("使用 Pandas merge_asof 合并数据...")
            try:
                pdf_merged = pd.merge_asof(
                    pdf_demand_with_loc,
                    pdf_weather_filtered,
                    on='timestamp',
                    by='location_id',
                    direction='backward',
                    tolerance=pd.Timedelta('1hour')
                )
                logger.info(f"最终合并完成，得到 {len(pdf_merged):,} 行数据。")
            except ValueError as ve:
                 # 即使检查通过，有时仍可能因边缘情况或类型问题失败
                 logger.error(f"Pandas merge_asof 意外失败 (即使排序检查通过): {ve}")
                 logger.exception("merge_asof 执行期间出错")
                 return # 失败则返回
            except Exception as merge_exc:
                 logger.exception("合并数据时发生未知错误")
                 return # 失败则返回

        if pdf_merged.empty:
             logger.warning("最终合并后数据为空，无法进行关系分析。")
             return

        # 5. 计算相关性
        logger.info("计算 Demand (y) 与天气特征的相关性...")
        correlation_cols = ['y'] + [col for col in weather_cols if col in pdf_merged.columns and col not in ['location_id', 'timestamp']]
        # 确保所有相关列都是数值类型
        numeric_correlation_cols = [col for col in correlation_cols if pd.api.types.is_numeric_dtype(pdf_merged[col])]
        if len(numeric_correlation_cols) < len(correlation_cols):
             skipped_cols = set(correlation_cols) - set(numeric_correlation_cols)
             logger.warning(f"以下列不是数值类型，将从相关性计算中排除: {skipped_cols}")

        if 'y' not in numeric_correlation_cols or len(numeric_correlation_cols) <= 1:
             logger.warning("没有足够的数值列（包括 'y'）来计算相关性。")
        else:
            correlation_matrix = pdf_merged[numeric_correlation_cols].corr()
            logger.info(f"相关性矩阵 ('y' vs Weather):\n{correlation_matrix['y'].to_string()}")

            # 6. 可视化
            # --- 相关性热力图 ---
            logger.info("绘制相关性热力图...")
            plt.style.use('seaborn-v0_8-whitegrid')
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
            ax_corr.set_title(f'相关性矩阵: Demand (y) vs 天气特征 (抽样 {n_sample_ids} IDs)')
            plt.tight_layout()
            _save_plot(fig_corr, 'demand_vs_weather_correlation_heatmap.png', plots_dir)

            # --- 散点图 ---
            scatter_cols = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature']
            scatter_cols = [col for col in scatter_cols if col in numeric_correlation_cols] # 只绘制存在的数值列
            logger.info(f"绘制 Demand (y) vs {', '.join(scatter_cols)} 的散点图 (绘图抽样比例: {plot_sample_frac:.1%})")

            plot_sample = pdf_merged.sample(frac=plot_sample_frac, random_state=random_state) if len(pdf_merged) * plot_sample_frac >= 1 else pdf_merged

            for col in scatter_cols:
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=plot_sample, x=col, y='y', alpha=0.5, s=10, ax=ax_scatter)
                ax_scatter.set_title(f'Demand (y) vs {col} (抽样数据)')
                ax_scatter.set_xlabel(col)
                ax_scatter.set_ylabel('电力需求 (y) in kWh')
                corr_val = correlation_matrix.loc['y', col]
                ax_scatter.text(0.05, 0.95, f'相关系数: {corr_val:.3f}', transform=ax_scatter.transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                plt.tight_layout()
                _save_plot(fig_scatter, f'demand_vs_weather_{col}_scatterplot.png', plots_dir)

        logger.info("Demand vs Weather 分析完成。")

    except Exception as e:
        logger.exception(f"分析 Demand vs Weather 关系时发生错误: {e}")
    # finally 中不再需要显式清理


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
    # y_sample_pd = analyze_demand_y_distribution(ddf_demand, sample_frac=0.005)
    # if y_sample_pd is not None:
    #      plot_demand_y_distribution(y_sample_pd, plots_dir, plot_sample_size=100000)
    # analyze_demand_timeseries_sample(ddf_demand, n_samples=3, plots_dir=plots_dir) # 减少样本量加快速度
    logger.info("--- 完成 Demand 'y' 分析 (或已注释) ---")


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
    logger.info("--- 开始 Demand vs Weather 分析 ---")
    try:
        analyze_demand_vs_weather(ddf_demand, pdf_metadata, ddf_weather, plots_dir=plots_dir, n_sample_ids=50)
    except Exception as e:
         # analyze_demand_vs_weather 内部已记录详细错误，这里只记录简要信息
         logger.error(f"Demand vs Weather 分析执行期间遇到问题: {e}")
    logger.info("--- 完成 Demand vs Weather 分析 (或已跳过) ---")


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