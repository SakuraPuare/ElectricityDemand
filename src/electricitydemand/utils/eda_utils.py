import os
import contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from dask import persist
from dask.distributed import Client, futures_of
from distributed.client import Future # 检查 future 类型

# --- 辅助函数 ---

def save_plot(fig, filename, plots_dir):
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

def plot_numerical_distribution(data_pd_series, col_name, filename_base, plots_dir, title_prefix="", kde=True, showfliers=True):
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
        save_plot(fig, f"{filename_base}.png", plots_dir) # Use the renamed function
    except Exception as e:
        logger.exception(f"Error plotting distribution for column '{col_name}': {e}")
        plt.close(fig) # Ensure figure is closed

def plot_categorical_distribution(data_pd_series, col_name, filename_base, plots_dir, top_n=10, title_prefix=""):
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
                    # Use pd.concat instead of direct assignment for pandas > 1.x
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
            # Ensure we use the original index for sorting, then convert to string
            sort_order_index = data_to_plot.sort_index().index
            sort_order = sort_order_index.astype(str)
            # Filter plot_index to only include those in sort_order (handles 'Others')
            plot_index_ordered = [idx for idx in sort_order if idx in plot_index]
            # Ensure 'Others' is last if it exists
            if 'Others' in plot_index_ordered:
                plot_index_ordered.remove('Others')
                plot_index_ordered.append('Others')

        except TypeError: # Cannot sort mixed types or complex index
            logger.debug(f"Cannot sort index for '{col_name}' easily, using count order.")
            sort_order = data_to_plot.index.astype(str) # Order by appearance (count order)
            plot_index_ordered = plot_index.tolist() # Use original order as list

        sns.barplot(x=plot_index, y=data_to_plot.values, ax=ax, palette="viridis", order=plot_index_ordered)
        ax.set_title(title)
        ax.set_xlabel(col_name)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_plot(fig, f"{filename_base}.png", plots_dir) # Use the renamed function

    except Exception as e:
        logger.exception(f"Error plotting distribution for column '{col_name}': {e}")
        plt.close(fig)

def log_value_counts(data_pd_series, col_name, top_n=10, is_already_counts=False):
    """记录 Pandas Series 的值计数和百分比。

    Args:
        data_pd_series: Pandas Series containing the data or pre-computed value counts.
        col_name: Name of the original column for logging.
        top_n: How many top values to display in the log.
        is_already_counts: Set to True if data_pd_series is already the result of value_counts().
    """
    logger.info(f"--- 分析列: {col_name} 值分布 ---")

    # Input validation
    if not isinstance(data_pd_series, pd.Series):
        logger.warning(f"输入给 log_value_counts 的 '{col_name}' 不是 Pandas Series，跳过。类型: {type(data_pd_series)}")
        return
    if data_pd_series.empty:
        logger.warning(f"列 '{col_name}' 的数据为空，跳过值计数记录。")
        return

    if is_already_counts:
        counts = data_pd_series
        total_rows = int(counts.sum()) # Total is the sum of counts
        logger.debug(f"处理预先计算的计数值，总计: {total_rows}")
    else:
        total_rows = len(data_pd_series)
        logger.debug(f"计算原始数据的计数值，总行数: {total_rows}")
        counts = data_pd_series.value_counts(dropna=False) # 包含 NaN

    if total_rows > 0:
        percentage = (counts / total_rows * 100).round(2)
        dist_df = pd.DataFrame({'计数': counts.astype(int), '百分比 (%)': percentage}) # Cast counts to int for clarity
    else:
        dist_df = pd.DataFrame({'计数': counts.astype(int), '百分比 (%)': 0.0})
        logger.warning(f"列 '{col_name}' 总行数为 0，百分比将为 0。")


    log_str = f"值分布 (含 NaN):\n{dist_df.head(top_n).to_string()}"
    if len(dist_df) > top_n:
        log_str += "\n..."
    logger.info(log_str)

    # Unique count calculation needs care if input is counts
    if is_already_counts:
        # If counts include NaN, it will be in the index
        num_unique = len(counts)
        has_nan = counts.index.isnull().any()
    else:
        num_unique = data_pd_series.nunique(dropna=False)
        has_nan = data_pd_series.isnull().any()

    logger.info(f"列 '{col_name}' 唯一值数量 (含 NaN): {num_unique}")
    if has_nan:
        # Find the NaN count properly depending on input type
        if is_already_counts:
            nan_count = counts[counts.index.isnull()].sum()
        else:
            nan_count = data_pd_series.isnull().sum()
        nan_perc = (nan_count / total_rows * 100).round(2) if total_rows > 0 else 0
        logger.warning(f"列 '{col_name}' 存在缺失值 (NaN)。数量: {nan_count}, 百分比: {nan_perc:.2f}%")


@contextlib.contextmanager
def dask_compute_context(*dask_objects):
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
                # 检查是否有活动的客户端
                with contextlib.suppress(RuntimeError, ImportError): # 忽略 'no client found' 和 Import Error
                     client = Client.current() # 可能引发 RuntimeError
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
                         logger.debug("取消 Futures 尝试完成。")
                     else:
                         logger.debug("未找到 Dask Futures 进行取消 (可能已完成或无 Client)。")

            except ImportError:
                # This case is now handled by the suppress above, but kept for clarity
                logger.debug("未安装 dask.distributed，跳过显式内存清理。")
            except Exception as e:
                # Log other unexpected exceptions during cleanup
                logger.warning(f"清理 Dask 内存时出现其他异常: {e}", exc_info=False)
            logger.debug("Dask 上下文退出。") 