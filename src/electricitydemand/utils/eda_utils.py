import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

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
        logger.info(f"图表已保存到：{filepath}")
        plt.close(fig) # 保存后关闭图形
    except Exception as e:
        logger.exception(f"保存图表 '{filename}' 时出错：{e}")
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
        # 如果输入已经是计算好的 counts (Series, index=value, value=count)
        if isinstance(data_pd_series, pd.Series) and pd.api.types.is_numeric_dtype(data_pd_series):
            value_counts = data_pd_series
        # 如果输入是原始 Series
        elif isinstance(data_pd_series, pd.Series):
            value_counts = data_pd_series.value_counts(dropna=False) # Include NaN
        else:
            logger.error(f"Unsupported data type for categorical plot: {type(data_pd_series)}")
            return

        num_unique = len(value_counts)

        if num_unique > top_n:
            logger.info(f"Column '{col_name}' has {num_unique} unique values, showing Top {top_n} and 'Others'.")
            top_values = value_counts.head(top_n).copy() # Use copy to avoid modifying original
            other_count = value_counts.iloc[top_n:].sum()
            if other_count > 0:
                others_label = 'Others'
                others_series = pd.Series([other_count], index=[others_label])
                if others_label in top_values.index:
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
            # Use the index of the data *before* potential 'Others' addition for sorting basis
            sort_basis_index = value_counts.index
            sort_order_index = sort_basis_index.sort_values()
            sort_order = sort_order_index.astype(str)
            # Filter plot_index to only include those in sort_order
            plot_index_ordered = [idx for idx in sort_order if idx in plot_index]
            # Ensure 'Others' is last if it exists and was added
            if 'Others' in plot_index and 'Others' not in sort_order:
                if 'Others' in plot_index_ordered: # Should not happen if logic is right, but safety check
                     plot_index_ordered.remove('Others')
                plot_index_ordered.append('Others')
            elif 'Others' in plot_index and 'Others' in sort_order: # If 'Others' was an original category
                 pass # Keep its sorted position

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

def log_value_counts(data, column_name, top_n=10, is_already_counts=False, normalize=True):
    """记录 Series 或 Dask Series 的值分布。"""
    logger.info(f"--- 分析列：{column_name} 值分布 ---")

    dist_df = None # Initialize dist_df
    total_count = 0 # Initialize total_count

    # --- Existing logic to calculate dist_df and total_count ---
    if is_already_counts:
        # Expecting a Pandas DataFrame with value and count columns
        if not isinstance(data, pd.DataFrame):
             logger.error(f"当 is_already_counts=True 时，期望输入为 Pandas DataFrame，但收到 {type(data)}")
             return
        # Assume the first column is the value and the second is the count
        if len(data.columns) < 2:
            logger.error("当 is_already_counts=True 时，输入的 DataFrame 至少需要两列 (值和计数)。")
            return
        dist_df = data.copy()
        value_col = dist_df.columns[0]
        count_col = dist_df.columns[1]
        # Rename for consistency
        dist_df.columns = ['value', '计数'] # Rename to standard names
        count_col = '计数' # Update count_col name

        total_count = dist_df[count_col].sum()
        if normalize and total_count > 0:
            dist_df['百分比 (%)'] = (dist_df[count_col] / total_count * 100).round(2)
        else:
            dist_df['百分比 (%)'] = 0.0

        # Set index to the value column for consistent logging format
        dist_df = dist_df.set_index('value')


    elif isinstance(data, (pd.Series, pd.Index)):
         try:
             # Pandas Series or Index
             value_counts = data.value_counts(dropna=False) # Include NaNs
             total_count = len(data) # Total count including NaNs
             dist_df = pd.DataFrame({'计数': value_counts})
             if normalize and total_count > 0:
                 dist_df['百分比 (%)'] = (dist_df['计数'] / total_count * 100).round(2)
             else:
                 dist_df['百分比 (%)'] = 0.0
         except Exception as e:
             logger.error(f"计算 Pandas Series/Index '{column_name}' 值计数时出错：{e}")
             return
    # Removed Dask Series case
    # elif isinstance(data, dd.Series): ...
    else:
        logger.error(f"不支持的数据类型进行值计数：{type(data)} for column '{column_name}'")
        return

    # --- End of logic to calculate dist_df ---

    # Check if dist_df was successfully created
    if dist_df is None or dist_df.empty:
         logger.warning(f"未能计算列 '{column_name}' 的值分布，或结果为空。")
         # Check for NaN values in the original data if applicable and not already done
         if isinstance(data, (pd.Series, pd.Index)) and not is_already_counts:
              nan_count = data.isnull().sum()
              if total_count > 0 and nan_count > 0:
                   logger.warning(f"列 '{column_name}' 包含 {nan_count} 个 NaN 值 ({nan_count/total_count*100:.2f}%)。")
              elif total_count > 0:
                   logger.info(f"列 '{column_name}' 不包含 NaN 值。")
              else:
                   logger.warning(f"无法计算 Pandas Series/Index '{column_name}' 的大小。")
         # Removed Dask NaN check
         return # Exit if dist_df is None or empty

    # --- Sort by count descending ---
    count_col_name = '计数' # Standardized name
    try:
        dist_df = dist_df.sort_values(by=count_col_name, ascending=False)
    except Exception as e:
        logger.error(f"按 '{count_col_name}' 列排序时出错：{e}\nDataFrame:\n{dist_df.head()}")
        return # Stop if sorting fails

    # --- Handle top_n logic ---
    original_len = len(dist_df)
    # *** 修改在这里：只有当 top_n 不是 None 时才进行比较和截断 ***
    if top_n is not None and original_len > top_n:
        logger.info(f"列 '{column_name}' 唯一值过多 ({original_len})，仅显示 Top {top_n}。")
        display_df = dist_df.head(top_n)
        log_message = f"值分布 (Top {top_n}, 含 NaN):\n{display_df.to_string()}"
    else:
        # If top_n is None or len <= top_n, display all
        display_df = dist_df
        log_message = f"值分布 (含 NaN):\n{display_df.to_string()}"

    logger.info(log_message)


    # --- 检查 NaN 值 ---
    # Check for NaN in the index of the counts DataFrame
    nan_present_in_index = False
    try:
        # Check if any index element is NaN using pd.isna()
        nan_present_in_index = any(pd.isna(idx) for idx in display_df.index)
    except TypeError: # Handle cases like multi-index where isna might fail directly
         logger.warning(f"无法直接检查列 '{column_name}' 值计数索引中的 NaN。")

    if nan_present_in_index:
         try:
             # Select rows where the index is NaN
             nan_rows = display_df[[pd.isna(idx) for idx in display_df.index]]
             if not nan_rows.empty:
                 # Assuming there's only one NaN category row
                 nan_count = nan_rows['计数'].iloc[0]
                 nan_perc = nan_rows['百分比 (%)'].iloc[0]
                 logger.warning(f"列 '{column_name}' 存在缺失值 (NaN)。数量：{nan_count}, 百分比：{nan_perc:.2f}%")
             else:
                 # This case should not happen if nan_present_in_index is True, but adding for safety
                 logger.info(f"列 '{column_name}' 值计数中未明确找到 NaN 行，尽管索引检查提示存在。")
         except KeyError:
            logger.warning(f"无法在列 '{column_name}' 的值计数中定位 NaN 行进行统计 (KeyError)。")
         except Exception as e:
             logger.warning(f"检查列 '{column_name}' 的 NaN 统计时出错：{e}")
    else:
         # If no NaN in index, still good to confirm no NaNs in original if counts were calculated
         if not is_already_counts and isinstance(data, (pd.Series, pd.Index)):
             if data.isnull().sum() == 0:
                 logger.info(f"列 '{column_name}' 值计数中未发现 NaN。")
             # else: This case indicates NaNs were present but lost somehow, covered by earlier checks?


    # Log unique value count
    unique_count = original_len # Number of unique values including NaN if present
    logger.info(f"列 '{column_name}' 唯一值数量 (含 NaN): {unique_count}")

    logger.info("-" * 20) # Separator

# 移除 dask_compute_context
# @contextlib.contextmanager
# def dask_compute_context(*dask_objects):
#    """上下文管理器，用于触发 Dask persist/compute 并尝试清理。"""
#    persisted_objects = []
#    try:
#        if dask_objects:
#            logger.debug(f"尝试持久化 {len(dask_objects)} 个 Dask 对象...")
#            persisted_objects = persist(*dask_objects)
#            logger.debug("Dask 对象持久化完成。")
#        yield persisted_objects
#    finally:
#        if persisted_objects:
#            logger.debug("尝试释放 Dask 持久化对象...")
#            try:
#                with contextlib.suppress(RuntimeError, ImportError):
#                     client = Client.current()
#                     futures = []
#                     for obj in persisted_objects:
#                         if hasattr(obj, 'dask'):
#                            futures.extend(futures_of(obj))
#                         elif isinstance(obj, Future):
#                            futures.append(obj)
#
#                     if futures:
#                         logger.debug(f"找到 {len(futures)} 个 futures，尝试取消...")
#                         client.cancel(futures, force=True)
#                         logger.debug("取消 Futures 尝试完成。")
#                     else:
#                         logger.debug("未找到 Dask Futures 进行取消 (可能已完成或无 Client)。")
#
#            except ImportError:
#                logger.debug("未安装 dask.distributed，跳过显式内存清理。")
#            except Exception as e:
#                logger.warning(f"清理 Dask 内存时出现其他异常：{e}", exc_info=False)
#            logger.debug("Dask 上下文退出。") 