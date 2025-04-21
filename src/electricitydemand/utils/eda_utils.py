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

def log_value_counts(data, column_name, top_n=10, is_already_counts=False, normalize=True):
    """记录 Series 或 Dask Series 的值分布。"""
    logger.info(f"--- 分析列：{column_name} 值分布 ---")

    dist_df = None # Initialize dist_df
    total_count = 0 # Initialize total_count

    # --- Existing logic to calculate dist_df and total_count ---
    if is_already_counts:
        # ... (代码未改变) ...
        if not isinstance(data, (pd.DataFrame, pd.Series)):
             logger.error(f"当 is_already_counts=True 时，期望输入为 Pandas DataFrame/Series，但收到 {type(data)}")
             return
        dist_df = data.copy()
        # Try to infer count and percentage columns or require specific names
        count_col = '计数' if '计数' in dist_df.columns else (dist_df.columns[0] if len(dist_df.columns) > 0 else None)
        perc_col = '百分比 (%)' if '百分比 (%)' in dist_df.columns else None

        if count_col is None:
             logger.error("无法在已计数的 DataFrame 中找到计数列。")
             return

        total_count = dist_df[count_col].sum()
        if perc_col is None and normalize and total_count > 0:
            dist_df['百分比 (%)'] = (dist_df[count_col] / total_count * 100).round(2)
            perc_col = '百分比 (%)'
        elif perc_col is None:
            dist_df['百分比 (%)'] = 0.0 # Default percentage if not calculable
            perc_col = '百分比 (%)'
        # Ensure columns exist for sorting
        if count_col not in dist_df.columns or perc_col not in dist_df.columns:
             logger.error(f"计数或百分比列 ('{count_col}', '{perc_col}') 在 DataFrame 中缺失。")
             return

    elif isinstance(data, pd.Series):
        # ... (代码未改变) ...
        try:
            # Pandas Series
            value_counts = data.value_counts(dropna=False)
            total_count = len(data) # More direct for Pandas
            dist_df = pd.DataFrame({'计数': value_counts})
            if normalize and total_count > 0:
                 dist_df['百分比 (%)'] = (dist_df['计数'] / total_count * 100).round(2)
            else:
                 dist_df['百分比 (%)'] = 0.0
        except Exception as e:
            logger.error(f"计算 Pandas Series '{column_name}' 值计数时出错：{e}")
            return

    elif isinstance(data, (pd.Series, pd.Index)):
         # ... (代码未改变) ...
         try:
             # Pandas Series or Index
             value_counts = data.value_counts(dropna=False)
             total_count = len(data) # More direct for Pandas
             dist_df = pd.DataFrame({'计数': value_counts})
             if normalize and total_count > 0:
                 dist_df['百分比 (%)'] = (dist_df['计数'] / total_count * 100).round(2)
             else:
                 dist_df['百分比 (%)'] = 0.0
         except Exception as e:
             logger.error(f"计算 Pandas Series/Index '{column_name}' 值计数时出错：{e}")
             return
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
              if nan_count > 0:
                   logger.warning(f"列 '{column_name}' 包含 {nan_count} 个 NaN 值 ({nan_count/len(data)*100:.2f}%)。")
              else:
                   logger.info(f"列 '{column_name}' 不包含 NaN 值。")
         elif isinstance(data, pd.Series) and not is_already_counts:
              try:
                   nan_count = data.isnull().sum()
                   total_s = len(data)
                   if total_s > 0 and nan_count > 0:
                        logger.warning(f"列 '{column_name}' 包含 {nan_count} 个 NaN 值 ({nan_count/total_s*100:.2f}%)。")
                   elif total_s > 0:
                        logger.info(f"列 '{column_name}' 不包含 NaN 值。")
                   else:
                        logger.warning(f"无法计算 Pandas Series '{column_name}' 的大小。")

              except Exception as e:
                   logger.warning(f"检查 Pandas Series '{column_name}' 的 NaN 时出错：{e}")

         return # Exit if dist_df is None or empty

    # --- Sort by count descending ---
    # Ensure '计数' column exists before sorting
    count_col_name = '计数' if '计数' in dist_df.columns else (dist_df.columns[0] if len(dist_df.columns) > 0 else None)
    if count_col_name:
        try:
            dist_df = dist_df.sort_values(by=count_col_name, ascending=False)
        except Exception as e:
            logger.error(f"按 '{count_col_name}' 列排序时出错：{e}\nDataFrame:\n{dist_df.head()}")
            return # Stop if sorting fails
    else:
        logger.error("在 dist_df 中找不到用于排序的计数列。")
        return


    # --- Handle top_n logic ---
    original_len = len(dist_df)
    # *** 修改在这里：只有当 top_n 不是 None 时才进行比较和截断 ***
    if top_n is not None and original_len > top_n:
        logger.info(f"列 '{column_name}' 唯一值过多 ({original_len})，仅显示 Top {top_n}。")
        display_df = dist_df.head(top_n)
        # Note: Adding 'Others' row removed for simplicity, can be added back if needed carefully
        log_message = f"值分布 (Top {top_n}, 含 NaN):\n{display_df.to_string()}"
    else:
        # If top_n is None or len <= top_n, display all
        display_df = dist_df
        log_message = f"值分布 (含 NaN):\n{display_df.to_string()}"

    logger.info(log_message)


    # --- 检查 NaN 值 ---
    # Check for NaN in the index of the counts DataFrame if normalization was done
    # If dropna=False was used, NaN should appear as an index level
    nan_present_in_index = pd.NA in display_df.index or None in display_df.index or np.nan in display_df.index
    # More robust check across index types
    try:
        nan_present_in_index = any(pd.isna(idx) for idx in display_df.index)
    except TypeError: # Handle cases like multi-index where isna might fail directly
         logger.warning(f"无法直接检查列 '{column_name}' 值计数索引中的 NaN。")
         nan_present_in_index = False # Assume no NaNs if check fails


    if nan_present_in_index:
         try:
             nan_row = display_df.loc[[idx for idx in display_df.index if pd.isna(idx)]]
             if not nan_row.empty:
                 nan_count = nan_row['计数'].iloc[0] # Get count from the first NaN row found
                 nan_perc = nan_row['百分比 (%)'].iloc[0] # Get percentage
                 logger.warning(f"列 '{column_name}' 存在缺失值 (NaN)。数量：{nan_count}, 百分比：{nan_perc:.2f}%")
             else:
                # This case should not happen if nan_present_in_index is True, but adding for safety
                logger.info(f"列 '{column_name}' 值计数中未明确找到 NaN 行，尽管索引检查提示存在。")
         except KeyError:
            logger.warning(f"无法在列 '{column_name}' 的值计数中定位 NaN 行进行统计。")
         except Exception as e:
             logger.warning(f"检查列 '{column_name}' 的 NaN 统计时出错：{e}")

    else:
        # Check original data if NaN wasn't in counts (e.g., if dropna=True was somehow used or data had no NaNs)
        # This check might be redundant if already performed when dist_df was empty, but can be a fallback.
        if not is_already_counts: # Only check original if we calculated counts
            nan_in_original = False
            original_nan_count = 0
            original_total = 0
            if isinstance(data, (pd.Series, pd.Index)):
                 original_nan_count = data.isnull().sum()
                 original_total = len(data)
                 nan_in_original = original_nan_count > 0
            elif isinstance(data, pd.Series):
                try:
                    original_nan_count = data.isnull().sum()
                    original_total = len(data)
                    nan_in_original = original_nan_count > 0
                except Exception as e:
                    logger.warning(f"检查原始 Pandas Series '{column_name}' 的 NaN 时出错：{e}")

            if nan_in_original and original_total > 0:
                 logger.warning(f"列 '{column_name}' 原始数据中发现 NaN，但在最终值计数中未显示。"
                               f" 原始 NaN 数量：{original_nan_count}, 百分比：{original_nan_count/original_total*100:.2f}%")
            elif not nan_in_original:
                 logger.info(f"列 '{column_name}' 值计数中未发现 NaN。")


    # Log unique value count
    unique_count = original_len # Number of unique values including NaN if present
    logger.info(f"列 '{column_name}' 唯一值数量 (含 NaN): {unique_count}")

    logger.info("-" * 20) # Separator

@contextlib.contextmanager
def dask_compute_context(*dask_objects):
    """上下文管理器，用于触发 Dask persist/compute 并尝试清理。"""
    persisted_objects = []
    try:
        if dask_objects:
            logger.debug(f"尝试持久化 {len(dask_objects)} 个 Dask 对象...")
            # 注意：persist 返回新的对象列表
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
                logger.warning(f"清理 Dask 内存时出现其他异常：{e}", exc_info=False)
            logger.debug("Dask 上下文退出。") 