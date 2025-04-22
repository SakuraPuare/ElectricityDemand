import os
from pathlib import Path  # Import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from pyspark.rdd import RDD  # Add RDD
from pyspark.sql import DataFrame, SparkSession  # Add SparkSession, DataFrame
from pyspark.sql import functions as F  # Add Spark functions

# --- 辅助函数 ---


def save_plot(fig, filename: str, plots_dir):
    """保存 matplotlib 图表到指定目录并记录日志。"""
    if not plots_dir:
        logger.error("未提供图表保存目录 (plots_dir)，无法保存图表。")
        plt.close(fig)  # 关闭图形以释放内存
        return
    # Ensure plots_dir is a Path object
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    filepath = plots_dir / filename  # Use Path object for joining
    try:
        fig.savefig(filepath)
        # Changed log message to English
        logger.info(f"Plot saved to: {filepath}")
        plt.close(fig)  # 保存后关闭图形
    except Exception as e:
        logger.exception(f"Error saving plot '{filename}': {e}")
        plt.close(fig)  # 出错也要关闭


def plot_numerical_distribution(
    data_pd_series: pd.Series,
    col_name: str,
    filename_base: str,
    plots_dir: Path,  # Expect Path object
    title: str = None,  # Add title parameter
    xlabel: str = None,  # Add xlabel parameter
    ylabel_hist: str = 'Frequency',  # Default English ylabel for hist
    kde=True,
    showfliers=True
):
    """绘制数值型 Pandas Series 的分布图 (直方图和箱线图)，使用英文标签。"""
    if data_pd_series is None or data_pd_series.empty:
        logger.warning(f"Column '{col_name}' data is empty, skipping plot.")
        return

    # Ensure plots_dir is Path
    plots_dir = Path(plots_dir)

    logger.info(
        f"Plotting distribution for column '{col_name}' (Sample size: {len(data_pd_series):,})...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Set default title and xlabel if not provided
    if title is None:
        title = f'{col_name} Distribution'
    if xlabel is None:
        xlabel = col_name

    try:
        # Histogram
        sns.histplot(data_pd_series, kde=kde, ax=axes[0])
        axes[0].set_title(f'{title} (Histogram)')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel_hist)  # Use English ylabel

        # Box Plot
        sns.boxplot(x=data_pd_series, ax=axes[1], showfliers=showfliers)
        axes[1].set_title(f'{title} (Box Plot)')
        axes[1].set_xlabel(xlabel)

        plt.tight_layout()
        save_plot(fig, f"{filename_base}.png", plots_dir)  # Pass Path object
    except Exception as e:
        logger.exception(
            f"Error plotting distribution for column '{col_name}': {e}")
        plt.close(fig)  # Ensure figure is closed


def plot_categorical_distribution(
    data_pd_series: pd.Series,
    col_name: str,
    filename_base: str,
    plots_dir: Path,  # Expect Path object
    top_n=10,
    title: str = None,  # Add title parameter
    xlabel: str = None,  # Add xlabel parameter
    ylabel: str = 'Count'  # Default English ylabel
):
    """绘制分类 Pandas Series 的分布图 (条形图)，使用英文标签。"""
    if data_pd_series is None or data_pd_series.empty:
        logger.warning(f"Column '{col_name}' data is empty, skipping plot.")
        return

    # Ensure plots_dir is Path
    plots_dir = Path(plots_dir)

    logger.info(
        f"Plotting categorical distribution for column '{col_name}' (Top {top_n})...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set default title and xlabel if not provided
    if xlabel is None:
        xlabel = col_name

    try:
        # --- Value counting logic (remains the same) ---
        if isinstance(data_pd_series, pd.Series) and pd.api.types.is_numeric_dtype(data_pd_series):
            value_counts = data_pd_series
        elif isinstance(data_pd_series, pd.Series):
            value_counts = data_pd_series.value_counts(dropna=False)
        else:
            logger.error(
                f"Unsupported data type for categorical plot: {type(data_pd_series)}")
            return

        num_unique = len(value_counts)

        if num_unique > top_n:
            logger.info(
                f"Column '{col_name}' has {num_unique} unique values, showing Top {top_n} and 'Others'.")
            top_values = value_counts.head(top_n).copy()
            other_count = value_counts.iloc[top_n:].sum()
            if other_count > 0:
                others_label = 'Others'
                others_series = pd.Series([other_count], index=[others_label])
                if others_label in top_values.index:
                    top_values[others_label] += other_count
                else:
                    top_values = pd.concat([top_values, others_series])
            data_to_plot = top_values
            default_title = f'Top {top_n} {col_name} Distribution'
        else:
            data_to_plot = value_counts
            default_title = f'{col_name} Distribution'

        # Use provided title or default
        plot_title = title if title is not None else default_title

        # --- Sorting and plotting logic (remains the same, minor adjustments for clarity) ---
        plot_index = data_to_plot.index.astype(str)
        try:
            sort_basis_index = value_counts.index
            sort_order_index = sort_basis_index.sort_values()
            sort_order = sort_order_index.astype(str)
            plot_index_ordered = [
                idx for idx in sort_order if idx in plot_index]
            if 'Others' in plot_index and 'Others' not in sort_order:
                if 'Others' in plot_index_ordered:
                    plot_index_ordered.remove('Others')
                plot_index_ordered.append('Others')
            elif 'Others' in plot_index and 'Others' in sort_order:
                pass
        except TypeError:
            logger.debug(
                f"Cannot sort index for '{col_name}', using count order.")
            sort_order = data_to_plot.index.astype(str)
            plot_index_ordered = plot_index.tolist()

        sns.barplot(x=plot_index, y=data_to_plot.values, ax=ax,
                    hue=plot_index, palette="viridis", legend=False,
                    order=plot_index_ordered)
        ax.set_title(plot_title)  # Use the final title
        ax.set_xlabel(xlabel)  # Use English xlabel
        ax.set_ylabel(ylabel)  # Use English ylabel
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_plot(fig, f"{filename_base}.png", plots_dir)  # Pass Path object

    except Exception as e:
        logger.exception(
            f"Error plotting distribution for column '{col_name}': {e}")
        plt.close(fig)


def log_value_counts(data, column_name, top_n=10, is_already_counts=False, normalize=True):
    """记录 Series 或 Dask Series 的值分布。"""
    logger.info(f"--- 分析列：{column_name} 值分布 ---")

    dist_df = None  # Initialize dist_df
    total_count = 0  # Initialize total_count

    # --- Existing logic to calculate dist_df and total_count ---
    if is_already_counts:
        if not isinstance(data, pd.DataFrame):
            logger.error(
                f"当 is_already_counts=True 时，期望输入为 Pandas DataFrame，但收到 {type(data)}")
            return
        if len(data.columns) < 2:
            logger.error(
                "当 is_already_counts=True 时，输入的 DataFrame 至少需要两列 (值和计数)。")
            return
        dist_df = data.copy()
        value_col = dist_df.columns[0]
        count_col = dist_df.columns[1]
        dist_df.columns = ['value', '计数']
        count_col = '计数'

        total_count = dist_df[count_col].sum()
        if normalize and total_count > 0:
            dist_df['百分比 (%)'] = (dist_df[count_col] /
                                  total_count * 100).round(2)
        else:
            dist_df['百分比 (%)'] = 0.0
        dist_df = dist_df.set_index('value')

    elif isinstance(data, (pd.Series, pd.Index)):
        try:
            value_counts = data.value_counts(dropna=False)
            total_count = len(data)
            dist_df = pd.DataFrame({'计数': value_counts})
            if normalize and total_count > 0:
                dist_df['百分比 (%)'] = (dist_df['计数'] /
                                      total_count * 100).round(2)
            else:
                dist_df['百分比 (%)'] = 0.0
        except Exception as e:
            logger.error(f"计算 Pandas Series/Index '{column_name}' 值计数时出错：{e}")
            return
    else:
        logger.error(f"不支持的数据类型进行值计数：{type(data)} for column '{column_name}'")
        return

    if dist_df is None or dist_df.empty:
        logger.warning(f"未能计算列 '{column_name}' 的值分布，或结果为空。")
        if isinstance(data, (pd.Series, pd.Index)) and not is_already_counts:
            nan_count = data.isnull().sum()
            if total_count > 0 and nan_count > 0:
                logger.warning(
                    f"列 '{column_name}' 包含 {nan_count} 个 NaN 值 ({nan_count/total_count*100:.2f}%)。")
            elif total_count > 0:
                logger.info(f"列 '{column_name}' 不包含 NaN 值。")
            else:
                logger.warning(
                    f"无法计算 Pandas Series/Index '{column_name}' 的大小。")
        return

    count_col_name = '计数'
    try:
        dist_df = dist_df.sort_values(by=count_col_name, ascending=False)
    except Exception as e:
        logger.error(
            f"按 '{count_col_name}' 列排序时出错：{e}\nDataFrame:\n{dist_df.head()}")
        return

    original_len = len(dist_df)
    if top_n is not None and original_len > top_n:
        logger.info(
            f"列 '{column_name}' 唯一值过多 ({original_len})，仅显示 Top {top_n}。")
        display_df = dist_df.head(top_n)
        log_message = f"值分布 (Top {top_n}, 含 NaN):\n{display_df.to_string()}"
    else:
        display_df = dist_df
        log_message = f"值分布 (含 NaN):\n{display_df.to_string()}"

    logger.info(log_message)

    nan_present_in_index = False
    try:
        nan_present_in_index = any(pd.isna(idx) for idx in display_df.index)
    except TypeError:
        logger.warning(f"无法直接检查列 '{column_name}' 值计数索引中的 NaN。")

    if nan_present_in_index:
        try:
            nan_rows = display_df[[pd.isna(idx) for idx in display_df.index]]
            if not nan_rows.empty:
                nan_count = nan_rows['计数'].iloc[0]
                nan_perc = nan_rows['百分比 (%)'].iloc[0]
                logger.warning(
                    f"列 '{column_name}' 存在缺失值 (NaN)。数量：{nan_count}, 百分比：{nan_perc:.2f}%")
            else:
                logger.info(f"列 '{column_name}' 值计数中未明确找到 NaN 行，尽管索引检查提示存在。")
        except KeyError:
            logger.warning(
                f"无法在列 '{column_name}' 的值计数中定位 NaN 行进行统计 (KeyError)。")
        except Exception as e:
            logger.warning(f"检查列 '{column_name}' 的 NaN 统计时出错：{e}")
    else:
        if not is_already_counts and isinstance(data, (pd.Series, pd.Index)):
            if data.isnull().sum() == 0:
                logger.info(f"列 '{column_name}' 值计数中未发现 NaN。")

    unique_count = original_len
    logger.info(f"列 '{column_name}' 唯一值数量 (含 NaN): {unique_count}")
    logger.info("-" * 20)


def analyze_timestamp_frequency_pandas(
    pdf: pd.DataFrame,
    id_val: str,
    id_col_name: str,
    timestamp_col: str = 'timestamp'
):
    """
    Analyzes and logs the frequency of timestamp intervals for a given ID in a Pandas DataFrame.
    Assumes the DataFrame contains data for a single ID and is sorted by timestamp.

    Args:
        pdf (pd.DataFrame): Pandas DataFrame containing timestamp data for a single ID.
        id_val (str): The specific unique identifier value being analyzed (for logging).
        id_col_name (str): The name of the ID column (e.g., 'unique_id', 'location_id').
        timestamp_col (str): The name of the timestamp column.
    """
    if pdf is None or pdf.empty:
        logger.warning(
            f"DataFrame for {id_col_name} '{id_val}' is empty, skipping frequency analysis.")
        return
    if timestamp_col not in pdf.columns:
        logger.error(
            f"Timestamp column '{timestamp_col}' not found in DataFrame for {id_col_name} '{id_val}'.")
        return
    if not pd.api.types.is_datetime64_any_dtype(pdf[timestamp_col]):
        logger.warning(
            f"Timestamp column '{timestamp_col}' in DataFrame for {id_col_name} '{id_val}' is not datetime type. Attempting conversion.")
        pdf[timestamp_col] = pd.to_datetime(
            pdf[timestamp_col], errors='coerce')
        pdf = pdf.dropna(subset=[timestamp_col])
        if pdf.empty:
            logger.error("No valid timestamps after conversion.")
            return

    try:
        # Ensure sorted (although typically called on pre-sorted data)
        pdf_sorted = pdf.sort_values(timestamp_col)
        # Calculate difference and remove first NaN
        time_diffs = pdf_sorted[timestamp_col].diff().dropna()

        if not time_diffs.empty:
            # Convert Timedelta to string for value_counts
            freq_counts = time_diffs.astype(str).value_counts()
            if not freq_counts.empty:
                logger.info(
                    f"--- {id_col_name}: {id_val} Timestamp Interval Frequency (Pandas Sample) ---")
                log_value_counts(freq_counts, f"ID {id_val} Interval", top_n=5)
                if len(freq_counts) > 1:
                    logger.warning(
                        f" {id_col_name} '{id_val}' has multiple time intervals detected in sample.")
            else:
                # This case is unlikely if time_diffs is not empty, but included for completeness
                logger.info(
                    f" {id_col_name} '{id_val}' has only one unique time interval after diff/dropna in sample (or value_counts failed).")
        elif len(pdf) > 1:
            # If diffs are empty but more than 1 row exists, it might imply identical timestamps
            logger.warning(
                f" {id_col_name} '{id_val}' has multiple rows but no calculable time differences (duplicate timestamps?).")
        else:  # len(pdf) <= 1
            logger.info(
                f" {id_col_name} '{id_val}' has less than two timestamps in sample, cannot calculate intervals.")
    except Exception as freq_e:
        logger.exception(
            f"Error analyzing frequency for {id_col_name} '{id_val}' in sample: {freq_e}")


# --- Add function for comparison boxplots ---
def plot_comparison_boxplot(
    pdf: pd.DataFrame,
    x_col: str,
    y_col: str,
    plots_dir: Path,
    filename_prefix: str,
    title_prefix: str,
    x_label: Optional[str] = None,
    y_label_orig: Optional[str] = None,
    y_label_log: Optional[str] = None,
    order: Optional[list] = None,
    palette: str = "viridis",
    log_epsilon: float = 1e-6
):
    """
    Generates and saves two boxplots for y_col vs x_col:
    1. Original scale (outliers hidden by default).
    2. Log1p scale (outliers shown by default).

    Args:
        pdf (pd.DataFrame): Input Pandas DataFrame.
        x_col (str): Name of the categorical column for the x-axis.
        y_col (str): Name of the numerical column for the y-axis.
        plots_dir (Path): Directory to save the plots.
        filename_prefix (str): Prefix for the output plot filenames.
        title_prefix (str): Prefix for the plot titles.
        x_label (Optional[str]): Label for the x-axis. Defaults to x_col capitalized.
        y_label_orig (Optional[str]): Label for the y-axis (original scale). Defaults to 'y_col'.
        y_label_log (Optional[str]): Label for the y-axis (log scale). Defaults to 'log1p(y_col + epsilon)'.
        order (Optional[list]): Order for plotting the categories on the x-axis.
        palette (str): Color palette for seaborn plots.
        log_epsilon (float): Small value added before log1p transformation.
    """
    if pdf is None or pdf.empty:
        logger.warning(
            f"Input DataFrame for '{title_prefix}' is empty, skipping boxplots.")
        return
    if x_col not in pdf.columns or y_col not in pdf.columns:
        logger.error(
            f"Missing required columns '{x_col}' or '{y_col}' for boxplots.")
        return

    # Prepare labels if not provided
    x_label = x_label or x_col.replace('_', ' ').capitalize()
    y_label_orig = y_label_orig or y_col
    y_label_log = y_label_log or f'log1p({y_col} + epsilon)'

    # Ensure plots_dir is a Path object and exists
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Original scale plot ---
    logger.debug(
        f"Plotting original scale boxplot for {title_prefix} vs {x_col}")
    fig_orig = None  # Initialize figure variable
    try:
        fig_orig, ax_orig = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=pdf, x=x_col, y=y_col, hue=x_col,
                    legend=False, showfliers=False, ax=ax_orig, palette=palette, order=order)
        ax_orig.set_title(
            f'{title_prefix} by {x_label} (Original Scale, No Outliers)')
        ax_orig.set_xlabel(x_label)
        ax_orig.set_ylabel(y_label_orig)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        save_plot(fig_orig, f'{filename_prefix}_boxplot_orig.png', plots_dir)
    except Exception as plot_orig_e:
        logger.exception(
            f"Error plotting original scale boxplot for {x_col}: {plot_orig_e}")
    finally:
        if fig_orig:  # Check if figure was created before trying to close
            plt.close(fig_orig)

    # --- Log scale plot ---
    logger.debug(f"Plotting log1p scale boxplot for {title_prefix} vs {x_col}")
    fig_log = None  # Initialize figure variable
    try:
        # Create a copy for manipulation to avoid SettingWithCopyWarning
        pdf_log = pdf[[x_col, y_col]].copy()
        pdf_log['y_numeric'] = pd.to_numeric(pdf_log[y_col], errors='coerce')
        pdf_log['y_log1p'] = np.nan  # Initialize log column

        # Apply log1p only to valid, non-negative numeric values
        valid_mask = (pdf_log['y_numeric'] >= 0) & pdf_log['y_numeric'].notna(
        ) & pdf_log[x_col].notna()

        if not valid_mask.any():
            logger.warning(
                f"No valid non-negative numeric '{y_col}' values found for log plot with {x_col}.")
            return

        pdf_log.loc[valid_mask, 'y_log1p'] = np.log1p(
            pdf_log.loc[valid_mask, 'y_numeric'] + log_epsilon)

        pdf_plot_log = pdf_log.dropna(subset=['y_log1p', x_col])

        if pdf_plot_log.empty:
            logger.warning(
                f"No valid log values or target values to plot for {x_col}, skipping log scale box plot.")
        else:
            fig_log, ax_log = plt.subplots(figsize=(12, 7))
            sns.boxplot(data=pdf_plot_log, x=x_col, y='y_log1p', hue=x_col,
                        legend=False, showfliers=True, ax=ax_log, palette=palette, order=order)
            ax_log.set_title(f'{title_prefix} by {x_label} (Log1p Scale)')
            ax_log.set_xlabel(x_label)
            ax_log.set_ylabel(y_label_log)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            save_plot(
                fig_log, f'{filename_prefix}_boxplot_log1p.png', plots_dir)
    except Exception as plot_log_e:
        logger.exception(
            f"Error plotting log scale boxplot for {x_col}: {plot_log_e}")
    finally:
        if fig_log:  # Check if figure was created before trying to close
            plt.close(fig_log)


def sample_spark_ids_and_collect(
    sdf: DataFrame,
    id_col: str,
    n_samples: int,
    random_state: int,
    select_cols: List[str],
    spark: SparkSession,
    timestamp_col: str = 'timestamp'  # Assume timestamp col exists for sorting
) -> Optional[pd.DataFrame]:
    """
    Samples n_samples distinct IDs from a Spark DataFrame, filters the DataFrame,
    selects specified columns, collects to Pandas, and sorts by ID and timestamp.

    Args:
        sdf (DataFrame): Input Spark DataFrame.
        id_col (str): Name of the column containing the IDs to sample from.
        n_samples (int): Number of distinct IDs to sample.
        random_state (int): Random seed for sampling.
        select_cols (List[str]): List of column names to select before collecting.
        spark (SparkSession): Active SparkSession.
        timestamp_col (str): Name of the timestamp column for sorting. Defaults to 'timestamp'.

    Returns:
        Optional[pd.DataFrame]: A Pandas DataFrame containing the sampled, collected,
                                 and sorted data, or None if an error occurs.
    """
    logger.info(
        f"Starting Spark ID sampling and collection for column '{id_col}'...")
    if not isinstance(sdf, DataFrame):
        logger.error("Input 'sdf' must be a Spark DataFrame.")
        return None
    if id_col not in sdf.columns:
        logger.error(f"ID column '{id_col}' not found in Spark DataFrame.")
        return None
    if not all(col in sdf.columns for col in select_cols):
        missing_select = [c for c in select_cols if c not in sdf.columns]
        logger.error(
            f"Columns specified in 'select_cols' not found: {missing_select}")
        return None
    if timestamp_col not in select_cols:
        logger.warning(
            f"Timestamp column '{timestamp_col}' not in 'select_cols'. Will sort only by ID if timestamp is missing.")
        # Adjust sort columns later if timestamp is missing
        sort_cols = [id_col]
    else:
        sort_cols = [id_col, timestamp_col]

    pdf_collected = None  # Initialize

    try:
        # 1. Spark 获取并抽样 unique_id
        logger.info(
            f"Getting distinct '{id_col}' values from Spark DataFrame...")
        all_ids_sdf = sdf.select(id_col).distinct()
        num_distinct_ids = all_ids_sdf.count()  # Action

        if num_distinct_ids == 0:
            logger.warning(f"No distinct IDs found in column '{id_col}'.")
            return None

        if n_samples <= 0:
            logger.warning(
                f"Requested sample size ({n_samples}) is not positive. Skipping.")
            return None

        actual_n_samples = min(n_samples, num_distinct_ids)
        if num_distinct_ids < n_samples:
            logger.warning(
                f"Total distinct ID count ({num_distinct_ids}) is less than requested sample size ({n_samples}), using all {num_distinct_ids} IDs.")

        logger.info(
            f"Randomly sampling {actual_n_samples} '{id_col}' values using Spark RDD takeSample...")
        # takeSample returns a list of Row objects
        sampled_ids_rows: List = all_ids_sdf.rdd.takeSample(
            False, actual_n_samples, seed=random_state)  # Action

        # Extract ID values from Row objects
        sampled_ids = [row[id_col] for row in sampled_ids_rows]
        if not sampled_ids:
            logger.error(f"Failed to sample IDs from column '{id_col}'.")
            return None
        logger.info(
            f"Selected {len(sampled_ids)} IDs (first 5): {sampled_ids[:5]}")

        # 2. Spark 过滤数据
        logger.info(
            f"Filtering Spark DataFrame for selected IDs in '{id_col}'...")
        # Use isin for filtering based on the collected list of IDs
        sdf_sample_filtered = sdf.filter(F.col(id_col).isin(sampled_ids))

        # 3. 选择列并收集到 Pandas
        logger.info(
            f"Selecting columns {select_cols} and collecting filtered data to Pandas...")
        sdf_to_collect = sdf_sample_filtered.select(select_cols)
        try:
            pdf_collected = sdf_to_collect.toPandas()  # Action
            if pdf_collected is None or pdf_collected.empty:
                logger.error(
                    "Collecting sample data to Pandas failed or resulted in an empty DataFrame.")
                return None
            logger.info(
                f"Pandas DataFrame collected, containing {len(pdf_collected):,} rows for {len(sampled_ids)} IDs.")
        except Exception as collect_e:
            logger.exception(
                f"Error collecting Spark sample data to Pandas: {collect_e}. Trying without Arrow...")
            try:
                # Try disabling Arrow and retry
                spark.conf.set(
                    "spark.sql.execution.arrow.pyspark.enabled", "false")
                pdf_collected = sdf_to_collect.toPandas()  # Action
                if pdf_collected is None or pdf_collected.empty:
                    logger.error(
                        "Collecting sample data to Pandas (without Arrow) still failed or resulted empty.")
                    return None
                logger.info(
                    f"(Arrow disabled) Pandas DataFrame collected, {len(pdf_collected):,} rows.")
            except Exception as collect_e_no_arrow:
                logger.exception(
                    f"Error collecting Spark sample data to Pandas (without Arrow): {collect_e_no_arrow}")
                return None
            finally:
                # Restore Arrow setting (optional, but good practice)
                spark.conf.set(
                    "spark.sql.execution.arrow.pyspark.enabled", "true")

        # 4. 在 Pandas 中排序和处理时间戳
        logger.info(f"Sorting Pandas DataFrame by {sort_cols}...")
        pdf_collected = pdf_collected.sort_values(sort_cols)

        # Ensure timestamp column is datetime type if it exists
        if timestamp_col in pdf_collected.columns:
            if not pd.api.types.is_datetime64_any_dtype(pdf_collected[timestamp_col]):
                logger.info(
                    f"Converting '{timestamp_col}' column to datetime objects in Pandas...")
                pdf_collected[timestamp_col] = pd.to_datetime(
                    pdf_collected[timestamp_col], errors='coerce')
                # Check for conversion failures
                if pdf_collected[timestamp_col].isnull().any():
                    original_rows = len(pdf_collected)
                    pdf_collected = pdf_collected.dropna(
                        subset=[timestamp_col])
                    logger.warning(
                        f"Removed {original_rows - len(pdf_collected)} rows due to timestamp conversion errors.")
                    if pdf_collected.empty:
                        logger.error(
                            "DataFrame became empty after removing rows with timestamp conversion errors.")
                        return None
        else:
            logger.debug(
                f"Timestamp column '{timestamp_col}' not found in collected data for datetime conversion.")

        logger.info(
            "Spark ID sampling, collection, and initial Pandas processing complete.")
        return pdf_collected

    except Exception as e:
        logger.exception(
            f"Error during Spark ID sampling and collection for '{id_col}': {e}")
        return None
