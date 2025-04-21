import pandas as pd
from loguru import logger
from pathlib import Path  # Use pathlib

# 使用相对导入
from ..utils.eda_utils import log_value_counts, plot_categorical_distribution, plot_numerical_distribution


def analyze_metadata_categorical(pdf_metadata: pd.DataFrame, columns_to_analyze=None):
    """分析 Metadata DataFrame 中指定分类列的分布并记录。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过分类特征分析。")
        return
    if not isinstance(pdf_metadata, pd.DataFrame):
        logger.error("输入必须是 Pandas DataFrame。")
        return

    if columns_to_analyze is None:
        columns_to_analyze = ['building_class',
                              'location', 'freq', 'timezone', 'dataset']
    logger.info(
        f"--- 开始分析 Metadata 分类特征分布 ({', '.join(columns_to_analyze)}) ---")

    for col in columns_to_analyze:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过。")
            continue
        # Check if column contains only nulls
        if pdf_metadata[col].isnull().all():
            logger.warning(f"列 '{col}' 只包含 null 值，跳过分析。")
            continue
        # Use helper function, limit log output
        log_value_counts(pdf_metadata[col], col, top_n=20)

    logger.info("Metadata 分类特征分析完成。")


def plot_metadata_categorical(pdf_metadata: pd.DataFrame, columns_to_plot=None, top_n=10, plots_dir=None):
    """可视化 Metadata DataFrame 中指定分类列的分布并保存。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过分类特征绘图。")
        return
    if not isinstance(pdf_metadata, pd.DataFrame):
        logger.error("输入必须是 Pandas DataFrame。")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存 Metadata 分类特征图。")
        return
    plots_dir = Path(plots_dir)  # Convert to Path
    plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    if columns_to_plot is None:
        columns_to_plot = ['building_class',
                           'location', 'freq', 'timezone', 'dataset']
    logger.info(
        f"--- Starting plotting Metadata categorical feature distributions ({', '.join(columns_to_plot)}) ---")

    for col in columns_to_plot:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过绘图。")
            continue
        # Check if column contains only nulls
        if pdf_metadata[col].isnull().all():
            logger.warning(f"列 '{col}' 只包含 null 值，跳过绘图。")
            continue

        plot_categorical_distribution(pdf_metadata[col], col,
                                      f'metadata_distribution_{col}', plots_dir,
                                      top_n=top_n, title_prefix="Metadata ")  # Use helper

    logger.info("Metadata categorical feature plotting complete.")


def analyze_metadata_numerical(pdf_metadata: pd.DataFrame, columns_to_analyze=None, plots_dir=None):
    """分析 Metadata DataFrame 中指定数值列的分布并记录/绘图。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过数值特征分析。")
        return
    if not isinstance(pdf_metadata, pd.DataFrame):
        logger.error("输入必须是 Pandas DataFrame。")
        return

    plot = plots_dir is not None  # Whether to plot
    if plot:
        plots_dir = Path(plots_dir)  # Convert to Path
        plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    else:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Metadata 数值特征图。")

    if columns_to_analyze is None:
        columns_to_analyze = ['latitude', 'longitude', 'cluster_size']
    logger.info(
        f"--- Starting analysis of Metadata numerical feature distributions ({', '.join(columns_to_analyze)}) ---")

    for col in columns_to_analyze:
        if col not in pdf_metadata.columns:
            logger.warning(f"列 '{col}' 不在 Metadata DataFrame 中，跳过。")
            continue
        # Check if column contains only nulls
        if pdf_metadata[col].isnull().all():
            logger.warning(f"列 '{col}' 只包含 null 值，跳过数值分析。")
            continue
        # Try converting to numeric, coerce errors
        pdf_metadata_col_numeric = pd.to_numeric(
            pdf_metadata[col], errors='coerce')
        if pdf_metadata_col_numeric.isnull().all():
            logger.warning(f"列 '{col}' 转换为数值后全为 null，跳过数值分析。")
            continue

        logger.info(f"--- Analyzing column: {col} ---")
        try:
            desc_stats = pdf_metadata_col_numeric.describe()
            logger.info(f"Descriptive Statistics:\n{desc_stats.to_string()}")

            missing_count = pdf_metadata_col_numeric.isnull().sum()
            original_missing = pdf_metadata[col].isnull().sum()
            if original_missing != missing_count:
                logger.warning(
                    f"列 '{col}' 包含 {original_missing - missing_count} 个非数值类型的值被转换为 NaN。")

            if missing_count > 0:
                missing_perc = (
                    missing_count / len(pdf_metadata) * 100).round(2)
                logger.warning(
                    f"列 '{col}' 存在 {missing_count} ({missing_perc}%) 个缺失或无法转换的值。")

            if plot:
                # Drop NaN before plotting
                data_to_plot = pdf_metadata_col_numeric.dropna()
                if data_to_plot.empty:
                    logger.warning(f"列 '{col}' 除去 NaN 后为空，跳过绘图。")
                else:
                    # Use helper function to plot
                    plot_numerical_distribution(data_to_plot, col,
                                                f'metadata_distribution_{col}', plots_dir,
                                                title_prefix="Metadata ", kde=True,  # Default kde=True
                                                # Maybe hide outliers for cluster_size
                                                showfliers=(col != 'cluster_size'))
        except Exception as e:
            logger.exception(
                f"Error analyzing or plotting numerical column '{col}': {e}")

    logger.info("Metadata numerical feature analysis complete.")


def analyze_missing_locations(pdf_metadata: pd.DataFrame):
    """分析 Metadata 中位置信息缺失的行。"""
    if pdf_metadata is None or pdf_metadata.empty:
        logger.warning("输入的 Metadata DataFrame 为空，跳过缺失位置分析。")
        return
    if not isinstance(pdf_metadata, pd.DataFrame):
        logger.error("输入必须是 Pandas DataFrame。")
        return

    logger.info("--- 开始分析 Metadata 中缺失的位置信息 ---")
    location_cols = ['location_id', 'latitude', 'longitude', 'location']
    existing_location_cols = [
        col for col in location_cols if col in pdf_metadata.columns]
    if not existing_location_cols:
        logger.warning("Metadata 中不包含任何位置信息列，跳过分析。")
        return

    try:
        # Create a mask where all existing location columns are null
        missing_mask = pdf_metadata[existing_location_cols].isnull().all(
            axis=1)
        missing_rows_count = missing_mask.sum()

        if missing_rows_count == 0:
            logger.info("未发现所有位置信息均缺失的行。")
            return

        logger.info(
            f"发现 {missing_rows_count} 行的所有位置信息 ({', '.join(existing_location_cols)}) 均为缺失。")

        # Use .loc with the boolean mask which is generally safer
        # Use .copy() to avoid SettingWithCopyWarning
        missing_df = pdf_metadata.loc[missing_mask].copy()
        if not missing_df.empty:  # Check if the subset is not empty
            logger.info(f"缺失位置信息的行 (前 5 行):\n{missing_df.head().to_string()}")
            logger.info("分析缺失位置信息行的其他特征分布:")
            analyze_cols = ['dataset', 'building_class', 'freq']  # 分析示例列
            for col in analyze_cols:
                if col in missing_df.columns:
                    # Check if the column exists and the series is not empty before calling
                    if not missing_df[col].isnull().all():
                        # is_already_counts defaults to False
                        log_value_counts(
                            missing_df[col], f"(缺失位置) {col}", top_n=None)
                    else:
                        logger.warning(
                            f"列 '{col}' 在缺失位置信息的子集中为空或全为 null，跳过计数。")

                else:
                    logger.warning(f"列 '{col}' 不在缺失位置信息的 DataFrame 中。")
        else:
            logger.warning("创建缺失位置信息的子集 DataFrame 失败或为空。")
        logger.info("缺失位置信息分析完成。")

    except Exception as e:
        logger.exception(f"Error analyzing missing locations: {e}")
