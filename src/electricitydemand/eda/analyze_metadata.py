from pathlib import Path  # Use pathlib

import matplotlib.pyplot as plt
import numpy as np  # Import numpy
import pandas as pd
import seaborn as sns
from loguru import logger

# 使用相对导入
from ..utils.eda_utils import (
    log_value_counts,
    plot_categorical_distribution,
    plot_numerical_distribution,
)


def analyze_metadata_categorical(pdf_metadata: pd.DataFrame):
    """
    Analyzes categorical features in the metadata Pandas DataFrame.

    Args:
        pdf_metadata: Pandas DataFrame containing metadata.
    """
    logger.info("--- 分析 Metadata 分类特征 ---")
    # Include the new columns in the analysis
    categorical_cols = [
        'dataset', 'location', 'timezone', 'freq', 'building_class'
    ]
    for col in categorical_cols:
        if col in pdf_metadata.columns:
            logger.info(f"--- 分析列: {col} ---")
            # logger.info(f"唯一值数量: {pdf_metadata[col].nunique()}") # Already logged in plot function
            # Value counts, handling potential NaN values
            counts = pdf_metadata[col].value_counts(dropna=False)
            logger.info(f"值分布 (Top 10):\n{counts.head(10).to_string()}")
            if len(counts) > 10:
                logger.info(f"... (共 {len(counts)} 个唯一值)")
        else:
            logger.warning(f"在 Metadata 中未找到分类列: {col}")
    logger.info("--- 完成 Metadata 分类特征分析 ---")


def plot_metadata_categorical(pdf_metadata: pd.DataFrame, plots_dir: str):
    """
    Plots distributions for categorical features in the metadata Pandas DataFrame.

    Args:
        pdf_metadata: Pandas DataFrame containing metadata.
        plots_dir: Directory to save the plots.
    """
    logger.info("--- 绘制 Metadata 分类特征分布图 ---")
    # Include the new columns in the plotting
    categorical_cols = [
        'dataset', 'location', 'timezone', 'freq', 'building_class'
    ]
    # Limit the number of categories to plot for clarity (e.g., for 'location')
    max_categories_to_plot = 15

    for col in categorical_cols:
        if col in pdf_metadata.columns:
            try:
                n_unique = pdf_metadata[col].nunique(dropna=False) # Include NaN in count
                logger.info(f"绘制 '{col}' 的分布图 (共 {n_unique} 个唯一值)...")

                plt.figure(figsize=(10, 6))
                # Handle NaNs explicitly if they exist
                if pdf_metadata[col].isnull().any():
                     # Use value_counts with dropna=False and fillna for plotting
                     plot_data = pdf_metadata[col].fillna('NaN').value_counts()
                else:
                    plot_data = pdf_metadata[col].value_counts()

                # Limit categories plotted if too many
                if n_unique > max_categories_to_plot:
                    top_categories = plot_data.nlargest(max_categories_to_plot)
                    # Create an 'Other' category for the rest
                    other_count = plot_data.iloc[max_categories_to_plot:].sum()
                    if other_count > 0:
                         top_categories['Other'] = other_count # Add 'Other' category using .loc
                    plot_data_final = top_categories
                    title = f'{col} 分布 (Top {max_categories_to_plot} & Other)'
                else:
                    plot_data_final = plot_data
                    title = f'{col} 分布'

                sns.barplot(x=plot_data_final.index, y=plot_data_final.values, palette='viridis', order=plot_data_final.index)
                plt.title(title)
                plt.xlabel(col)
                plt.ylabel('数量')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_filename = plots_dir / f"metadata_dist_{col}.png"
                plt.savefig(plot_filename)
                plt.close()
                logger.info(f"图表已保存: {plot_filename}")
            except Exception as e:
                logger.error(f"绘制列 '{col}' 时出错: {e}")
        else:
            logger.warning(f"在 Metadata 中未找到用于绘图的分类列: {col}")
    logger.info("--- 完成 Metadata 分类特征分布图绘制 ---")


def analyze_metadata_numerical(pdf_metadata: pd.DataFrame, plots_dir: str):
    """
    Analyzes numerical features (latitude, longitude, cluster_size) in the metadata.

    Args:
        pdf_metadata: Pandas DataFrame containing metadata.
        plots_dir: Directory to save the plots.
    """
    logger.info("--- 分析 Metadata 数值特征 ---")
    # Include 'cluster_size'
    numerical_cols = ['latitude', 'longitude', 'cluster_size']

    valid_cols = [col for col in numerical_cols if col in pdf_metadata.columns]

    if not valid_cols:
        logger.warning("Metadata 中未找到任何指定的数值列进行分析。")
        return

    try:
        # Basic statistics
        desc = pdf_metadata[valid_cols].describe()
        logger.info(f"数值特征描述性统计:\n{desc.to_string()}")

        # Plot histograms
        for col in valid_cols:
             # Check if column is numeric and not all NaN before plotting
             if pd.api.types.is_numeric_dtype(pdf_metadata[col]) and not pdf_metadata[col].isnull().all():
                 plt.figure(figsize=(10, 5))
                 # Handle potential infinite values if necessary before plotting histogram
                 data_to_plot = pdf_metadata[col].replace([np.inf, -np.inf], np.nan).dropna()
                 if not data_to_plot.empty:
                     sns.histplot(data_to_plot, kde=True, bins=30)
                     plt.title(f'{col} 分布')
                     plt.xlabel(col)
                     plt.ylabel('频数')
                     plt.tight_layout()
                     plot_filename = plots_dir / f"metadata_hist_{col}.png"
                     plt.savefig(plot_filename)
                     plt.close()
                     logger.info(f"'{col}' 直方图已保存: {plot_filename}")
                 else:
                     logger.warning(f"列 '{col}' 移除 NaN/inf 后为空，跳过绘制直方图。")
             else:
                  logger.warning(f"列 '{col}' 不是数值类型或全为 NaN，跳过绘制直方图。")

        # Scatter plot for latitude vs longitude (if both exist)
        if 'latitude' in valid_cols and 'longitude' in valid_cols:
            plt.figure(figsize=(8, 8))
            # Drop rows with NaN lat/lon for scatter plot
            scatter_data = pdf_metadata[['latitude', 'longitude']].dropna()
            if not scatter_data.empty:
                sns.scatterplot(data=scatter_data, x='longitude', y='latitude', alpha=0.5, s=10) # Smaller points
                plt.title('地理位置分布 (Latitude vs Longitude)')
                plt.xlabel('经度 (Longitude)')
                plt.ylabel('纬度 (Latitude)')
                plt.grid(True)
                plt.tight_layout()
                plot_filename = plots_dir / "metadata_location_scatter.png"
                plt.savefig(plot_filename)
                plt.close()
                logger.info(f"地理位置散点图已保存: {plot_filename}")
            else:
                 logger.warning("无有效的经纬度数据，跳过绘制地理位置散点图。")


    except Exception as e:
        logger.exception(f"分析 Metadata 数值特征时出错: {e}")

    logger.info("--- 完成 Metadata 数值特征分析 ---")


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
