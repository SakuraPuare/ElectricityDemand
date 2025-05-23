import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# 移除 Dask 导入
# import dask.dataframe as dd
# 引入 Spark 相关库
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Add StringType
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, StringType

# 使用相对导入
# 移除 dask_compute_context
from ..utils.eda_utils import plot_comparison_boxplot, save_plot


# Helper function modified for Spark demand input
def merge_demand_metadata_sample(
    sdf_demand: DataFrame,  # Changed input type
    pdf_metadata: pd.DataFrame,
    metadata_cols: List[str],
    sample_frac: float = 0.001,
    random_state: int = 42,
    spark: SparkSession = None,
) -> Optional[pd.DataFrame]:
    """
    Samples the Spark demand data, converts sample to Pandas,
    merges it with selected metadata columns, and returns a Pandas DataFrame.
    Logs remain Chinese.

    Args:
        sdf_demand: Spark DataFrame for demand data.
        pdf_metadata: Pandas DataFrame for metadata (must have unique_id as index or column).
        metadata_cols: List of metadata columns to merge.
        sample_frac: Fraction of demand data to sample.
        random_state: Random seed for sampling.
        spark: SparkSession (optional, not used directly in plotting after merge).

    Returns:
        A Pandas DataFrame containing the sampled and merged data, or None if an error occurs or data is empty.
    """
    logger.info(
        f"Merging Spark demand sample (frac={sample_frac}) with Pandas metadata columns: {metadata_cols}")
    start_time = time.time()
    try:
        # Validate inputs
        if not isinstance(sdf_demand, DataFrame):
            logger.error("sdf_demand must be a Spark DataFrame.")
            return None
        if not isinstance(pdf_metadata, pd.DataFrame):
            logger.error("pdf_metadata must be a Pandas DataFrame.")
            return None

        # Ensure pdf_metadata has unique_id ready for merge (as index preferably)
        if pdf_metadata.index.name != 'unique_id':
            if 'unique_id' in pdf_metadata.columns:
                logger.debug("Setting 'unique_id' as index for pdf_metadata.")
                pdf_metadata = pdf_metadata.set_index('unique_id').copy()
            else:
                logger.error(
                    "'unique_id' column or index not found in pdf_metadata.")
                return None

        required_demand_cols = ['unique_id', 'y']
        if not all(col in sdf_demand.columns for col in required_demand_cols):
            logger.error(
                f"sdf_demand must contain columns: {required_demand_cols}")
            return None

        # --- Sample using Spark ---
        logger.info(f"Sampling Spark DataFrame (fraction={sample_frac})...")
        # Ensure we only sample relevant columns to reduce data size before toPandas
        sdf_demand_sample = sdf_demand.select(required_demand_cols).sample(
            withReplacement=False, fraction=sample_frac, seed=random_state
        )

        # --- Collect sample to Pandas ---
        logger.info("Converting Spark sample to Pandas...")
        try:
            pdf_demand_sample = sdf_demand_sample.toPandas()
            logger.info(f"Pandas sample size: {len(pdf_demand_sample)}")
        except Exception as collect_e:
            logger.exception(
                f"Error converting Spark sample to Pandas: {collect_e}")
            # Consider adding Arrow fallback if needed
            return None

        if pdf_demand_sample.empty:
            logger.warning("Sampled demand data converted to Pandas is empty.")
            return None

        # --- Merge in Pandas ---
        # Set unique_id as index for merging
        if 'unique_id' not in pdf_demand_sample.columns:
            logger.error(
                "'unique_id' not found in the sampled demand data after conversion to Pandas.")
            return None
        pdf_demand_sample = pdf_demand_sample.set_index('unique_id')

        # Select only the required metadata columns that actually exist
        metadata_cols_to_select = [
            col for col in metadata_cols if col in pdf_metadata.columns]
        if not metadata_cols_to_select:
            logger.warning(
                f"None of the requested metadata columns {metadata_cols} found in pdf_metadata. Returning only demand sample.")
            # Return only the demand sample if no metadata cols are valid
            return pdf_demand_sample.reset_index()

        pdf_metadata_selected = pdf_metadata[metadata_cols_to_select]

        logger.debug(
            f"Merging sampled demand (size {len(pdf_demand_sample)}) with selected metadata (size {len(pdf_metadata_selected)}) on 'unique_id' index...")
        # Use pandas merge (join is also possible but merge is more explicit)
        # Validate index types before merging if necessary
        pdf_merged = pdf_demand_sample.merge(
            pdf_metadata_selected,
            left_index=True,
            right_index=True,
            how='left'  # Keep all demand samples, add metadata if available
        )
        logger.debug(f"Merge complete. Result size: {len(pdf_merged)}")

        # Reset index to make unique_id a regular column
        pdf_merged = pdf_merged.reset_index()

        # Log timing
        end_time = time.time()
        logger.info(
            f"Demand sample merge took {end_time - start_time:.2f} seconds. Result shape: {pdf_merged.shape}")

        if pdf_merged.empty:
            logger.warning("Merged DataFrame is empty.")
            return None

        return pdf_merged

    except Exception as e:
        # Log the full traceback for debugging
        logger.exception(f"Error in merge_demand_metadata_sample: {e}")
        return None


def analyze_demand_vs_metadata(sdf_demand: DataFrame, pdf_metadata: pd.DataFrame, target_col: str, plots_dir=None, sample_frac=0.001, random_state=42):
    """分析 Demand (y) 与 Metadata 特定分类特征的关系 (英文绘图标签)。"""
    if sdf_demand is None or pdf_metadata is None:
        logger.warning(
            "需要 Spark Demand DataFrame 和 Pandas Metadata DataFrame 进行关系分析。")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存关系分析图。")
        return
    plots_dir = Path(plots_dir)  # Convert to Path
    plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    if target_col not in pdf_metadata.columns:
        logger.error(
            f"Metadata (Pandas) 中缺少目标列或索引 '{target_col}'。无法分析关系。")
        return

    logger.info(
        f"--- 开始分析 Demand vs {target_col} 关系 (抽样比例: {sample_frac:.1%}) ---")

    try:
        # Use the updated helper function
        pdf_merged = merge_demand_metadata_sample(
            sdf_demand, pdf_metadata, [target_col], sample_frac, random_state)

        if pdf_merged is None or pdf_merged.empty:
            logger.warning(
                f"未能获取合并数据或数据为空，无法分析 Demand vs {target_col}。")
            return
        if 'y' not in pdf_merged.columns or target_col not in pdf_merged.columns:
            logger.warning(
                f"合并后的数据缺少 'y' 或 '{target_col}' 列。")
            return

        # Fill NA in target column for plotting if it's categorical
        if pd.api.types.is_categorical_dtype(pdf_merged[target_col]) or pd.api.types.is_object_dtype(pdf_merged[target_col]):
            # Use fillna on a copy to avoid SettingWithCopyWarning if pdf_merged is used later
            pdf_merged_plot = pdf_merged.copy()
            pdf_merged_plot[target_col] = pdf_merged_plot[target_col].fillna(
                '__MISSING__')
        else:
            pdf_merged_plot = pdf_merged  # Use original if no filling needed

        logger.info(f"绘制 Demand (y) vs {target_col} 对比箱线图...")

        # --- Use the new plotting function ---
        plot_comparison_boxplot(
            pdf=pdf_merged_plot,
            x_col=target_col,
            y_col='y',
            plots_dir=plots_dir,
            filename_prefix=f'demand_vs_{target_col}',
            title_prefix=f'Demand (y) vs {target_col.capitalize()}',
            y_label_orig='Electricity Demand (y) in kWh',
            y_label_log='log1p(Demand + epsilon)'
            # order=None (default)
        )

        logger.info(f"Demand vs {target_col} 分析完成。")

    except Exception as e:
        logger.exception(
            f"分析 Demand vs {target_col} 关系时出错: {e}")


# analyze_demand_vs_location uses the same merge helper, plotting logic is similar
def analyze_demand_vs_location(sdf_demand: DataFrame, pdf_metadata: pd.DataFrame, plots_dir=None, sample_frac=0.001, top_n=10, random_state=42):
    """分析 Demand (y) 与 Top N location 的关系 (英文绘图标签)。"""
    target_col = 'location'
    if pdf_metadata is None:
        logger.warning(
            "需要 Pandas Metadata DataFrame 进行关系分析。")
        return
    if target_col not in pdf_metadata.columns:
        logger.warning(
            f"Metadata (Pandas) 中缺少目标列 '{target_col}'。尝试使用 'location_id'。")
        target_col = 'location_id'  # Fallback to location_id if location is missing
        if target_col not in pdf_metadata.columns:
            logger.error(
                f"Metadata (Pandas) 中也缺少 '{target_col}'。无法分析关系。")
            return

    logger.info(
        f"--- 开始分析 Demand vs {target_col} 关系 (Top {top_n}, 抽样比例: {sample_frac:.1%}) ---")

    try:
        # Preprocess target column in metadata before merging (using a copy)
        pdf_metadata_processed = pdf_metadata.copy()
        pdf_metadata_processed[target_col] = pdf_metadata_processed[target_col].fillna(
            'Missing')
        # Ensure unique_id is index if not already
        if pdf_metadata_processed.index.name != 'unique_id' and 'unique_id' in pdf_metadata_processed.columns:
            pdf_metadata_processed = pdf_metadata_processed.set_index(
                'unique_id')
        elif pdf_metadata_processed.index.name != 'unique_id':
            logger.error(
                "Metadata needs 'unique_id' as index or column for merging.")
            return

        pdf_merged = merge_demand_metadata_sample(sdf_demand, pdf_metadata_processed, [
                                                  target_col], sample_frac, random_state)

        if pdf_merged is None or pdf_merged.empty:
            logger.warning(
                f"Failed to get merged data or data is empty, cannot analyze Demand vs {target_col}.")
            return
        if 'y' not in pdf_merged.columns or target_col not in pdf_merged.columns:
            logger.warning(
                f"Merged data is missing 'y' or '{target_col}' column.")
            return

        # Determine Top N locations from the *merged sample*
        location_counts = pdf_merged[target_col].value_counts()
        if location_counts.empty:
            logger.warning(
                f"No locations found in the merged sample for column '{target_col}'.")
            return

        top_locations = location_counts.head(top_n).index.tolist()
        logger.info(
            f"Top {top_n} locations (by data points in sample): {top_locations}")

        pdf_merged_top_n = pdf_merged[pdf_merged[target_col].isin(
            top_locations)].copy()
        if pdf_merged_top_n.empty:
            logger.warning(
                f"Could not find data for Top {top_n} locations in the sample, cannot plot.")
            return

        logger.info(f"绘制 Demand (y) vs Top {top_n} {target_col} 对比箱线图...")

        # --- Use the new plotting function ---
        plot_comparison_boxplot(
            pdf=pdf_merged_top_n,
            x_col=target_col,
            y_col='y',
            plots_dir=plots_dir,
            filename_prefix=f'demand_vs_top{top_n}_{target_col}',
            title_prefix=f'Demand (y) vs Top {top_n} {target_col.capitalize()}',
            y_label_orig='Electricity Demand (y) in kWh',
            y_label_log='log1p(Demand + epsilon)',
            order=top_locations  # Pass the order
        )

        logger.info(f"Demand vs {target_col} analysis complete.")

    except Exception as e:
        logger.exception(
            f"Error analyzing Demand vs {target_col} relationship: {e}")


# --- analyze_demand_vs_weather: Spark implementation ---
def analyze_demand_vs_weather(
    sdf_demand: DataFrame,
    pdf_metadata: pd.DataFrame,  # Keep metadata as Pandas for easy ID sampling
    sdf_weather: DataFrame,
    spark: SparkSession,  # Pass SparkSession
    plots_dir: Path,  # Use Path object
    n_sample_ids: int = 50,
    random_state: int = 42,
) -> None:
    """
    Analyzes the relationship between demand and weather features using Spark for merging.

    Args:
        sdf_demand: Spark DataFrame for demand data.
        pdf_metadata: Pandas DataFrame for metadata (needs location_id, unique_id).
        sdf_weather: Spark DataFrame for weather data.
        spark: Active SparkSession.
        plots_dir: Directory Path object to save plots.
        n_sample_ids: Number of unique_ids to sample for the analysis.
        random_state: Random seed for sampling unique_ids.
    """
    logger.info(
        f"--- Starting analysis of Demand vs Weather relationship (sampling {n_sample_ids} unique_ids) ---")
    start_time = time.time()
    plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure plot dir exists

    try:
        logger.info("Selecting random sample of unique IDs...")
        # Get unique IDs from metadata
        all_unique_ids = pdf_metadata.index.tolist()
        if len(all_unique_ids) <= n_sample_ids:
            sample_ids = all_unique_ids
            logger.warning(
                f"Requested sample size ({n_sample_ids}) >= total unique IDs ({len(all_unique_ids)}). Using all IDs.")
        else:
            sample_ids = random.sample(all_unique_ids, n_sample_ids)
        logger.info(f"Selected {len(sample_ids)} unique IDs for analysis.")

        # Filter demand data for sampled IDs
        logger.info("Filtering demand data for selected IDs...")
        sdf_demand_sample = sdf_demand.filter(
            F.col('unique_id').isin(sample_ids))

        # --- Align Timestamps ---
        # 1. Align demand data: create timestamp_hour and filter nulls/errors
        logger.info("Aligning demand timestamps to the hour...")
        sdf_demand_aligned = sdf_demand_sample.withColumn(
            "timestamp_hour", F.date_trunc('hour', F.col('timestamp'))
            # Ensure timestamp_hour is valid
        ).filter(F.col('timestamp_hour').isNotNull())

        # Join demand with metadata ONCE to get location_id efficiently
        logger.info(
            "Joining sampled demand data with metadata to get location_id...")
        # Get unique_id and location_id
        pdf_metadata_locations = pdf_metadata[['location_id']].reset_index()
        sdf_metadata_locations = spark.createDataFrame(pdf_metadata_locations)
        # Broadcast the smaller metadata DataFrame for efficiency
        sdf_demand_aligned = sdf_demand_aligned.join(
            F.broadcast(sdf_metadata_locations),
            'unique_id',
            'left'
            # Keep only rows with a valid location_id
        ).filter(F.col('location_id').isNotNull())

        # 2. Align weather data: create timestamp_hour and select relevant columns
        logger.info(
            "Aligning weather timestamps to the hour and selecting relevant columns...")
        weather_cols = [
            'location_id',
            'timestamp',  # Keep original timestamp for now, create timestamp_hour
            'temperature_2m',
            'relative_humidity_2m',
            'apparent_temperature',
            'precipitation',
            'wind_speed_10m',
            'cloud_cover',
        ]
        # Ensure timestamp is truncated to hour for joining
        sdf_weather_aligned = sdf_weather.withColumn(
            "timestamp_hour", F.date_trunc('hour', F.col('timestamp'))
            # Filter any nulls after truncation
        ).filter(F.col('timestamp_hour').isNotNull())

        # Drop the original timestamp AFTER creating timestamp_hour if needed,
        # or select explicitly. Select is safer.
        sdf_weather_aligned = sdf_weather_aligned.select(
            "location_id",
            "timestamp_hour",  # Use the newly created hour-truncated timestamp
            'temperature_2m',
            'relative_humidity_2m',
            'apparent_temperature',
            'precipitation',
            'wind_speed_10m',
            'cloud_cover',
        )

        # --- Join Data ---
        logger.info(
            "Joining aligned demand+location with aligned weather data using Spark...")
        # Join on both location_id and the hour-truncated timestamp
        sdf_merged = sdf_demand_aligned.join(
            sdf_weather_aligned,
            # Use list for multiple columns
            on=['location_id', 'timestamp_hour'],
            how='left'
        )

        # --- Collect and Analyze ---
        logger.info(
            f"Collecting the merged data ({n_sample_ids} IDs) to Pandas DataFrame for correlation analysis...")
        # Limit the collected data if it's potentially huge, though n_sample_ids should keep it reasonable.
        # Add a safety limit if needed: .limit(500000) # Example limit
        pdf_merged = sdf_merged.toPandas()
        logger.info(f"Collected {len(pdf_merged):,} rows into Pandas.")

        # --- Correlation Analysis ---
        if not pdf_merged.empty:
            weather_features = [
                'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                'precipitation', 'wind_speed_10m', 'cloud_cover'
            ]
            numeric_cols_for_corr = ['y'] + weather_features

            # Check if all expected columns exist before calculating correlation
            missing_cols = [
                col for col in numeric_cols_for_corr if col not in pdf_merged.columns]
            if missing_cols:
                logger.error(
                    f"Missing columns needed for correlation: {missing_cols}. Skipping correlation calculation.")
            else:
                # Ensure columns are numeric, coercing errors to NaN
                for col in numeric_cols_for_corr:
                    pdf_merged[col] = pd.to_numeric(
                        pdf_merged[col], errors='coerce')

                # Drop rows with NaN in the columns used for correlation
                pdf_merged_cleaned = pdf_merged.dropna(
                    subset=numeric_cols_for_corr)

                if not pdf_merged_cleaned.empty:
                    correlation_matrix = pdf_merged_cleaned[numeric_cols_for_corr].corr(
                    )
                    logger.info("Correlation matrix between demand (y) and weather features:\n" +
                                correlation_matrix['y'].to_string())
                else:
                    logger.warning(
                        "DataFrame is empty after dropping NaNs for correlation analysis. Cannot compute correlation.")

        else:
            logger.warning(
                "Merged DataFrame is empty. Cannot perform correlation analysis.")

    except Exception as e:
        logger.exception(f"Error during Demand vs Weather analysis: {e}")
    finally:
        # Optional: Clean up cache if used
        # spark.catalog.clearCache()
        end_time = time.time()
        logger.info(
            f"--- Demand vs Weather analysis took {end_time - start_time:.2f} seconds ---")


def analyze_demand_vs_dataset(
    sdf_demand: DataFrame,
    pdf_metadata: pd.DataFrame,
    plots_dir: Path,
    sample_frac: float = 0.001,  # 使用与之前分析一致的抽样比例
    spark: SparkSession = None  # Keep spark=None if not directly needed after merge
):
    """
    Analyzes and visualizes the relationship between electricity demand (y)
    and the source dataset using Spark sampling and Pandas plotting.

    Args:
        sdf_demand: Spark DataFrame containing 'unique_id' and 'y'.
        pdf_metadata: Pandas DataFrame containing 'unique_id' (as index) and 'dataset'.
        plots_dir: Path to save the plots.
        sample_frac: Fraction of demand data to sample for analysis.
        spark: SparkSession (optional, not used directly in plotting after merge).
    """
    target_col = 'dataset'
    logger.info(
        f"--- Starting analysis of Demand vs {target_col} relationship (sample fraction: {sample_frac*100:.1f}%) ---")
    start_time = time.time()

    # 1. Merge sample demand with metadata dataset column
    merged_sample_df = merge_demand_metadata_sample(
        sdf_demand,
        pdf_metadata,
        [target_col],  # Only need the dataset column
        sample_frac=sample_frac,
        spark=spark  # Pass spark if needed by merge function internally
    )

    if merged_sample_df is None or merged_sample_df.empty:
        logger.warning(
            f"No merged data available for Demand vs {target_col} analysis. Skipping.")
        return

    # Use a copy for potential modifications
    plot_df = merged_sample_df.copy()

    # Replace potential None values in target_col if necessary (though unlikely for dataset)
    if plot_df[target_col].isnull().any():
        logger.warning(
            f"Found missing values in '{target_col}'. Filling with 'Missing'.")
        plot_df[target_col] = plot_df[target_col].fillna('Missing')

    # 2. Log descriptive statistics per dataset
    try:
        # Calculate stats on the potentially modified plot_df
        stats_per_dataset = plot_df.groupby(target_col)['y'].describe()
        logger.info(
            f"'y' descriptive statistics grouped by {target_col} (based on {sample_frac*100:.1f}% sample):")
        # Use context manager for better formatting
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            logger.info(f"\n{stats_per_dataset}")

    except Exception as e:
        logger.exception(
            f"Error calculating descriptive stats for y by {target_col}: {e}")

    # 3. Plotting using the new function
    logger.info(f"Plotting Demand (y) vs {target_col} comparison box plot...")

    plot_comparison_boxplot(
        pdf=plot_df,  # Use the (potentially modified) plot_df
        x_col=target_col,
        y_col='y',
        plots_dir=plots_dir,
        filename_prefix=f"demand_vs_{target_col}",
        title_prefix=f"Demand (y) vs {target_col.capitalize()}",
        y_label_orig="Demand (y) in kWh",
        y_label_log="Log1p(Demand + epsilon)",
        # Order might be useful if many datasets, otherwise default alphabetical/numerical
        # order=plot_df[target_col].unique().tolist() # Example: plot in observed order
    )

    end_time = time.time()
    logger.info(
        f"--- Demand vs {target_col} analysis took {end_time - start_time:.2f} seconds ---")
