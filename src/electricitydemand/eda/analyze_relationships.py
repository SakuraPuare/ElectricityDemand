import pandas as pd
import numpy as np
# 移除 Dask 导入
# import dask.dataframe as dd
# 引入 Spark 相关库
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# 使用相对导入
from ..utils.eda_utils import save_plot # 移除 dask_compute_context

# Helper function moved and renamed (no longer internal)
def merge_demand_metadata_sample(ddf_demand, pdf_metadata, metadata_cols, sample_frac, random_state):
    """Samples demand data and merges specified metadata columns."""
    logger.info(f"Merging demand sample (frac={sample_frac}) with metadata columns: {metadata_cols}")
    pdf_merged = pd.DataFrame() # Initialize empty DataFrame
    try:
        # 1. Sample demand data
        logger.debug("Sampling demand data...")
        # Select only necessary columns and sample
        ddf_demand_sample = ddf_demand[['unique_id', 'y']].sample(frac=sample_frac, random_state=random_state)

        # 2. Prepare metadata
        metadata_cols_to_select = ['unique_id'] + [col for col in metadata_cols if col in pdf_metadata.columns and col != 'unique_id']
        if not metadata_cols_to_select or 'unique_id' not in metadata_cols_to_select:
             logger.error("Metadata must contain 'unique_id' for merging.")
             return pdf_merged # Return empty
        pdf_metadata_subset = pdf_metadata[metadata_cols_to_select].drop_duplicates(subset=['unique_id']).copy()

        # 3. Merge (compute demand sample first)
        logger.debug("Computing demand sample to Pandas DataFrame...")
        with dask_compute_context(ddf_demand_sample) as persisted_demand:
             if not persisted_demand:
                 logger.error("Failed to persist demand sample for merge.")
                 return pdf_merged
             try:
                 pdf_demand_sample = persisted_demand[0].compute()
             except Exception as compute_e:
                 logger.exception(f"Error computing demand sample: {compute_e}")
                 return pdf_merged # Return empty

        if pdf_demand_sample.empty:
             logger.warning("Demand sample is empty after computation.")
             return pdf_merged

        logger.debug(f"Performing inner merge on 'unique_id' (Demand sample: {len(pdf_demand_sample)}, Metadata subset: {len(pdf_metadata_subset)})...")
        # Ensure 'unique_id' types match before merge
        try:
            pdf_demand_sample['unique_id'] = pdf_demand_sample['unique_id'].astype(str)
            pdf_metadata_subset['unique_id'] = pdf_metadata_subset['unique_id'].astype(str)
        except Exception as type_e:
            logger.error(f"Error converting 'unique_id' types before merge: {type_e}")
            return pdf_merged

        pdf_merged = pd.merge(pdf_demand_sample, pdf_metadata_subset, on='unique_id', how='inner')
        logger.info(f"Merge complete. Resulting sample size: {len(pdf_merged)} rows.")

        # Basic validation after merge
        if 'y' not in pdf_merged.columns:
            logger.error("Merged DataFrame is missing the 'y' column.")
            return pd.DataFrame() # Return empty
        for col in metadata_cols:
             if col not in pdf_merged.columns:
                  logger.warning(f"Merged DataFrame is missing the expected metadata column: '{col}'")


    except Exception as e:
        logger.exception(f"Error in merge_demand_metadata_sample: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    return pdf_merged


def analyze_demand_vs_metadata(ddf_demand, pdf_metadata, plots_dir=None, sample_frac=0.001, random_state=42):
    """分析 Demand (y) 与 Metadata 特征 (如 building_class) 的关系。"""
    if ddf_demand is None or pdf_metadata is None:
        logger.warning("Need Demand and Metadata data for relationship analysis.")
        return
    if plots_dir is None:
        logger.error("未提供 plots_dir，无法保存关系分析图。")
        return

    target_col = 'building_class'
    if target_col not in pdf_metadata.columns:
         logger.error(f"Metadata is missing the target column '{target_col}'. Cannot analyze relationship.")
         return

    logger.info(f"--- Starting analysis of Demand vs {target_col} relationship (sample fraction: {sample_frac:.1%}) ---")

    try:
        pdf_merged = merge_demand_metadata_sample(ddf_demand, pdf_metadata, [target_col], sample_frac, random_state)

        if pdf_merged is None or pdf_merged.empty:
             logger.warning(f"Failed to get merged data or data is empty, cannot analyze Demand vs {target_col}.")
             return
        if 'y' not in pdf_merged.columns or target_col not in pdf_merged.columns:
             logger.warning(f"Merged data is missing 'y' or '{target_col}' column.")
             return

        logger.info(f"Plotting Demand (y) vs {target_col} box plot...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- Original scale ---
        try:
             fig_orig, ax_orig = plt.subplots(figsize=(10, 6))
             sns.boxplot(data=pdf_merged, x=target_col, y='y', showfliers=False, ax=ax_orig, palette="viridis")
             ax_orig.set_title(f'Demand (y) Distribution by {target_col} (Original Scale, No Outliers)')
             ax_orig.set_xlabel(target_col)
             ax_orig.set_ylabel('Electricity Demand (y) in kWh')
             plt.tight_layout()
             save_plot(fig_orig, f'demand_vs_{target_col}_boxplot_orig.png', plots_dir)
        except Exception as plot_orig_e:
             logger.exception(f"Error plotting original scale boxplot for {target_col}: {plot_orig_e}")
             plt.close(fig_orig)


        # --- Log scale ---
        try:
            epsilon = 1e-6
            # Create log column safely, handle potential negative values and NaNs
            pdf_merged['y_log1p'] = np.nan # Initialize with NaN
            valid_mask = (pdf_merged['y'] >= 0) & (pdf_merged['y'].notna())
            pdf_merged.loc[valid_mask, 'y_log1p'] = np.log1p(pdf_merged.loc[valid_mask, 'y'] + epsilon)

            pdf_plot_log = pdf_merged.dropna(subset=['y_log1p', target_col]) # Drop NaN in y_log1p and target column

            if pdf_plot_log.empty:
                 logger.warning(f"No valid log values or target values to plot for {target_col}, skipping log scale box plot.")
            else:
                fig_log, ax_log = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=pdf_plot_log, x=target_col, y='y_log1p', showfliers=True, ax=ax_log, palette="viridis")
                ax_log.set_title(f'Demand (y) Distribution by {target_col} (Log1p Scale)')
                ax_log.set_xlabel(target_col)
                ax_log.set_ylabel('log1p(Electricity Demand (y) + epsilon)')
                plt.tight_layout()
                save_plot(fig_log, f'demand_vs_{target_col}_boxplot_log1p.png', plots_dir)
        except Exception as plot_log_e:
             logger.exception(f"Error plotting log scale boxplot for {target_col}: {plot_log_e}")
             if 'fig_log' in locals(): # Check if figure exists before trying to close
                plt.close(fig_log)


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
    if target_col not in pdf_metadata.columns:
         logger.error(f"Metadata is missing the target column '{target_col}'. Cannot analyze relationship.")
         return

    logger.info(f"--- Starting analysis of Demand vs {target_col} relationship (Top {top_n}, sample fraction: {sample_frac:.1%}) ---")

    try:
        # Preprocess location column before merging
        pdf_metadata_processed = pdf_metadata.copy()
        if target_col in pdf_metadata_processed.columns:
             pdf_metadata_processed[target_col] = pdf_metadata_processed[target_col].fillna('Missing')
        # else: # This case is already checked above
        #      logger.error(f"Metadata is missing '{target_col}' column, cannot proceed.")
        #      return

        pdf_merged = merge_demand_metadata_sample(ddf_demand, pdf_metadata_processed, [target_col], sample_frac, random_state)

        if pdf_merged is None or pdf_merged.empty:
             logger.warning(f"Failed to get merged data or data is empty, cannot analyze Demand vs {target_col}.")
             return
        if 'y' not in pdf_merged.columns or target_col not in pdf_merged.columns:
             logger.warning(f"Merged data is missing 'y' or '{target_col}' column.")
             return


        # Determine Top N locations from the *merged sample*
        location_counts = pdf_merged[target_col].value_counts()
        if location_counts.empty:
             logger.warning(f"No locations found in the merged sample for column '{target_col}'.")
             return

        top_locations = location_counts.head(top_n).index.tolist()
        logger.info(f"Top {top_n} locations (by data points in sample): {top_locations}")

        pdf_merged_top_n = pdf_merged[pdf_merged[target_col].isin(top_locations)].copy()
        if pdf_merged_top_n.empty:
             logger.warning(f"Could not find data for Top {top_n} locations in the sample, cannot plot.")
             return

        logger.info(f"Plotting Demand (y) vs Top {top_n} {target_col} box plot...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- Original scale ---
        try:
            fig_orig, ax_orig = plt.subplots(figsize=(15, 7))
            sns.boxplot(data=pdf_merged_top_n, x=target_col, y='y', showfliers=False, ax=ax_orig, palette="viridis", order=top_locations)
            ax_orig.set_title(f'Demand (y) Distribution by Top {top_n} {target_col} (Original Scale, No Outliers)')
            ax_orig.set_xlabel(target_col)
            ax_orig.set_ylabel('Electricity Demand (y) in kWh')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_plot(fig_orig, f'demand_vs_top{top_n}_{target_col}_boxplot_orig.png', plots_dir)
        except Exception as plot_orig_e:
             logger.exception(f"Error plotting original scale boxplot for {target_col}: {plot_orig_e}")
             plt.close(fig_orig)


        # --- Log scale ---
        try:
            epsilon = 1e-6
            # Create log column safely
            pdf_merged_top_n['y_log1p'] = np.nan # Initialize with NaN
            valid_mask = (pdf_merged_top_n['y'] >= 0) & (pdf_merged_top_n['y'].notna())
            pdf_merged_top_n.loc[valid_mask, 'y_log1p'] = np.log1p(pdf_merged_top_n.loc[valid_mask, 'y'] + epsilon)

            pdf_plot_log = pdf_merged_top_n.dropna(subset=['y_log1p', target_col]) # Drop NaN in y_log1p and target column

            if pdf_plot_log.empty:
                 logger.warning(f"No valid log values or target values to plot for Top {top_n} {target_col}, skipping log scale box plot.")
            else:
                fig_log, ax_log = plt.subplots(figsize=(15, 7))
                sns.boxplot(data=pdf_plot_log, x=target_col, y='y_log1p', showfliers=True, ax=ax_log, palette="viridis", order=top_locations)
                ax_log.set_title(f'Demand (y) Distribution by Top {top_n} {target_col} (Log1p Scale)')
                ax_log.set_xlabel(target_col)
                ax_log.set_ylabel('log1p(Electricity Demand (y) + epsilon)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                save_plot(fig_log, f'demand_vs_top{top_n}_{target_col}_boxplot_log1p.png', plots_dir)
        except Exception as plot_log_e:
             logger.exception(f"Error plotting log scale boxplot for {target_col}: {plot_log_e}")
             if 'fig_log' in locals():
                 plt.close(fig_log)


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
        if 'unique_id' not in pdf_metadata.columns:
            logger.error("Metadata is missing 'unique_id' column.")
            return
        valid_ids = pdf_metadata['unique_id'].dropna().unique()
        if len(valid_ids) == 0:
             logger.error("No valid unique_ids found in Metadata.")
             return

        if len(valid_ids) < n_sample_ids:
            logger.warning(f"Number of unique_ids in Metadata ({len(valid_ids)}) is less than requested sample size ({n_sample_ids}), using all available IDs.")
            sampled_unique_ids = valid_ids
            n_sample_ids = len(sampled_unique_ids)
        else:
            np.random.seed(random_state)
            sampled_unique_ids = np.random.choice(valid_ids, n_sample_ids, replace=False)
        logger.info(f"Selected unique_ids (first 10 if available): {sampled_unique_ids[:10].tolist()}...")

        # 2. Prepare Demand data (filter + get location_id)
        logger.info("Filtering Demand data and merging Metadata to get location_id...")
        if 'location_id' not in pdf_metadata.columns:
            logger.error("Metadata is missing 'location_id' column.")
            return
        # Ensure we get unique_id and location_id, handle potential NaNs
        pdf_metadata_sample = pdf_metadata[pdf_metadata['unique_id'].isin(sampled_unique_ids)][['unique_id', 'location_id']].drop_duplicates().dropna()
        if pdf_metadata_sample.empty:
            logger.warning("Sampled unique_ids have no valid non-NaN (unique_id, location_id) pairs in Metadata, cannot proceed.")
            return

        # Filter demand data
        ddf_demand_filtered = ddf_demand[ddf_demand['unique_id'].isin(sampled_unique_ids)][['unique_id', 'timestamp', 'y']].dropna(subset=['y', 'timestamp', 'unique_id'])
        # Dask merge Dask DF with Pandas DF needs careful handling of divisions/index
        # It's often safer to compute the smaller part (metadata) and use Dask's merge
        logger.debug("Performing Dask merge between filtered demand and metadata sample...")
        try:
             # Ensure compatible types for merge key
             ddf_demand_filtered['unique_id'] = ddf_demand_filtered['unique_id'].astype(str)
             pdf_metadata_sample['unique_id'] = pdf_metadata_sample['unique_id'].astype(str)
             # Dask merge needs known divisions if joining on index, but joining on column is generally safer
             ddf_demand_with_loc = dd.merge(ddf_demand_filtered, pdf_metadata_sample, on='unique_id', how='inner')
             # Persist the result of the merge
             ddf_demand_with_loc = ddf_demand_with_loc.persist()
             logger.info("Dask merge for location_id complete and persisted.")
        except Exception as merge_loc_e:
             logger.exception(f"Error during Dask merge for location_id: {merge_loc_e}")
             return


        # 3. Prepare Weather data
        weather_cols = ['location_id', 'timestamp', 'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'precipitation', 'wind_speed_10m', 'cloud_cover']
        existing_weather_cols = [col for col in weather_cols if col in ddf_weather.columns]
        if 'location_id' not in existing_weather_cols or 'timestamp' not in existing_weather_cols:
             logger.error("Weather data must contain 'location_id' and 'timestamp'.")
             return
        ddf_weather_subset = ddf_weather[existing_weather_cols].dropna(subset=['location_id', 'timestamp']) # Drop NaN keys early

        # 4. Merge (convert to Pandas for merge_asof)
        logger.info("Computing sampled Demand (with location_id) to Pandas DataFrame...")
        with dask_compute_context(ddf_demand_with_loc) as p_demand:
            if not p_demand:
                 logger.error("Failed to persist/get demand data with location.")
                 return
            try:
                pdf_demand_with_loc = p_demand[0].compute()
            except Exception as compute_demand_e:
                 logger.exception(f"Error computing demand data with location: {compute_demand_e}")
                 return
        num_demand_rows = len(pdf_demand_with_loc)
        logger.info(f"Demand Pandas DataFrame computed, containing {num_demand_rows:,} rows.")
        if pdf_demand_with_loc.empty:
             logger.warning("Demand data with location_id is empty after computation, cannot proceed.")
             return

        relevant_location_ids = pdf_demand_with_loc['location_id'].unique()
        if len(relevant_location_ids) == 0:
             logger.warning("No relevant location_ids found in the computed demand data.")
             return
        logger.info(f"Filtering Weather data, keeping only {len(relevant_location_ids)} relevant location_ids...")
        ddf_weather_filtered = ddf_weather_subset[ddf_weather_subset['location_id'].isin(relevant_location_ids)]

        logger.info("Computing filtered Weather data to Pandas DataFrame...")
        with dask_compute_context(ddf_weather_filtered) as p_weather:
             if not p_weather:
                 logger.error("Failed to persist/get filtered weather data.")
                 return
             try:
                 pdf_weather_filtered = p_weather[0].compute()
             except Exception as compute_weather_e:
                 logger.exception(f"Error computing filtered weather data: {compute_weather_e}")
                 return
        logger.info(f"Weather Pandas DataFrame computed, containing {len(pdf_weather_filtered):,} rows.")
        if pdf_weather_filtered.empty:
             logger.warning("Filtered weather data is empty after computation.")
             return


        logger.info("Preparing data for merge (sorting, deduplicating, type conversion)...")
        # Ensure correct dtypes before sort/dedup
        try:
            pdf_demand_with_loc['timestamp'] = pd.to_datetime(pdf_demand_with_loc['timestamp'])
            pdf_weather_filtered['timestamp'] = pd.to_datetime(pdf_weather_filtered['timestamp'])
            pdf_demand_with_loc['location_id'] = pdf_demand_with_loc['location_id'].astype(str)
            pdf_weather_filtered['location_id'] = pdf_weather_filtered['location_id'].astype(str)
        except Exception as dtype_e:
             logger.error(f"Error converting dtypes before merge_asof: {dtype_e}")
             return

        # Drop NaN keys, sort, deduplicate
        pdf_demand_with_loc.dropna(subset=['location_id', 'timestamp'], inplace=True)
        pdf_weather_filtered.dropna(subset=['location_id', 'timestamp'], inplace=True)

        pdf_demand_with_loc.sort_values(by=['location_id', 'timestamp'], inplace=True)
        pdf_weather_filtered.sort_values(by=['location_id', 'timestamp'], inplace=True)

        rows_before_dedup_demand = len(pdf_demand_with_loc)
        pdf_demand_with_loc.drop_duplicates(subset=['location_id', 'timestamp'], keep='first', inplace=True)
        rows_after_dedup_demand = len(pdf_demand_with_loc)
        if rows_before_dedup_demand > rows_after_dedup_demand: logger.warning(f"Removed {rows_before_dedup_demand - rows_after_dedup_demand} duplicate (location_id, timestamp) rows from Demand data")

        rows_before_dedup_weather = len(pdf_weather_filtered)
        pdf_weather_filtered.drop_duplicates(subset=['location_id', 'timestamp'], keep='first', inplace=True)
        rows_after_dedup_weather = len(pdf_weather_filtered)
        if rows_before_dedup_weather > rows_after_dedup_weather: logger.warning(f"Removed {rows_before_dedup_weather - rows_after_dedup_weather} duplicate (location_id, timestamp) rows from Weather data")

        pdf_demand_with_loc.reset_index(drop=True, inplace=True)
        pdf_weather_filtered.reset_index(drop=True, inplace=True)

        # Double-check sorting (merge_asof requires it strictly) - Optional but good practice
        # is_left_sorted = pdf_demand_with_loc.groupby('location_id')['timestamp'].is_monotonic_increasing.all() # Requires Pandas >= 1.2
        # is_right_sorted = pdf_weather_filtered.groupby('location_id')['timestamp'].is_monotonic_increasing.all()
        # Check manually for older pandas
        is_left_sorted = all(g['timestamp'].is_monotonic_increasing for _, g in pdf_demand_with_loc.groupby('location_id'))
        is_right_sorted = all(g['timestamp'].is_monotonic_increasing for _, g in pdf_weather_filtered.groupby('location_id'))

        if not is_left_sorted or not is_right_sorted:
             logger.error(f"Data sorting check failed! Left sorted: {is_left_sorted}, Right sorted: {is_right_sorted}. Cannot safely execute merge_asof.")
             # Add more detailed diagnostics here if needed
             # ...
             logger.error("Skipping merge_asof step due to sorting issues.")
             return # Return directly, do not proceed
        else:
            logger.info("Data sorting check passed.")
            logger.info("Merging data using Pandas merge_asof...")
            logger.warning("FIXME: THIS IS A BUG, IT IS NOT WORKING")
            try:
                pdf_merged = pd.merge_asof(
                    pdf_demand_with_loc,
                    pdf_weather_filtered,
                    on='timestamp',
                    by='location_id',
                    direction='backward', # Find nearest weather observation backward in time
                    tolerance=pd.Timedelta('1hour') # Allow up to 1 hour difference
                )
                logger.info(f"Final merge complete, resulting in {len(pdf_merged):,} rows.")
            except ValueError as ve:
                 # Can still fail sometimes due to edge cases or type issues
                 logger.error(f"Pandas merge_asof failed unexpectedly: {ve}")
                 logger.exception("Error during merge_asof execution")
                 return # Return on failure
            except Exception as merge_exc:
                 logger.exception("Unknown error during data merge")
                 return # Return on failure

        if pdf_merged.empty:
             logger.warning("Final merged data is empty after merge_asof, cannot perform relationship analysis.")
             return

        # 5. Calculate correlation
        logger.info("Calculating correlations between Demand (y) and weather features...")
        # Use only the weather columns that actually exist after the merge + 'y'
        correlation_cols_present = ['y'] + [col for col in existing_weather_cols if col in pdf_merged.columns and col not in ['location_id', 'timestamp']]

        # Ensure all columns for correlation are numeric
        numeric_correlation_cols = []
        for col in correlation_cols_present:
             # Check if column exists and is numeric
             if col in pdf_merged.columns and pd.api.types.is_numeric_dtype(pdf_merged[col]):
                  numeric_correlation_cols.append(col)
             else:
                  logger.warning(f"Column '{col}' is not numeric or not present in the final merged data, excluding from correlation.")


        if 'y' not in numeric_correlation_cols or len(numeric_correlation_cols) <= 1:
             logger.warning("Not enough numeric columns (including 'y') to calculate correlations.")
        else:
            # Drop rows where any of the correlation columns are NaN before calculating corr
            pdf_corr = pdf_merged[numeric_correlation_cols].dropna()
            if pdf_corr.empty:
                 logger.warning("Data is empty after dropping NaNs in correlation columns. Cannot calculate correlation.")
            else:
                try:
                    correlation_matrix = pdf_corr.corr()
                    if 'y' in correlation_matrix.columns:
                        logger.info(f"Correlation Matrix ('y' vs Weather):\n{correlation_matrix['y'].to_string()}")
                    else:
                         logger.warning("Correlation matrix computed, but 'y' column missing.")

                    # 6. Visualize
                    # --- Correlation heatmap ---
                    logger.info("Plotting correlation heatmap...")
                    plt.style.use('seaborn-v0_8-whitegrid')
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
                    ax_corr.set_title(f'Correlation Matrix: Demand (y) vs Weather Features (Sampled {n_sample_ids} IDs)')
                    plt.tight_layout()
                    save_plot(fig_corr, 'demand_vs_weather_correlation_heatmap.png', plots_dir)

                    # --- Scatter plots ---
                    scatter_cols_options = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature']
                    # Select only those present and used in correlation
                    scatter_cols = [col for col in scatter_cols_options if col in numeric_correlation_cols and col != 'y']
                    logger.info(f"Plotting scatter plots for Demand (y) vs {', '.join(scatter_cols)} (plot sample fraction: {plot_sample_frac:.1%})")

                    # Sample *after* potential NaNs were dropped for correlation for consistency, or sample original merged data? Let's sample original merged data for scatter.
                    if len(pdf_merged) * plot_sample_frac >= 1 and plot_sample_frac < 1.0:
                         plot_sample = pdf_merged.sample(frac=plot_sample_frac, random_state=random_state)
                    else:
                         plot_sample = pdf_merged # Use all data if sample frac is too small or 1.0

                    for col in scatter_cols:
                         # Ensure columns exist in the sample (might be dropped if all NaN)
                         if col not in plot_sample.columns or 'y' not in plot_sample.columns:
                              logger.warning(f"Skipping scatter plot for '{col}' as columns are missing in the plot sample.")
                              continue

                         # Drop NaNs only for the specific pair being plotted
                         plot_sample_pair = plot_sample[[col, 'y']].dropna()
                         if plot_sample_pair.empty:
                              logger.warning(f"Skipping scatter plot for '{col}' vs 'y' as there is no non-NaN data in the sample.")
                              continue

                         fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                         sns.scatterplot(data=plot_sample_pair, x=col, y='y', alpha=0.5, s=10, ax=ax_scatter)
                         ax_scatter.set_title(f'Demand (y) vs {col} (Sampled Data)')
                         ax_scatter.set_xlabel(col)
                         ax_scatter.set_ylabel('Electricity Demand (y) in kWh')
                         # Get correlation value safely
                         corr_val = correlation_matrix.loc['y', col] if ('y' in correlation_matrix.index and col in correlation_matrix.columns) else np.nan
                         if pd.notna(corr_val):
                            ax_scatter.text(0.05, 0.95, f'Correlation: {corr_val:.3f}', transform=ax_scatter.transAxes,
                                            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                         plt.tight_layout()
                         save_plot(fig_scatter, f'demand_vs_weather_{col}_scatterplot.png', plots_dir)
                except Exception as corr_viz_e:
                    logger.exception(f"Error during correlation calculation or visualization: {corr_viz_e}")
                    if 'fig_corr' in locals(): plt.close(fig_corr)
                    if 'fig_scatter' in locals(): plt.close(fig_scatter)


        logger.info("Demand vs Weather analysis complete.")

    except Exception as e:
        logger.exception(f"Error analyzing Demand vs Weather relationship: {e}")
    # No explicit cleanup needed due to context manager 