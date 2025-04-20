import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# 使用相对导入
from ..utils.eda_utils import dask_compute_context, plot_numerical_distribution, log_value_counts, save_plot

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
            with dask_compute_context(ddf_weather[relevant_cols].describe()) as persisted_stats:
                 if not persisted_stats:
                      logger.error("Failed to persist weather descriptive statistics.")
                 else:
                    try:
                        desc_stats_all = persisted_stats[0].compute()
                        if desc_stats_all is None or desc_stats_all.empty:
                            logger.warning("计算天气特征描述性统计信息失败或结果为空。")
                        else:
                            logger.info(f"Descriptive Statistics:\n{desc_stats_all.to_string()}")
                    except Exception as compute_desc_e:
                        logger.exception(f"Error computing weather descriptive statistics: {compute_desc_e}")

        except Exception as e:
            logger.exception(f"Error during weather feature descriptive statistics calculation: {e}")

    # Check for negative values
    precipitation_cols = ['precipitation', 'rain', 'snowfall']
    neg_check_cols = [col for col in precipitation_cols if col in relevant_cols]
    if neg_check_cols:
        logger.info(f"Checking columns {neg_check_cols} for negative values...")
        try:
            # Compute negative counts in parallel
            neg_counts_futures = [(col, (ddf_weather[col] < 0).sum()) for col in neg_check_cols]
            with dask_compute_context(*[f[1] for f in neg_counts_futures]) as persisted_counts:
                if not persisted_counts or len(persisted_counts) != len(neg_counts_futures):
                     logger.error("Failed to persist negative count computations.")
                else:
                    try:
                        neg_counts_computed = [p.compute() for p in persisted_counts]
                        for i, (col, _) in enumerate(neg_counts_futures):
                            negative_count = neg_counts_computed[i] if i < len(neg_counts_computed) else None
                            if negative_count is not None and negative_count > 0:
                                logger.warning(f"Column '{col}' detected {negative_count} negative values! Check data source or handle.")
                            elif negative_count is not None:
                                logger.info(f"Column '{col}' has no negative values detected.")
                            else:
                                 logger.warning(f"Could not compute negative count for column '{col}'.")
                    except Exception as compute_neg_e:
                         logger.exception(f"Error computing negative counts: {compute_neg_e}")

        except Exception as e:
            logger.exception(f"Error checking for negative values: {e}")


    # Plot column by column (if needed)
    if plot:
        logger.info(f"Starting to plot weather feature distributions (sample fraction: {plot_sample_frac:.1%}) ...")
        for col in relevant_cols:
            # Check if stats were successfully computed for this column
            if desc_stats_all is None or col not in desc_stats_all.columns:
                 logger.warning(f"Statistics for column '{col}' are unavailable, skipping plot.")
                 continue

            logger.info(f"Plotting column: {col}")
            try:
                # Sample and compute Pandas Series
                logger.debug(f"Sampling and computing Pandas Series for column '{col}'...")
                # Dropna before sampling
                col_sample_dask = ddf_weather[col].dropna().sample(frac=plot_sample_frac, random_state=random_state)
                if col_sample_dask.npartitions == 0:
                     logger.warning(f"Sampling result for column '{col}' has 0 partitions, likely empty. Skipping plot.")
                     continue

                with dask_compute_context(col_sample_dask) as persisted_sample:
                     if not persisted_sample:
                          logger.error(f"Failed to persist sample for column '{col}'. Skipping plot.")
                          continue
                     try:
                        col_sample_pd = persisted_sample[0].compute()
                        logger.debug(f"Column '{col}' sampling complete, sample size: {len(col_sample_pd):,}")
                     except Exception as compute_sample_e:
                        logger.exception(f"Error computing sample for column '{col}': {compute_sample_e}")
                        continue # Skip plot if compute fails

                if col_sample_pd is None or col_sample_pd.empty:
                    logger.warning(f"Sample result for column '{col}' is empty or computation failed, skipping plot.")
                    continue

                # Use helper function to plot
                plot_numerical_distribution(col_sample_pd, col,
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
            counts_dask = ddf_weather[col].value_counts()
            with dask_compute_context(counts_dask) as persisted_counts:
                 if not persisted_counts:
                     logger.error(f"Failed to persist value counts for column '{col}'.")
                     continue
                 try:
                     counts_pd = persisted_counts[0].compute() # Get Pandas Series
                 except Exception as compute_counts_e:
                     logger.exception(f"Error computing value counts for column '{col}': {compute_counts_e}")
                     continue # Skip if compute fails


            if counts_pd is None or counts_pd.empty:
                 logger.warning(f"Value counts for column '{col}' are empty or computation failed.")
                 continue

            # Log value counts (using Pandas Series, passing is_already_counts=True)
            log_value_counts(counts_pd, col, top_n=top_n, is_already_counts=True) # <-- Use helper

            # Plot (if needed)
            if plot and not counts_pd.empty:
                logger.info(f"Plotting column: {col}")
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    data_to_plot = counts_pd.head(top_n)
                    num_unique = len(counts_pd)
                    title = f'Weather Top {top_n} {col} Distribution'
                    if num_unique > top_n:
                        title += " (Others not shown)"

                    # Ensure index is treated as string for plotting
                    plot_index = data_to_plot.index.astype(str)
                    order = plot_index # Use count order (default) or try numeric sort if appropriate

                    # Optional: Try sorting by numeric value of code if possible
                    try:
                        # Sort original index numerically, then convert to string for ordering
                        numeric_index_order = data_to_plot.index.astype(float).sort_values().astype(str)
                        # Reindex plot_index based on numeric order found
                        order = [idx for idx in numeric_index_order if idx in plot_index]
                        logger.debug(f"Using numeric sort order for '{col}' plot.")
                    except (ValueError, TypeError):
                        logger.debug(f"Cannot sort '{col}' index numerically, using default order.")
                        order = plot_index.tolist() # Fallback to default order as list

                    sns.barplot(x=plot_index, y=data_to_plot.values, ax=ax, palette="viridis", order=order)
                    ax.set_title(title)
                    ax.set_xlabel(f'{col} (WMO Code)' if col == 'weather_code' else col)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    save_plot(fig, f'weather_distribution_{col}.png', plots_dir)
                except Exception as plot_e:
                    logger.exception(f"Error plotting categorical distribution for '{col}': {plot_e}")
                    plt.close(fig) # Ensure plot is closed


        except Exception as e:
            logger.exception(f"Error analyzing column '{col}': {e}")

    logger.info("Weather categorical feature analysis complete.") 