import math
from pathlib import Path  # Use pathlib for paths

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pyspark.ml.feature import VectorAssembler  # For Spark correlation
from pyspark.ml.stat import Correlation  # For Spark correlation

# 移除 Dask 导入
# import dask.dataframe as dd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, NumericType, TimestampType
from tqdm import tqdm  # For timeseries sample progress

# 使用相对导入
from ..utils.eda_utils import (
    log_value_counts,
    plot_categorical_distribution,
    plot_numerical_distribution,
    save_plot,
    analyze_timestamp_frequency_pandas,
    sample_spark_ids_and_collect
)

# 移除 dask_compute_context


# Changed sample frac default
def analyze_weather_numerical(sdf_weather: DataFrame, columns_to_analyze=None, plot_sample_frac=0.05, plots_dir=None, random_state=42):
    """分析 Weather Spark DataFrame 中指定数值列的分布并记录/绘图。"""
    if sdf_weather is None:
        logger.warning("输入的 Weather Spark DataFrame 为空，跳过数值特征分析。")
        return
    plot = plots_dir is not None
    if plot:
        plots_dir = Path(plots_dir)  # Convert to Path object
        plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    else:
        logger.warning("未提供 plots_dir，将仅记录统计信息，不绘制 Weather 数值特征图。")

    if columns_to_analyze is None:
        # 根据数据集 README 扩展列
        columns_to_analyze = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth",
            "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
            "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
            "vapour_pressure_deficit", "wind_speed_10m", "wind_direction_10m",
            "wind_gusts_10m",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
            "sunshine_duration", "shortwave_radiation", "direct_radiation",
            "diffuse_radiation", "direct_normal_irradiance", "terrestrial_radiation"
        ]
    logger.info(f"--- 开始分析 Weather 数值特征分布 (Spark) ---")
    logger.debug(f"分析的列: {', '.join(columns_to_analyze)}")

    # 1. 过滤出实际存在且为数值类型的列
    numerical_cols = []
    datetime_cols = []
    other_cols = []
    schema = sdf_weather.schema
    for col in columns_to_analyze:
        if col in schema.names:
            if isinstance(schema[col].dataType, NumericType):
                numerical_cols.append(col)
            # Check for time types
            elif isinstance(schema[col].dataType, (TimestampType, DateType)):
                datetime_cols.append(col)
            else:
                other_cols.append(col)
                logger.warning(
                    f"列 '{col}' 类型为 {schema[col].dataType}，将从数值分析中排除。")
        else:
            logger.warning(f"列 '{col}' 不在 Weather DataFrame 中，跳过。")

    if not numerical_cols:
        logger.warning("在指定列中未找到可分析的数值列。")
        return

    # 2. 计算描述性统计 (Spark)
    desc_stats_pd = None
    logger.info("使用 Spark 计算数值特征的描述性统计...")
    try:
        # Spark describe() 返回字符串，改用 agg()
        agg_exprs = [F.count(F.col(c)).alias(
            f"{c}_count") for c in numerical_cols]
        agg_exprs.extend([F.mean(F.col(c)).alias(
            f"{c}_mean") for c in numerical_cols])
        agg_exprs.extend([F.stddev(F.col(c)).alias(
            f"{c}_stddev") for c in numerical_cols])
        agg_exprs.extend([F.min(F.col(c)).alias(
            f"{c}_min") for c in numerical_cols])
        agg_exprs.extend([F.expr(f"percentile_approx({c}, 0.25)").alias(
            f"{c}_25%") for c in numerical_cols])
        agg_exprs.extend([F.expr(f"percentile_approx({c}, 0.5)").alias(
            f"{c}_50%") for c in numerical_cols])
        agg_exprs.extend([F.expr(f"percentile_approx({c}, 0.75)").alias(
            f"{c}_75%") for c in numerical_cols])
        agg_exprs.extend([F.max(F.col(c)).alias(
            f"{c}_max") for c in numerical_cols])

        stats_row = sdf_weather.agg(*agg_exprs).first()

        if stats_row:
            # 将 Row 结果重新组织为类似 describe() 的 Pandas DataFrame
            stats_dict = stats_row.asDict()
            desc_data = {}
            for c in numerical_cols:
                desc_data[c] = {
                    "count": stats_dict.get(f"{c}_count"),
                    "mean": stats_dict.get(f"{c}_mean"),
                    "stddev": stats_dict.get(f"{c}_stddev"),
                    "min": stats_dict.get(f"{c}_min"),
                    "25%": stats_dict.get(f"{c}_25%"),
                    "50%": stats_dict.get(f"{c}_50%"),
                    "75%": stats_dict.get(f"{c}_75%"),
                    "max": stats_dict.get(f"{c}_max"),
                }
            desc_stats_pd = pd.DataFrame(desc_data)
            logger.info(f"Spark 计算的描述性统计:\n{desc_stats_pd.to_string()}")
        else:
            logger.error("未能计算描述性统计。")

    except Exception as e:
        logger.exception(f"使用 Spark 计算描述性统计时出错: {e}")

    # 3. 检查负值 (Spark)
    precipitation_cols = ['precipitation', 'rain', 'snowfall', 'snow_depth', 'et0_fao_evapotranspiration',
                          'sunshine_duration', 'direct_radiation', 'diffuse_radiation']  # 这些理论上不应为负
    neg_check_cols = [
        col for col in precipitation_cols if col in numerical_cols]  # 只检查存在的数值列
    if neg_check_cols:
        logger.info(f"使用 Spark 检查列 {neg_check_cols} 中的负值...")
        try:
            for col in neg_check_cols:
                negative_count = sdf_weather.filter(
                    F.col(col) < 0).count()  # Action
                if negative_count > 0:
                    logger.warning(
                        f"列 '{col}' 检测到 {negative_count:,} 个负值！请检查数据源或处理。")
                else:
                    logger.info(f"列 '{col}' 未检测到负值。")
        except Exception as e:
            logger.exception(f"使用 Spark 检查负值时出错: {e}")

    # 4. 绘图 (Spark 抽样 -> Pandas 绘图)
    if plot and numerical_cols:
        logger.info(f"开始绘制 Weather 特征分布图 (抽样比例: {plot_sample_frac:.1%})...")
        for col in numerical_cols:
            logger.info(f"绘制列: {col}")
            try:
                # Spark 抽样
                logger.debug(f"对列 '{col}' 进行 Spark 抽样...")
                # 先过滤 NaN/Null，然后抽样
                col_sample_sdf = sdf_weather.select(col).dropna().sample(
                    False, plot_sample_frac, seed=random_state)

                # 收集到 Pandas
                logger.debug(f"将列 '{col}' 的抽样结果收集到 Pandas...")
                col_sample_pd = col_sample_sdf.toPandas()[
                    col]  # 获取 Pandas Series
                logger.debug(f"列 '{col}' 抽样完成，样本大小: {len(col_sample_pd):,}")

                if col_sample_pd is None or col_sample_pd.empty:
                    logger.warning(f"列 '{col}' 的抽样结果为空或收集失败，跳过绘图。")
                    continue

                # 使用辅助函数绘图 (接收 Pandas Series)
                plot_numerical_distribution(col_sample_pd, col,
                                            f'weather_distribution_{col}', plots_dir,
                                            title_prefix="Weather ", kde=True,  # Use KDE for weather features
                                            showfliers=False)  # Hide outliers for potentially wide ranges
            except Exception as e:
                logger.exception(f"绘制列 '{col}' 的分布图时出错: {e}")

    logger.info("Weather 数值特征分析完成。")


def analyze_weather_categorical(sdf_weather: DataFrame, plots_dir: str):
    """
    Analyzes categorical features (like is_day, weather_code) in the weather Spark DataFrame.

    Args:
        sdf_weather: Spark DataFrame containing weather data.
        plots_dir: Directory to save plots.
    """
    logger.info("--- 分析 Weather 分类特征 (Spark) ---")
    # Add 'weather_code' to the list
    categorical_cols = ['is_day', 'weather_code']

    for col in categorical_cols:
        if col in sdf_weather.columns:
            logger.info(f"--- 分析列: {col} ---")
            try:
                # Get value counts using Spark
                value_counts_sdf = sdf_weather.groupBy(
                    col).count().orderBy(F.desc('count'))
                value_counts_pd = value_counts_sdf.toPandas()  # Collect results

                logger.info(
                    f"'{col}' 值分布 (Top 20):\n{value_counts_pd.head(20).to_string(index=False)}")
                if value_counts_pd.shape[0] > 20:
                    logger.info(f"... (共 {value_counts_pd.shape[0]} 个唯一值)")

                # Plotting (if feasible number of categories)
                # For weather_code, there might be many codes, plot top N
                max_categories_to_plot = 20
                # Plot all if few, or always for is_day
                if value_counts_pd.shape[0] <= max_categories_to_plot or col == 'is_day':
                    plot_data = value_counts_pd
                    title = f'{col} 分布'
                else:  # Plot Top N + Other for weather_code if too many
                    top_categories = value_counts_pd.nlargest(
                        max_categories_to_plot, 'count')
                    other_count = value_counts_pd.iloc[max_categories_to_plot:]['count'].sum(
                    )
                    if other_count > 0:
                        # Create a new row for 'Other'
                        other_row = pd.DataFrame(
                            {col: ['Other'], 'count': [other_count]})
                        # Use pandas.concat
                        plot_data = pd.concat(
                            [top_categories, other_row], ignore_index=True)

                    title = f'{col} 分布 (Top {max_categories_to_plot} & Other)'

                plt.figure(figsize=(12, 6))
                # Ensure the column used for labels is treated as string for plotting
                sns.barplot(x=plot_data[col].astype(
                    str), y=plot_data['count'], palette='viridis', order=plot_data[col].astype(str).tolist())
                plt.title(title)
                plt.xlabel(col)
                plt.ylabel('数量')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_filename = plots_dir / f"weather_dist_{col}.png"
                plt.savefig(plot_filename)
                plt.close()
                logger.info(f"'{col}' 分布图已保存: {plot_filename}")

            except Exception as e:
                logger.exception(f"分析或绘制 Weather 分类列 '{col}' 时出错: {e}")
        else:
            logger.warning(f"在 Weather 数据中未找到分类列: {col}")
    logger.info("--- 完成 Weather 分类特征分析 ---")


def analyze_weather_correlation(sdf_weather: DataFrame, plots_dir: str):
    """
    Calculates and plots the correlation matrix for numerical weather features using Spark.

    Args:
        sdf_weather: Spark DataFrame containing weather data.
        plots_dir: Directory to save the plot.
    """
    logger.info("--- 分析 Weather 数值特征相关性 (Spark) ---")
    try:
        # Select only numerical columns suitable for correlation
        # Exclude IDs, categorical codes, and potentially redundant features if needed
        # Also exclude columns that might be all null or constant after filtering
        numerical_cols = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "snow_depth", "pressure_msl", "surface_pressure", "cloud_cover",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "et0_fao_evapotranspiration", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
            "sunshine_duration", "shortwave_radiation", "direct_radiation",
            "diffuse_radiation", "direct_normal_irradiance", "terrestrial_radiation"
        ]

        # Filter out columns that are not present in the DataFrame
        available_numerical_cols = [
            c for c in numerical_cols if c in sdf_weather.columns]

        if len(available_numerical_cols) < 2:
            logger.warning("数值列不足 (<2)，无法计算相关性矩阵。")
            return

        logger.info(f"计算以下列的相关性: {', '.join(available_numerical_cols)}")

        # Assemble features into a vector column needed for Spark's Correlation
        # skip rows with nulls in any column
        assembler = VectorAssembler(
            inputCols=available_numerical_cols, outputCol="features", handleInvalid="skip")
        sdf_vector = assembler.transform(sdf_weather).select("features")

        # Calculate Pearson correlation matrix using Spark MLlib
        correlation_matrix_spark = Correlation.corr(
            sdf_vector, "features").head()
        if correlation_matrix_spark is None:
            logger.error("无法计算相关性矩阵 (Spark 返回 None)。可能是因为数据不足或全为无效值。")
            return

        # Get the dense matrix
        corr_matrix_dense = correlation_matrix_spark[0].toArray()

        # Convert to Pandas DataFrame for plotting
        corr_matrix_pd = pd.DataFrame(
            corr_matrix_dense, index=available_numerical_cols, columns=available_numerical_cols)

        # Plot heatmap
        plt.figure(figsize=(18, 15))  # Adjust size as needed
        # annot=False for cleaner look with many features
        sns.heatmap(corr_matrix_pd, cmap='coolwarm',
                    annot=False, fmt=".2f", linewidths=.5)
        plt.title('Weather Numerical Feature Correlation Matrix', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plot_filename = plots_dir / "weather_correlation_matrix.png"
        plt.savefig(plot_filename)
        plt.close()
        logger.info(
            f"Weather Feature Correlation Matrix saved: {plot_filename}")

        # Log strong correlations (absolute value > threshold)
        threshold = 0.8
        strong_corrs = corr_matrix_pd.unstack().sort_values(
            ascending=False).drop_duplicates()
        strong_corrs = strong_corrs[abs(strong_corrs) > threshold]
        # Remove self-correlation
        strong_corrs = strong_corrs[strong_corrs != 1.0]

        if not strong_corrs.empty:
            logger.info(f"--- Strong Correlations (|corr| > {threshold}) ---")
            logger.info(strong_corrs.to_string())
        else:
            logger.info(f"未发现绝对值大于 {threshold} 的强相关性。")

    except Exception as e:
        logger.exception(f"分析 Weather 特征相关性时出错: {e}")
    logger.info("--- 完成 Weather 数值特征相关性分析 ---")


# --- 新增: analyze_weather_timeseries_sample ---
def analyze_weather_timeseries_sample(sdf_weather: DataFrame, n_samples=5, plots_dir=None, random_state=42):
    """
    使用 Spark 抽取 n_samples 个 location_id 的天气时间序列数据到 Pandas，
    然后分析时间戳频率。
    """
    logger.info(
        f"--- Starting analysis of Weather time series frequency (sampling {n_samples} location_id) ---")
    # plots_dir is optional here, only frequency is analyzed
    # Only need these for frequency
    required_cols = ['location_id', 'timestamp']
    if not all(col in sdf_weather.columns for col in required_cols):
        logger.error(
            f"Input Weather Spark DataFrame missing required columns ({', '.join(required_cols)}). Cannot analyze frequency.")
        return

    # --- Use the new sampling and collection function ---
    spark = SparkSession.getActiveSession()  # Get active Spark session
    if not spark:
        logger.error("Could not get active Spark session.")
        return

    pdf_sample = sample_spark_ids_and_collect(
        sdf=sdf_weather,
        id_col='location_id',
        n_samples=n_samples,
        random_state=random_state,
        select_cols=required_cols,  # Only select needed columns
        spark=spark,
        timestamp_col='timestamp'  # Specify timestamp column
    )

    if pdf_sample is None or pdf_sample.empty:
        logger.error(
            "Failed to obtain sampled and collected Pandas DataFrame for weather frequency analysis. Aborting.")
        return
    # pdf_sample is now sorted and timestamp is datetime

    # --- 在 Pandas 上进行频率分析 ---
    logger.info(
        "Starting to analyze weather timestamp frequency for each sample (in Pandas)...")
    grouped_data = pdf_sample.groupby('location_id')

    sampled_ids = pdf_sample['location_id'].unique()  # Get actual sampled IDs

    for location_id, df_id in tqdm(grouped_data, desc="Analyzing weather frequency (Pandas)"):
        # for location_id in tqdm(sampled_ids, desc="Analyzing weather frequency (Pandas)"): # Alternative
        # df_id = pdf_sample[pdf_sample['location_id'] == location_id] # Needed if not using groupby
        if df_id.empty:
            # This shouldn't happen if sample_spark_ids_and_collect worked correctly
            logger.warning(
                f"No weather data found for location_id '{location_id}' in the Pandas sample (post-processing), skipping frequency analysis.")
            continue

        # --- Use the existing frequency analysis function ---
        analyze_timestamp_frequency_pandas(
            df_id, location_id, 'location_id', timestamp_col='timestamp')

    logger.info(
        "Weather time series frequency sample analysis complete.")
