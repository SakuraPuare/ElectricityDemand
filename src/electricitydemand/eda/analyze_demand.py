import traceback
from pathlib import Path  # Import Path

import matplotlib.pyplot as plt
import numpy as np  # 添加 numpy 导入，用于 log1p
import pandas as pd
import seaborn as sns  # 添加 seaborn 导入，用于绘图
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
    analyze_timestamp_frequency_pandas,
    log_value_counts,
    plot_numerical_distribution,
    sample_spark_ids_and_collect,
    save_plot,
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
    logger.debug(f"分析的列：{', '.join(columns_to_analyze)}")

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
        logger.exception(f"使用 Spark 计算描述性统计时出错：{e}")

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
            logger.exception(f"使用 Spark 检查负值时出错：{e}")

    # 4. 绘图 (Spark 抽样 -> Pandas 绘图)
    if plot and numerical_cols:
        logger.info(f"开始绘制 Weather 特征分布图 (抽样比例：{plot_sample_frac:.1%})...")
        for col in numerical_cols:
            logger.info(f"绘制列：{col}")
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
                logger.debug(f"列 '{col}' 抽样完成，样本大小：{len(col_sample_pd):,}")

                if col_sample_pd is None or col_sample_pd.empty:
                    logger.warning(f"列 '{col}' 的抽样结果为空或收集失败，跳过绘图。")
                    continue

                # 使用辅助函数绘图 (接收 Pandas Series)
                plot_numerical_distribution(col_sample_pd, col,
                                            f'weather_distribution_{col}', plots_dir,
                                            title=f"Weather {col}", kde=True,  # Changed title_prefix to title
                                            showfliers=False)  # Hide outliers for potentially wide ranges
            except Exception as e:
                logger.exception(f"绘制列 '{col}' 的分布图时出错：{e}")

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
            logger.info(f"--- 分析列：{col} ---")
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
                logger.info(f"'{col}' 分布图已保存：{plot_filename}")

            except Exception as e:
                logger.exception(f"分析或绘制 Weather 分类列 '{col}' 时出错：{e}")
        else:
            logger.warning(f"在 Weather 数据中未找到分类列：{col}")
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

        logger.info(f"计算以下列的相关性：{', '.join(available_numerical_cols)}")

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
        logger.exception(f"分析 Weather 特征相关性时出错：{e}")
    logger.info("--- 完成 Weather 数值特征相关性分析 ---")


# --- 新增：analyze_weather_timeseries_sample ---
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


def analyze_demand_y_distribution(sdf_demand: DataFrame, sample_frac=0.005, random_state=42, plots_dir: Path = None):  # Use Path
    """使用 Spark 分析 Demand 数据 'y' 列的分布 (基于 Spark 计算和抽样)。"""
    logger.info(
        f"--- 开始使用 Spark 分析 Demand 'y' 列分布 (抽样比例：{sample_frac:.3%}) ---")
    if 'y' not in sdf_demand.columns:
        logger.error("输入 Spark DataFrame 缺少 'y' 列。")
        return None
    # 确保 'y' 是数值类型
    if not isinstance(sdf_demand.schema['y'].dataType, NumericType):
        logger.error(
            f"'y' 列不是数值类型 ({sdf_demand.schema['y'].dataType})。无法进行分析。")
        return None

    y_sample_pd = None
    try:
        # --- 1. Spark 计算描述性统计 ---
        logger.info("使用 Spark 计算 'y' 列的描述性统计...")
        # 使用 .agg() 获得更结构化的统计结果
        stats_agg = sdf_demand.agg(
            F.count(F.col('y')).alias("count"),
            F.mean(F.col('y')).alias("mean"),
            F.stddev(F.col('y')).alias("stddev"),
            F.min(F.col('y')).alias("min"),
            F.expr("percentile_approx(y, 0.25)").alias("25%"),
            F.expr("percentile_approx(y, 0.5)").alias("50%"),
            F.expr("percentile_approx(y, 0.75)").alias("75%"),
            F.max(F.col('y')).alias("max")
        ).first()  # .first() returns a Row object

        if stats_agg is None:
            logger.error("无法计算 'y' 列的描述性统计。")
            return None

        # 将 Row 转换为更易读的格式
        desc_stats_spark = pd.Series(stats_agg.asDict())
        logger.info(
            f"'y' 列 (Spark 全量统计) 描述性统计:\n{desc_stats_spark.to_string()}")

        # 获取总行数 (用于计算百分比)
        total_rows_df = sdf_demand.count()  # Count all rows in the original DataFrame
        if total_rows_df == 0:
            logger.warning("原始 DataFrame 为空，无法进行进一步分析。")
            return None

        # Count of non-null 'y' values
        valid_y_count = stats_agg['count'] if stats_agg['count'] is not None else 0
        null_count = total_rows_df - valid_y_count
        logger.info(f"全量数据中 'y' 的总行数：{total_rows_df:,}")
        logger.info(f"全量数据中 'y' 非空数量：{valid_y_count:,}")
        if total_rows_df > 0:
            null_perc = (null_count / total_rows_df) * 100
            logger.info(f"全量数据中 'y' 缺失数量：{null_count:,} ({null_perc:.2f}%)")
        else:
            logger.info(f"全量数据中 'y' 缺失数量：0")

        # --- 2. Spark 计算非正值 ---
        logger.info("使用 Spark 检查 'y' 列中的非正值 (<= 0)...")
        # 包含 NaN 的情况也需要考虑，filter(y <= 0) 会自动忽略 null
        non_positive_count = sdf_demand.filter(F.col('y') <= 0).count()
        # 计算缺失值数量
        null_count = sdf_demand.filter(F.col('y').isNull()).count()
        logger.info(f"全量数据中 'y' 缺失的数量：{null_count:,}")
        if valid_y_count > 0:
            non_positive_perc = (non_positive_count / valid_y_count) * 100
            logger.info(
                f"全量数据中 'y' <= 0 的数量 (在非缺失值中): {non_positive_count:,} ({non_positive_perc:.2f}%)")
        else:
            logger.info("没有非空 'y' 值，无法计算非正值百分比。")

        # --- 3. Spark 抽样并收集到 Pandas (用于绘图) ---
        if plots_dir:
            plots_dir = Path(plots_dir)  # Ensure Path
            logger.info(f"对 'y' 列进行 Spark 抽样 (比例：{sample_frac:.1%}) 以便绘图...")
            # 先过滤掉 NaN/Null 值再抽样
            y_col_sdf = sdf_demand.select('y').dropna()
            # 使用 sample 方法
            y_sample_sdf = y_col_sdf.sample(
                withReplacement=False, fraction=sample_frac, seed=random_state)

            logger.info("将抽样结果收集到 Pandas Series...")
            try:
                # Explicitly handle potential empty DataFrame after sampling
                if y_sample_sdf.first() is None:
                    logger.warning("Spark 抽样结果为空，无法收集到 Pandas。")
                    y_sample_pd = None
                else:
                    y_sample_pd = y_sample_sdf.toPandas()['y']
                    if y_sample_pd is None or y_sample_pd.empty:
                        logger.warning("Spark 抽样结果收集到 Pandas 后为空。")
                        y_sample_pd = None
                    else:
                        num_samples = len(y_sample_pd)
                        logger.info(
                            f"抽样并收集完成，得到 {num_samples:,} 个非空样本 (Pandas)。")

                        # --- 4. 调用绘图函数 ---
                        plot_demand_y_distribution(
                            y_sample_pd, plots_dir, sample_frac)

            except Exception as collect_e:
                logger.exception(f"收集 Spark 抽样数据到 Pandas 或绘图时出错：{collect_e}")
                y_sample_pd = None  # 出错则返回 None
        else:
            logger.info("未提供 plots_dir，跳过 'y' 列分布的抽样和绘图。")

        # 函数现在主要执行分析和记录，绘图是可选的
        # 返回 Pandas 样本（如果生成了）或 None
        # return y_sample_pd # 或者可以不返回任何东西，因为分析结果已记录
        return  # 分析结果已记录，绘图已完成（如果提供了目录）

    except Exception as e:
        logger.exception(f"使用 Spark 分析 Demand 'y' 列分布时发生错误：{e}")
        # return None


# 定义 plot_demand_y_distribution 函数，不再重复导入
def plot_demand_y_distribution(
    y_sample: pd.Series,
    plots_dir: Path = Path("plots"),  # 直接使用 plots_dir 路径
    sample_frac: float = 0.01,
    col_name: str = 'y'
):
    """
    绘制电力需求 'y' 列的分布图 (直方图和箱线图)。

    Args:
        y_sample (pd.Series): 抽样的 'y' 数据。
        plots_dir (Path): 保存图表的目录。
        sample_frac (float): 用于图例的抽样比例。
        col_name (str): 列名 (通常是 'y')。
    """
    logger.info(f"绘制 'y' 分布图 (样本比例：{sample_frac*100:.1f}%)...")
    if y_sample.empty:
        logger.warning("抽样数据为空，无法绘制 'y' 分布图。")
        return

    try:
        # Plot on original scale
        plot_numerical_distribution(y_sample.dropna(), col_name,
                                    f"Demand Value ({col_name}) Distribution (Original Scale, Sample)",
                                    plots_dir, kde=True)

        # Plot on log scale (log1p)
        y_log_sample = np.log1p(y_sample.dropna())
        # Ensure log sample isn't all NaN/inf
        if y_log_sample.notna().any():
            plot_numerical_distribution(y_log_sample, f'{col_name} (log1p)',
                                        f"Demand Value ({col_name}) Distribution (Log1p Scale, Sample)",
                                        plots_dir, kde=True)  # Removed log_scale=True
        else:
             logger.warning("Log-transformed sample contains only NaN/inf, skipping log scale plot.")


    except Exception as e:
        logger.error(f"绘制 'y' 分布图时出错：{e}")
        logger.debug(traceback.format_exc())  # Log full traceback for debugging


# --- 迁移后的 analyze_demand_timeseries_sample ---
def analyze_demand_timeseries_sample(sdf_demand: DataFrame, n_samples=5, plots_dir: Path = None, random_state=42):  # Use Path
    """使用 Spark 抽取 n_samples 个 unique_id 的完整时间序列数据到 Pandas，然后绘制图形并分析频率。"""
    logger.info(
        f"--- Starting analysis of Demand time series (sampling {n_samples} unique_id) ---")
    if plots_dir is None:
        logger.error("plots_dir not provided, cannot save time series plots.")
        return
    plots_dir = Path(plots_dir)  # Ensure Path
    plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    required_cols = ['unique_id', 'timestamp', 'y']
    if not all(col in sdf_demand.columns for col in required_cols):
        logger.error(
            f"Input Spark DataFrame missing required columns ({', '.join(required_cols)}).")
        return

    # --- Use the new sampling and collection function ---
    spark = SparkSession.getActiveSession()  # Get active Spark session
    if not spark:
        logger.error("Could not get active Spark session.")
        return

    pdf_sample = sample_spark_ids_and_collect(
        sdf=sdf_demand,
        id_col='unique_id',
        n_samples=n_samples,
        random_state=random_state,
        select_cols=required_cols,
        spark=spark,
        timestamp_col='timestamp'  # Specify timestamp column
    )

    if pdf_sample is None or pdf_sample.empty:
        logger.error(
            "Failed to obtain sampled and collected Pandas DataFrame for demand time series. Aborting analysis.")
        return
    # pdf_sample is now sorted and timestamp is datetime

    # --- 在 Pandas 上进行绘图和频率分析 ---
    logger.info(
        "Starting to plot time series for each sample (in Pandas) and analyze frequency...")
    plt.style.use('seaborn-v0_8-whitegrid')

    grouped_data = pdf_sample.groupby('unique_id')  # 按 ID 分组以便处理

    for unique_id, df_id in tqdm(grouped_data, desc="Processing demand samples (Pandas)"):
        # for unique_id in tqdm(sampled_ids, desc="Processing demand samples (Pandas)"): # Alternative if groupby is slow
        # df_id = pdf_sample[pdf_sample['unique_id'] == unique_id] # Needed if not using groupby iterator
        if df_id.empty:
            # This shouldn't happen if sample_spark_ids_and_collect worked correctly
            logger.warning(
                f"No data found for unique_id '{unique_id}' in the Pandas sample (post-processing). Skipping.")
            continue

        # --- Plot time series for this ID (line chart) ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_id['timestamp'], df_id['y'], 
                marker='.', linestyle='-', markersize=2, 
                alpha=0.8, linewidth=1)
        
        ax.set_title(f'Demand Time Series for ID: {unique_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Demand (y)')
        ax.grid(True, alpha=0.3)
        
        # Improve x-axis date formatting - automatic date handling with smart rotation
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"timeseries_sample_{unique_id}.png"
        save_plot(fig, plot_filename, plots_dir)

        # --- Also analyze the timestamp frequency ---
        analyze_timestamp_frequency_pandas(
            df_id, unique_id, 'unique_id', timestamp_col='timestamp')

    logger.info("Demand time series sample analysis complete.")
