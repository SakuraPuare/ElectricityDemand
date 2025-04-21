import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql import functions as F
from pathlib import Path  # Import Path
from pyspark.sql.utils import AnalysisException  # Import specific exception

# 设置绘图风格
sns.set_theme(style="whitegrid")


def analyze_timestamp_consistency(sdf_demand: DataFrame, sdf_weather: DataFrame):
    """
    分析 Demand 和 Weather 数据的时间戳频率和一致性 (Spark版，日志保持中文)。

    Args:
        sdf_demand (DataFrame): Demand Spark DataFrame, needs 'unique_id' and 'timestamp'.
        sdf_weather (DataFrame): Weather Spark DataFrame, needs 'location_id' and 'timestamp'.
    """
    logger.info("--- 开始分析时间戳一致性 (Spark) ---")
    sample_frac_freq = 0.001  # Sample fraction for frequency analysis

    # --- Demand 时间戳频率分析 ---
    logger.info("分析 Demand 数据时间戳间隔...")
    if sdf_demand and isinstance(sdf_demand, DataFrame) and 'unique_id' in sdf_demand.columns and 'timestamp' in sdf_demand.columns:
        try:
            window_spec = Window.partitionBy("unique_id").orderBy("timestamp")
            sdf_demand_diff = sdf_demand.withColumn(
                "prev_timestamp", F.lag("timestamp", 1).over(window_spec)
            )
            sdf_demand_diff = sdf_demand_diff.withColumn(
                "diff_seconds",
                F.when(F.col("prev_timestamp").isNull(), None)
                .otherwise(F.col("timestamp").cast("long") - F.col("prev_timestamp").cast("long"))
                # Remove null diffs early
            ).filter(F.col("diff_seconds").isNotNull())

            logger.info(f"对 Demand 时间差进行抽样 (比例: {sample_frac_freq}) 以统计频率...")
            # Sample *after* calculating diffs
            common_intervals_demand_sdf = sdf_demand_diff \
                .select("diff_seconds") \
                .sample(withReplacement=False, fraction=sample_frac_freq)

            # Check if sample is empty before aggregation
            if common_intervals_demand_sdf.first() is None:
                logger.warning("Demand 时间差抽样后为空，无法统计频率。")
            else:
                common_intervals_demand = common_intervals_demand_sdf \
                    .groupBy("diff_seconds") \
                    .count() \
                    .orderBy(F.desc("count")) \
                    .limit(10) \
                    .toPandas()

                if not common_intervals_demand.empty:
                    logger.info("Demand 数据中最常见的 10 个时间间隔 (秒):")
                    logger.info(
                        f"\n{common_intervals_demand.to_string(index=False)}")
                    try:
                        common_intervals_demand['interval'] = pd.to_timedelta(
                            common_intervals_demand['diff_seconds'], unit='s', errors='coerce')
                        logger.info("Demand 数据中最常见的 10 个时间间隔 (易读格式):")
                        logger.info(
                            f"\n{common_intervals_demand[['interval', 'count']].to_string(index=False)}")
                    except Exception as td_e:
                        logger.warning(
                            f"转换 Demand 时间间隔为 Timedelta 时出错: {td_e}")
                else:
                    logger.warning("无法计算 Demand 的常见时间间隔 (聚合或转换 Pandas 后为空)。")

        except AnalysisException as ae:
            logger.error(f"分析 Demand 时间戳间隔时发生 Spark 分析错误: {ae}")
        except Exception as e:
            logger.exception(f"分析 Demand 时间戳间隔时发生未知错误: {e}")
    else:
        logger.warning("Demand DataFrame 无效或缺少必需列，跳过时间戳间隔分析。")

    # --- Weather 时间戳频率分析 ---
    logger.info("分析 Weather 数据时间戳间隔...")
    if sdf_weather and isinstance(sdf_weather, DataFrame) and 'location_id' in sdf_weather.columns and 'timestamp' in sdf_weather.columns:
        try:
            window_spec_weather = Window.partitionBy(
                "location_id").orderBy("timestamp")
            sdf_weather_diff = sdf_weather.withColumn(
                "prev_timestamp", F.lag(
                    "timestamp", 1).over(window_spec_weather)
            )
            sdf_weather_diff = sdf_weather_diff.withColumn(
                "diff_seconds",
                F.when(F.col("prev_timestamp").isNull(), None)
                .otherwise(F.col("timestamp").cast("long") - F.col("prev_timestamp").cast("long"))
            ).filter(F.col("diff_seconds").isNotNull())

            logger.info(f"对 Weather 时间差进行抽样 (比例: {sample_frac_freq}) 以统计频率...")
            common_intervals_weather_sdf = sdf_weather_diff \
                .select("diff_seconds") \
                .sample(withReplacement=False, fraction=sample_frac_freq)

            if common_intervals_weather_sdf.first() is None:
                logger.warning("Weather 时间差抽样后为空，无法统计频率。")
            else:
                common_intervals_weather = common_intervals_weather_sdf \
                    .groupBy("diff_seconds") \
                    .count() \
                    .orderBy(F.desc("count")) \
                    .limit(10) \
                    .toPandas()

                if not common_intervals_weather.empty:
                    logger.info("Weather 数据中最常见的 10 个时间间隔 (秒):")
                    logger.info(
                        f"\n{common_intervals_weather.to_string(index=False)}")
                    try:
                        common_intervals_weather['interval'] = pd.to_timedelta(
                            common_intervals_weather['diff_seconds'], unit='s', errors='coerce')
                        logger.info("Weather 数据中最常见的 10 个时间间隔 (易读格式):")
                        logger.info(
                            f"\n{common_intervals_weather[['interval', 'count']].to_string(index=False)}")

                        # 检查主要频率是否为 1 小时 (3600 秒)
                        if not common_intervals_weather.empty and common_intervals_weather.iloc[0]['diff_seconds'] == 3600:
                            logger.info("Weather 数据的主要时间间隔确认为 1 小时。")
                        else:
                            logger.warning(
                                f"Weather 数据的主要时间间隔似乎不是 1 小时 (最常见的是 {common_intervals_weather.iloc[0]['interval']})。")
                    except Exception as td_e:
                        logger.warning(
                            f"转换 Weather 时间间隔为 Timedelta 时出错: {td_e}")

                else:
                    logger.warning("无法计算 Weather 的常见时间间隔 (聚合或转换 Pandas 后为空)。")

        except AnalysisException as ae:
            logger.error(f"分析 Weather 时间戳间隔时发生 Spark 分析错误: {ae}")
        except Exception as e:
            logger.exception(f"分析 Weather 时间戳间隔时发生未知错误: {e}")
    else:
        logger.warning("Weather DataFrame 无效或缺少必需列，跳过时间戳间隔分析。")

    logger.info("--- 完成分析时间戳一致性 ---")


def analyze_datetime_features_spark(sdf: DataFrame, plots_dir: Path):
    """
    提取并分析 Demand DataFrame 中的日期时间特征 (Spark 版，英文绘图标签)。

    Args:
        sdf (DataFrame): Demand Spark DataFrame, needs 'timestamp' and 'y'.
        plots_dir (Path): Directory Path to save the plots.
    """
    logger.info("--- 开始分析日期时间特征 (Spark) ---")
    plots_dir = Path(plots_dir)  # Ensure Path
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(sdf, DataFrame) or 'timestamp' not in sdf.columns or 'y' not in sdf.columns:
        logger.error("输入 DataFrame 无效或缺少 'timestamp'/'y' 列。")
        return

    try:
        # 提取时间特征
        logger.debug("在 Spark 中提取日期时间特征...")
        sdf_with_features = sdf.withColumn("year", F.year("timestamp")) \
                               .withColumn("month", F.month("timestamp")) \
                               .withColumn("dayofweek", F.dayofweek("timestamp")) \
                               .withColumn("hour", F.hour("timestamp"))
        # Add more if needed: dayofyear, minute, date etc.
        # .withColumn("dayofyear", F.dayofyear("timestamp")) \
        # .withColumn("minute", F.minute("timestamp")) \
        # .withColumn("date", F.to_date("timestamp"))

        # --- 按小时分析平均需求 ---
        logger.info("计算每小时的平均电力需求...")
        avg_demand_by_hour_sdf = sdf_with_features.groupBy("hour") \
            .agg(F.mean("y").alias("avg_demand")) \
            .orderBy("hour")
        avg_demand_by_hour = avg_demand_by_hour_sdf.toPandas()

        if not avg_demand_by_hour.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x="hour", y="avg_demand", data=avg_demand_by_hour,
                        palette="viridis", hue="hour", legend=False)  # Added hue
            # English Title
            plt.title("Average Electricity Demand by Hour of Day")
            plt.xlabel("Hour of Day")  # English Label
            plt.ylabel("Average Demand (y)")  # English Label
            plt.tight_layout()
            fig = plt.gcf()
            save_plot(fig, "avg_demand_by_hour_spark.png",
                      plots_dir)  # Use save_plot
        else:
            logger.warning("无法计算每小时平均需求（聚合结果为空）。")

        # --- 按星期几分析平均需求 ---
        logger.info("计算每周中每天的平均电力需求...")
        avg_demand_by_dow_sdf = sdf_with_features.groupBy("dayofweek") \
            .agg(F.mean("y").alias("avg_demand")) \
            .orderBy("dayofweek")
        avg_demand_by_dow = avg_demand_by_dow_sdf.toPandas()

        if not avg_demand_by_dow.empty:
            # Use English day names, map according to Spark's Sunday=1 convention
            dow_map = {1: 'Sun', 2: 'Mon', 3: 'Tue',
                       4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
            avg_demand_by_dow['day_name'] = avg_demand_by_dow['dayofweek'].map(
                dow_map)
            # Define the correct order for plotting
            day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            # Filter out any days not present in the map (shouldn't happen with standard dayofweek)
            avg_demand_by_dow = avg_demand_by_dow[avg_demand_by_dow['day_name'].isin(
                day_order)]

            plt.figure(figsize=(12, 6))
            sns.barplot(x="day_name", y="avg_demand", data=avg_demand_by_dow, palette="magma",
                        order=day_order, hue="day_name", legend=False)  # Added hue, use defined order
            # English Title
            plt.title("Average Electricity Demand by Day of Week")
            plt.xlabel("Day of Week")  # English Label
            plt.ylabel("Average Demand (y)")  # English Label
            plt.tight_layout()
            fig = plt.gcf()
            save_plot(fig, "avg_demand_by_dayofweek_spark.png",
                      plots_dir)  # Use save_plot
        else:
            logger.warning("无法计算星期几平均需求（聚合结果为空）。")

        # --- 按月份分析平均需求 ---
        logger.info("计算每月平均电力需求...")
        avg_demand_by_month_sdf = sdf_with_features.groupBy("month") \
            .agg(F.mean("y").alias("avg_demand")) \
            .orderBy("month")
        avg_demand_by_month = avg_demand_by_month_sdf.toPandas()

        if not avg_demand_by_month.empty:
            # Use English month names
            month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                         7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            month_order = list(month_map.values())
            avg_demand_by_month['month_name'] = avg_demand_by_month['month'].map(
                month_map)
            avg_demand_by_month = avg_demand_by_month[avg_demand_by_month['month_name'].isin(
                month_order)]

            plt.figure(figsize=(12, 6))
            sns.barplot(x="month_name", y="avg_demand", data=avg_demand_by_month, palette="plasma",
                        order=month_order, hue="month_name", legend=False)  # Added hue, use defined order
            plt.title("Average Electricity Demand by Month")  # English Title
            plt.xlabel("Month")  # English Label
            plt.ylabel("Average Demand (y)")  # English Label
            plt.tight_layout()
            fig = plt.gcf()
            save_plot(fig, "avg_demand_by_month_spark.png",
                      plots_dir)  # Use save_plot
        else:
            logger.warning("无法计算月平均需求（聚合结果为空）。")

    except AnalysisException as ae:
        logger.error(f"分析日期时间特征时发生 Spark 分析错误: {ae}")
    except Exception as e:
        logger.exception(f"分析日期时间特征时发生未知错误: {e}")
        # Ensure plots are closed on error
        if 'plt' in locals() and plt.get_fignums():
            plt.close('all')

    logger.info("--- 完成分析日期时间特征 ---")
