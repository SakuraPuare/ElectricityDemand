import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# 设置绘图风格
sns.set_theme(style="whitegrid")


def analyze_timestamp_consistency(sdf_demand: DataFrame, sdf_weather: DataFrame):
    """
    分析 Demand 和 Weather 数据的时间戳频率和一致性 (Spark版)。

    Args:
        sdf_demand (DataFrame): Demand Spark DataFrame，需要包含 'unique_id' 和 'timestamp' 列。
        sdf_weather (DataFrame): Weather Spark DataFrame，需要包含 'location_id' 和 'timestamp' 列。
    """
    logger.info("--- 开始分析时间戳一致性 (Spark) ---")

    # --- Demand 时间戳频率分析 ---
    logger.info("分析 Demand 数据时间戳间隔...")
    if sdf_demand:
        try:
            # 对每个 unique_id 计算时间差
            window_spec = Window.partitionBy("unique_id").orderBy("timestamp")
            sdf_demand_diff = sdf_demand.withColumn(
                "prev_timestamp", F.lag("timestamp", 1).over(window_spec)
            )
            # 计算时间差（秒）
            sdf_demand_diff = sdf_demand_diff.withColumn(
                "diff_seconds",
                F.when(F.col("prev_timestamp").isNull(), None)
                .otherwise(F.col("timestamp").cast("long") - F.col("prev_timestamp").cast("long"))
            )
            # 统计最常见的时间间隔 (抽样以加速)
            # 注意：直接对整个 DataFrame 进行 groupBy().count() 可能非常慢
            # 我们对 diff_seconds 进行抽样统计
            sample_frac_freq = 0.001 # 使用更小的抽样比例进行频率统计
            logger.info(f"对 Demand 时间差进行抽样 (比例: {sample_frac_freq}) 以统计频率...")
            common_intervals_demand = sdf_demand_diff \
                .select("diff_seconds") \
                .sample(withReplacement=False, fraction=sample_frac_freq) \
                .groupBy("diff_seconds") \
                .count() \
                .orderBy(F.desc("count")) \
                .limit(10) \
                .toPandas() # 转换为 Pandas 以便打印

            if not common_intervals_demand.empty:
                logger.info("Demand 数据中最常见的 10 个时间间隔 (秒):")
                logger.info(f"\n{common_intervals_demand.to_string()}")
                # 将秒转换为更易读的格式
                common_intervals_demand['interval'] = pd.to_timedelta(common_intervals_demand['diff_seconds'], unit='s')
                logger.info("Demand 数据中最常见的 10 个时间间隔 (易读格式):")
                logger.info(f"\n{common_intervals_demand[['interval', 'count']].to_string()}")
            else:
                logger.warning("无法计算 Demand 的常见时间间隔 (可能抽样后为空或数据问题)。")

        except Exception as e:
            logger.exception(f"分析 Demand 时间戳间隔时出错: {e}")
    else:
        logger.warning("Demand DataFrame 为空，跳过时间戳间隔分析。")


    # --- Weather 时间戳频率分析 ---
    logger.info("分析 Weather 数据时间戳间隔...")
    if sdf_weather:
        try:
            # 对每个 location_id 计算时间差
            window_spec_weather = Window.partitionBy("location_id").orderBy("timestamp")
            sdf_weather_diff = sdf_weather.withColumn(
                "prev_timestamp", F.lag("timestamp", 1).over(window_spec_weather)
            )
            # 计算时间差（秒）
            sdf_weather_diff = sdf_weather_diff.withColumn(
                "diff_seconds",
                 F.when(F.col("prev_timestamp").isNull(), None)
                .otherwise(F.col("timestamp").cast("long") - F.col("prev_timestamp").cast("long"))
            )
            # 统计最常见的时间间隔 (抽样)
            logger.info(f"对 Weather 时间差进行抽样 (比例: {sample_frac_freq}) 以统计频率...")
            common_intervals_weather = sdf_weather_diff \
                .select("diff_seconds") \
                .sample(withReplacement=False, fraction=sample_frac_freq) \
                .groupBy("diff_seconds") \
                .count() \
                .orderBy(F.desc("count")) \
                .limit(10) \
                .toPandas() # 转换为 Pandas 以便打印

            if not common_intervals_weather.empty:
                logger.info("Weather 数据中最常见的 10 个时间间隔 (秒):")
                logger.info(f"\n{common_intervals_weather.to_string()}")
                # 将秒转换为更易读的格式
                common_intervals_weather['interval'] = pd.to_timedelta(common_intervals_weather['diff_seconds'], unit='s')
                logger.info("Weather 数据中最常见的 10 个时间间隔 (易读格式):")
                logger.info(f"\n{common_intervals_weather[['interval', 'count']].to_string()}")

                # 检查主要频率是否为 1 小时 (3600 秒)
                if not common_intervals_weather.empty and common_intervals_weather.iloc[0]['diff_seconds'] == 3600:
                    logger.info("Weather 数据的主要时间间隔确认为 1 小时。")
                else:
                    logger.warning("Weather 数据的主要时间间隔似乎不是 1 小时。")
            else:
                 logger.warning("无法计算 Weather 的常见时间间隔 (可能抽样后为空或数据问题)。")

        except Exception as e:
            logger.exception(f"分析 Weather 时间戳间隔时出错: {e}")
    else:
        logger.warning("Weather DataFrame 为空，跳过时间戳间隔分析。")

    logger.info("--- 完成分析时间戳一致性 ---")


def analyze_datetime_features_spark(sdf: DataFrame, plots_dir: str):
    """
    提取并分析 Demand DataFrame 中的日期时间特征 (Spark 版)。

    Args:
        sdf (DataFrame): Demand Spark DataFrame，需要包含 'timestamp' 和 'y' 列。
        plots_dir (str): 保存图表的目录路径。
    """
    logger.info("--- 开始分析日期时间特征 (Spark) ---")
    if not sdf or 'timestamp' not in sdf.columns or 'y' not in sdf.columns:
        logger.error("输入 DataFrame 无效或缺少 'timestamp'/'y' 列。")
        return

    try:
        # 提取时间特征
        sdf_with_features = sdf.withColumn("year", F.year("timestamp")) \
                               .withColumn("month", F.month("timestamp")) \
                               .withColumn("dayofweek", F.dayofweek("timestamp")) \
                               .withColumn("dayofyear", F.dayofyear("timestamp")) \
                               .withColumn("hour", F.hour("timestamp")) \
                               .withColumn("minute", F.minute("timestamp")) \
                               .withColumn("date", F.to_date("timestamp"))

        # --- 按小时分析平均需求 ---
        logger.info("计算每小时的平均电力需求...")
        # 直接在 Spark 中聚合计算，然后收集结果绘图
        avg_demand_by_hour = sdf_with_features.groupBy("hour") \
                                              .agg(F.mean("y").alias("avg_demand")) \
                                              .orderBy("hour") \
                                              .toPandas() # 收集结果到 Pandas

        if not avg_demand_by_hour.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x="hour", y="avg_demand", data=avg_demand_by_hour, palette="viridis")
            plt.title("Average Electricity Demand by Hour of Day (Spark Aggregation)")
            plt.xlabel("Hour of Day")
            plt.ylabel("Average Demand (y)")
            plot_path = f"{plots_dir}/avg_demand_by_hour_spark.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"小时平均需求图已保存至: {plot_path}")
        else:
            logger.warning("无法计算每小时平均需求（聚合结果为空）。")

        # --- 按星期几分析平均需求 ---
        logger.info("计算每周中每天的平均电力需求...")
        avg_demand_by_dow = sdf_with_features.groupBy("dayofweek") \
                                              .agg(F.mean("y").alias("avg_demand")) \
                                              .orderBy("dayofweek") \
                                              .toPandas()

        if not avg_demand_by_dow.empty:
            # 将数字转换为星期名称 (假设 1=Sunday, ..., 7=Saturday or 1=Monday,...7=Sunday based on Spark config)
            # Spark dayofweek: Sunday=1, Saturday=7. Let's map for clarity.
            dow_map = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
            avg_demand_by_dow['day_name'] = avg_demand_by_dow['dayofweek'].map(dow_map)
            # 排序以保持一致
            avg_demand_by_dow = avg_demand_by_dow.sort_values('dayofweek')


            plt.figure(figsize=(12, 6))
            sns.barplot(x="day_name", y="avg_demand", data=avg_demand_by_dow, palette="magma", order=dow_map.values())
            plt.title("Average Electricity Demand by Day of Week (Spark Aggregation)")
            plt.xlabel("Day of Week")
            plt.ylabel("Average Demand (y)")
            plot_path = f"{plots_dir}/avg_demand_by_dayofweek_spark.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"星期几平均需求图已保存至: {plot_path}")
        else:
             logger.warning("无法计算星期几平均需求（聚合结果为空）。")

        # --- 按月份分析平均需求 ---
        logger.info("计算每月平均电力需求...")
        avg_demand_by_month = sdf_with_features.groupBy("month") \
                                               .agg(F.mean("y").alias("avg_demand")) \
                                               .orderBy("month") \
                                               .toPandas()

        if not avg_demand_by_month.empty:
            month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            avg_demand_by_month['month_name'] = avg_demand_by_month['month'].map(month_map)
            avg_demand_by_month = avg_demand_by_month.sort_values('month')


            plt.figure(figsize=(12, 6))
            sns.barplot(x="month_name", y="avg_demand", data=avg_demand_by_month, palette="plasma", order=month_map.values())
            plt.title("Average Electricity Demand by Month (Spark Aggregation)")
            plt.xlabel("Month")
            plt.ylabel("Average Demand (y)")
            plot_path = f"{plots_dir}/avg_demand_by_month_spark.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"月平均需求图已保存至: {plot_path}")
        else:
            logger.warning("无法计算月平均需求（聚合结果为空）。")

    except Exception as e:
        logger.exception(f"分析日期时间特征时出错: {e}")

    logger.info("--- 完成分析日期时间特征 ---") 