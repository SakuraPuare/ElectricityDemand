import sys
import time
from pathlib import Path

from loguru import logger
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

# --- 项目设置 ---
try:
    from electricitydemand.utils.log_utils import setup_logger
    from electricitydemand.utils.project_utils import (
        create_spark_session,
        get_project_root,
        setup_project_paths,
        stop_spark_session,
    )
except ImportError as e:
    print(f"Error importing project utils: {e}", file=sys.stderr)
    sys.exit(1)

project_root = get_project_root()
src_path, data_dir, logs_dir, plots_dir = setup_project_paths(project_root)

# --- 配置日志 ---
log_prefix = Path(__file__).stem  # analyze_weather_completeness
try:
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
except NameError:
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

logger.info(f"项目根目录：{project_root}")
logger.info(f"数据目录：{data_dir}")
logger.info(f"日志目录：{logs_dir}")

# --- 数据文件路径 ---
weather_path = data_dir / "weather_converted.parquet"
merged_data_path = data_dir / "merged_data.parquet"  # 用于抽样检查

logger.info(f"待分析 Weather 数据路径：{weather_path}")
logger.info(f"用于抽样检查的合并数据路径：{merged_data_path}")


# ======================================================================
# ==                  Analysis Function                             ==
# ======================================================================


def analyze_weather_completeness_spark():
    """Analyzes the completeness of the weather data."""
    logger.info("==============================================")
    logger.info("=== 开始执行 Weather 数据完整性分析脚本 (Spark) ===")
    logger.info("==============================================")
    start_run_time = time.time()
    spark = None

    try:
        # --- 创建 SparkSession ---
        logger.info("创建 SparkSession...")
        spark = create_spark_session(
            app_name="WeatherCompletenessAnalysis",
        )

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 步骤 1: 加载 Weather 数据 ---
        logger.info(f"--- 步骤 1: 加载 Weather 数据 {weather_path} ---")
        try:
            sdf_weather = spark.read.parquet(str(weather_path))
            # 确保 timestamp 类型正确，如果需要转换则转换
            if "TimestampNTZType" in str(sdf_weather.schema["timestamp"].dataType) or \
                    "StringType" in str(sdf_weather.schema["timestamp"].dataType):
                logger.warning(
                    f"Weather timestamp type is {sdf_weather.schema['timestamp'].dataType}. Converting to standard TimestampType.")
                sdf_weather = sdf_weather.withColumn(
                    "timestamp", F.to_timestamp("timestamp"))

            if "TimestampType" not in str(sdf_weather.schema["timestamp"].dataType):
                logger.error(
                    f"Weather timestamp type is unexpected after conversion: {sdf_weather.schema['timestamp'].dataType}")
                raise TypeError("Weather timestamp is not TimestampType")

            # 去重，以防万一
            sdf_weather_dedup = sdf_weather.dropDuplicates(
                ["location_id", "timestamp"])
            original_count = sdf_weather.count()
            dedup_count = sdf_weather_dedup.count()
            if original_count != dedup_count:
                logger.warning(
                    f"Weather data had duplicates. Original: {original_count:,}, Deduped: {dedup_count:,}")
            else:
                logger.info(
                    f"Weather data loaded and no duplicates found. Row count: {dedup_count:,}")
            # Cache for multiple uses
            sdf_weather_dedup.persist(StorageLevel.MEMORY_AND_DISK)

        except Exception as load_e:
            logger.exception(f"加载 Weather Parquet 文件失败：{load_e}")
            raise

        # --- 步骤 2: 分析每个 location_id 的完整性 ---
        logger.info("--- 步骤 2: 分析每个 location_id 的数据完整性 ---")
        try:
            # 计算每个 location_id 的 min/max timestamp 和实际记录数
            weather_stats = sdf_weather_dedup.groupBy("location_id").agg(
                F.min("timestamp").alias("min_ts"),
                F.max("timestamp").alias("max_ts"),
                F.count("*").alias("actual_records")
            )

            # 计算理论小时数 (max_ts - min_ts) / 3600 seconds + 1 hour
            # F.unix_timestamp converts timestamp to seconds since epoch
            weather_stats = weather_stats.withColumn(
                "expected_hours",
                ((F.unix_timestamp("max_ts") - F.unix_timestamp("min_ts")) /
                 3600 + 1).cast("long")
            )

            # 计算完整性百分比
            weather_stats = weather_stats.withColumn(
                "completeness_pct",
                (F.col("actual_records") / F.col("expected_hours")) * 100
            )

            # 收集结果并打印
            logger.info("--- Weather Data Completeness by Location ID ---")
            weather_stats_collected = weather_stats.orderBy(
                "location_id").collect()
            total_actual_records = 0
            total_expected_hours = 0
            for row in weather_stats_collected:
                total_actual_records += row["actual_records"]
                total_expected_hours += row["expected_hours"]
                logger.info(
                    f"Location '{row['location_id']}': "
                    f"Range [{row['min_ts']} to {row['max_ts']}], "
                    f"Expected Hours: {row['expected_hours']:,}, "
                    f"Actual Records: {row['actual_records']:,}, "
                    f"Completeness: {row['completeness_pct']:.2f}%"
                )

            if total_expected_hours > 0:
                overall_completeness = (
                                               total_actual_records / total_expected_hours) * 100
                logger.info("--- Overall Weather Data Completeness ---")
                logger.info(
                    f"Total Expected Hours (Sum across locations): {total_expected_hours:,}")
                logger.info(
                    f"Total Actual Records Found: {total_actual_records:,}")
                logger.warning(
                    f"Overall Completeness: {overall_completeness:.2f}%")
            else:
                logger.warning(
                    "Could not calculate overall completeness (no expected hours).")

        except Exception as analysis_e:
            logger.exception(f"分析 Weather 完整性时出错：{analysis_e}")

        # --- 步骤 3: 抽样检查合并数据中的缺失 Weather ---
        logger.info("--- 步骤 3: 抽样检查合并数据中的缺失 Weather 记录 ---")
        try:
            logger.info(f"加载合并后的数据：{merged_data_path}")
            sdf_merged = spark.read.parquet(str(merged_data_path))

            # 筛选出天气信息缺失的行 (检查 temperature_2m)
            # 并且排除 location_id 为 null 或 那个已知不存在于 weather data 的 location id
            # (需要从上个脚本的日志获取那个 specific location id - 假设是 'loc_only_in_demand')
            # !! 这里需要手动填入那个只存在于 Demand+Meta 中的 location_id !!
            # <-- TODO: Fill this based on previous log (e.g., 'None')
            demand_only_loc_id = None
            # 从日志 logs/2_run_preprocessing_20250421_190751.log 看到 location_id 'None' 是缺失的
            demand_only_loc_id = 'None' if demand_only_loc_id is None else demand_only_loc_id

            logger.info(
                f"筛选天气缺失的行 (排除 location_id='{demand_only_loc_id}' 和 null)...")
            missing_weather_rows = sdf_merged.filter(
                F.col("temperature_2m").isNull() &
                F.col("location_id").isNotNull() &
                (F.col("location_id") !=
                 demand_only_loc_id if demand_only_loc_id else F.lit(True))
            )
            missing_weather_rows.persist(StorageLevel.MEMORY_AND_DISK)
            num_missing_rows = missing_weather_rows.count()
            logger.info(
                f"找到 {num_missing_rows:,} 行天气缺失 (排除已知不匹配的 location_id)。")

            if num_missing_rows > 0:
                sample_size = min(10, num_missing_rows)  # 抽样检查 10 条
                logger.info(f"随机抽取 {sample_size} 条缺失记录进行检查...")
                sampled_missing = missing_weather_rows.select(
                    "location_id", "timestamp").limit(sample_size).collect()

                logger.info(
                    "检查这些 (location_id, timestamp) 是否存在于原始 Weather 数据中：")
                missing_found_in_weather = 0
                for i, row in enumerate(sampled_missing):
                    loc_id = row["location_id"]
                    ts = row["timestamp"]
                    # 在去重后的 weather data 中查找
                    exists = sdf_weather_dedup.filter(
                        (F.col("location_id") == loc_id) & (
                                F.col("timestamp") == ts)
                    ).count() > 0
                    if exists:
                        logger.error(
                            f"  - [{i + 1}/{sample_size}] 错误！({loc_id}, {ts}) 存在于 Weather 数据中，但不应在缺失列表中！")
                        missing_found_in_weather += 1
                    else:
                        logger.info(
                            f"  - [{i + 1}/{sample_size}] OK. ({loc_id}, {ts}) 确实不存在于 Weather 数据中。")

                if missing_found_in_weather == 0:
                    logger.success(
                        "抽样检查确认：合并数据中天气缺失的行，其 (location_id, timestamp) 在原始 Weather 数据中确实不存在。")
                else:
                    logger.error(
                        f"抽样检查发现 {missing_found_in_weather} 条记录本应匹配但未匹配！可能存在 Join 问题或数据问题。")
            else:
                logger.info("没有发现需要抽样检查的天气缺失行（已排除已知不匹配）。")

            if missing_weather_rows.is_cached:
                missing_weather_rows.unpersist()

        except Exception as sample_check_e:
            logger.exception(f"抽样检查时出错：{sample_check_e}")

        if sdf_weather_dedup.is_cached:
            sdf_weather_dedup.unpersist()
        logger.info("==============================================")
        logger.info("=== Weather 数据完整性分析脚本 (Spark) 执行完毕 ===")
        logger.info("==============================================")

    except Exception as e:
        logger.critical(f"分析过程中发生严重错误：{e}")
        logger.exception("Traceback:")
    finally:
        if spark:
            try:
                if not spark.sparkContext._jsc.sc().isStopped():
                    logger.info("正在停止 SparkSession...")
                    spark.stop()
                    logger.info("SparkSession 已停止。")
                else:
                    logger.info("SparkSession 已停止。")
            except Exception as stop_e:
                logger.error(f"停止 SparkSession 时发生错误：{stop_e}")
        else:
            logger.info("SparkSession 未成功初始化或已停止。")

        end_run_time = time.time()
        logger.info(
            f"--- Spark Weather 完整性分析脚本总执行时间：{end_run_time - start_run_time:.2f} 秒 ---")


if __name__ == "__main__":
    analyze_weather_completeness_spark()
