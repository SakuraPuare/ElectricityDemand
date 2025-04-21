import sys
import time
from pathlib import Path  # 使用 Pathlib

from loguru import logger

# 移除 Dask 导入
# import dask.dataframe as dd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # 导入 SparkSession 和 functions
from pyspark.storagelevel import StorageLevel

# --- 项目设置 ---
# 使用工具函数
try:
    from electricitydemand.utils.project_utils import get_project_root, setup_project_paths, create_spark_session, stop_spark_session
    from electricitydemand.utils.log_utils import setup_logger  # 仍然直接导入 setup_logger
except ImportError as e:
    print(f"Error importing project utils: {e}", file=sys.stderr)
    sys.exit(1)

project_root = get_project_root()
src_path, data_dir, logs_dir, plots_dir = setup_project_paths(project_root)

# --- 配置日志 ---
log_prefix = Path(__file__).stem  # Use run_preprocessing as prefix
try:
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
except NameError:  # If setup_logger wasn't imported successfully
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

logger.info(f"项目根目录：{project_root}")
logger.info(f"数据目录：{data_dir}")
logger.info(f"日志目录：{logs_dir}")

# --- 数据文件路径 ---
demand_hourly_path = data_dir / "demand_converted.parquet"  # 重采样后的需求数据
metadata_path = data_dir / "metadata.parquet"  # 元数据
weather_path = data_dir / "weather_converted.parquet"  # 天气数据
merged_output_path = data_dir / "merged_data.parquet"  # 合并后数据输出路径

logger.info(f"小时 Demand 数据路径: {demand_hourly_path}")
logger.info(f"Metadata 数据路径: {metadata_path}")
logger.info(f"Weather 数据路径: {weather_path}")
logger.info(f"合并后数据输出路径: {merged_output_path}")


# --- 导入所需函数 ---
try:
    # 假设 load_data 现在也包含一个 Spark 加载函数或我们直接用 spark.read
    # 这里我们直接使用 spark.read.parquet
    # from electricitydemand.eda.load_data import load_demand_data_spark # 假设有这样一个函数
    from electricitydemand.funcs.preprocessing import (
        resample_demand_to_hourly_spark,  # 导入 Spark 版本的函数
    )
    from electricitydemand.funcs.preprocessing import (
        validate_resampling_spark,  # 导入 Spark 版本的函数
    )
except ImportError as e:
    logger.exception(f"Failed to import necessary modules: {e}")
    sys.exit(1)

# ======================================================================
# ==                  Demand Resampling Function                      ==
# ======================================================================


def run_demand_resampling_spark():
    """Loads demand data using Spark and resamples it to hourly frequency."""
    logger.info("=========================================")
    logger.info("=== 开始执行 Demand 数据重采样脚本 (Spark) ===")
    logger.info("=========================================")
    start_run_time = time.time()
    spark = None  # 初始化 SparkSession 变量

    try:
        # --- 创建 SparkSession ---
        logger.info("创建 SparkSession...")
        # 重采样可能需要较多 shuffle，保持一些特定配置
        resample_spark_configs = {
            "spark.sql.adaptive.coalescePartitions.minPartitionSize": "1000000",
            "spark.sql.adaptive.coalescePartitions.minNumPartitions": "100",
            "spark.sql.adaptive.coalescePartitions.maxNumPartitions": "1000",
            "spark.sql.shuffle.partitions": "200",
            "spark.default.parallelism": "200"
        }
        spark = create_spark_session(
            app_name="ElectricityDemandPreprocessing - Resample",
            driver_mem_ratio=0.4,  # 使用与 EDA 相同的比例和默认值
            executor_mem_ratio=0.4,
            default_mem_gb=2,
            additional_configs=resample_spark_configs
        )

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 加载数据 (使用 Spark) ---
        demand_path = data_dir / "demand_converted.parquet"  # 原始转换后数据
        logger.info("--- 步骤 1: 加载 Demand 数据 (Spark) ---")
        try:
            sdf_demand = spark.read.parquet(str(demand_path))
            logger.info("Demand Parquet 文件加载成功。")
            logger.info("Demand Schema:")
            sdf_demand.printSchema()
            # logger.info(f"Demand Count (可能触发计算): {sdf_demand.count():,}") # Count can be slow
        except Exception as load_e:
            logger.exception(f"加载 Demand Parquet 文件失败: {load_e}")
            raise  # 重新抛出异常以停止执行

        # --- 数据预处理：重采样 (使用 Spark) ---
        logger.info("--- 步骤 2: Demand 数据重采样 (Spark) ---")
        sdf_demand_hourly = resample_demand_to_hourly_spark(sdf_demand)

        # --- 验证重采样结果 (使用 Spark) ---
        if sdf_demand_hourly is not None:
            logger.info("--- 步骤 3: 验证重采样结果 (Spark) ---")
            # Check first 10 records using Spark
            validate_resampling_spark(sdf_demand_hourly, n_check=10)

            # --- 保存重采样后的数据 (使用 Spark) ---
            logger.info(f"--- 步骤 4: 保存重采样后的数据到 {demand_hourly_path} ---")
            try:
                logger.info("开始写入 Parquet 文件...")
                # 使用 Spark DataFrameWriter API 保存
                sdf_demand_hourly.write.mode(
                    "overwrite").parquet(str(demand_hourly_path))
                logger.success(f"成功保存重采样数据到：{demand_hourly_path}")
            except Exception as e:
                logger.exception(f"保存 Spark 重采样数据时出错：{e}")

        else:
            logger.error("Demand 数据 Spark 重采样失败，无法进行验证或保存。")

        logger.info("=========================================")
        logger.info("===  Demand 数据重采样脚本 (Spark) 执行完毕 ===")
        logger.info("=========================================")

    except Exception as e:
        logger.critical(f"执行过程中发生严重错误: {e}")
        logger.exception("Traceback:")
    finally:
        # Unpersist cached data if exists
        if 'sdf_demand_hourly' in locals() and sdf_demand_hourly is not None and sdf_demand_hourly.is_cached:
            sdf_demand_hourly.unpersist()
        stop_spark_session(spark)  # Use utility function

        end_run_time = time.time()
        logger.info(
            f"--- Spark 重采样脚本总执行时间: {end_run_time - start_run_time:.2f} 秒 ---")

# ======================================================================
# ==                      Merge Data Function                       ==
# ======================================================================


def run_merge_data_spark():
    """Loads resampled demand, metadata, and weather data using Spark and merges them."""
    logger.info("=========================================")
    logger.info("=== 开始执行 数据合并脚本 (Spark) ===")
    logger.info("=========================================")
    start_run_time = time.time()
    spark = None  # 初始化 SparkSession 变量

    try:
        # --- 创建 SparkSession ---
        logger.info("创建 SparkSession...")
        # 合并操作可能需要更多内存，适当增加比例或设置固定值
        merge_spark_configs = {
            "spark.sql.shuffle.partitions": "200"  # 保持 shuffle 分区数
            # "spark.sql.autoBroadcastJoinThreshold": "-1" # 如有 OOM 可考虑禁用广播
        }
        spark = create_spark_session(
            app_name="ElectricityDemandPreprocessing - Merge",
            driver_mem_ratio=0.5,  # 增加内存比例
            executor_mem_ratio=0.5,
            default_mem_gb=4,  # 增加默认值
            additional_configs=merge_spark_configs
        )

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 步骤 1: 加载数据 (Spark) ---
        logger.info("--- 步骤 1: 加载所需数据 (Spark) ---")
        try:
            logger.info(f"加载小时 Demand 数据: {demand_hourly_path}")
            sdf_demand = spark.read.parquet(str(demand_hourly_path))
            logger.info("小时 Demand 数据加载成功。")
            # sdf_demand.printSchema() # Schema in logs already

            logger.info(f"加载 Metadata 数据: {metadata_path}")
            sdf_meta = spark.read.parquet(str(metadata_path))
            # 选择需要的列，避免加载过多冗余信息，并重命名以防冲突
            sdf_meta = sdf_meta.select(
                "unique_id",
                "location_id",
                "building_class",
                "cluster_size",
                # 可以根据需要添加其他 metadata 列
            )
            logger.info("Metadata 数据加载成功。")
            # sdf_meta.printSchema() # Schema in logs already

            logger.info(f"加载 Weather 数据: {weather_path}")
            sdf_weather = spark.read.parquet(str(weather_path))
            # 确保 weather 的 timestamp 是 TimestampType
            if "StringType" in str(sdf_weather.schema["timestamp"].dataType):
                logger.warning(
                    "Weather 'timestamp' is StringType, attempting conversion...")
                sdf_weather = sdf_weather.withColumn(
                    "timestamp", F.to_timestamp("timestamp"))
                logger.info("Weather 'timestamp' converted to TimestampType.")
            # --- 新增/修改检查 ---
            elif "TimestampNTZType" in str(sdf_weather.schema["timestamp"].dataType):
                logger.warning(
                    "Weather 'timestamp' is TimestampNTZType, converting to TimestampType...")
                sdf_weather = sdf_weather.withColumn(
                    "timestamp", F.col("timestamp").cast("timestamp"))
                logger.info("Weather 'timestamp' converted to TimestampType.")
            elif "TimestampType" not in str(sdf_weather.schema["timestamp"].dataType):
                logger.error(
                    f"Weather 'timestamp' is of unexpected type: {sdf_weather.schema['timestamp'].dataType}. Join may fail.")
                # Potentially raise error or try another conversion
            # --- 结束新增/修改 ---
            logger.info("Weather 数据加载成功。")
            # sdf_weather.printSchema() # Schema in logs already

        except Exception as load_e:
            logger.exception(f"加载 Parquet 文件失败: {load_e}")
            raise

        # --- 步骤 2: 合并数据 (Spark) ---
        logger.info("--- 步骤 2: 合并 Demand, Metadata, 和 Weather 数据 ---")
        try:
            # 2.1 合并 Demand 和 Metadata (保留所有 Demand 记录)
            logger.info("合并 Demand 和 Metadata (left join on unique_id)...")
            # 检查 metadata 中 unique_id 是否唯一，如果不唯一 join 会导致行数增加
            # meta_distinct_count = sdf_meta.select("unique_id").distinct().count() # Costly check, skip for now
            # meta_total_count = sdf_meta.count()
            # if meta_distinct_count != meta_total_count:
            #     logger.warning(f"Metadata 中 unique_id 不唯一 ({meta_distinct_count} distinct vs {meta_total_count} total)! Join 可能导致行数增加。")

            sdf_merged_meta = sdf_demand.join(
                sdf_meta.alias("meta"), "unique_id", "left")
            logger.info("Demand 和 Metadata 合并完成。")
            # 使用 persist 代替 cache，更明确控制存储级别，这里默认 MEMORY_AND_DISK
            # Cache the result as it's used multiple times below
            sdf_merged_meta.persist(StorageLevel.MEMORY_AND_DISK)
            merged_meta_count = sdf_merged_meta.count()
            logger.info(f"合并后 (Demand+Meta) 中间行数: {merged_meta_count:,}")

            # --- *** 新增：对齐 Demand 时间戳到小时 *** ---
            logger.info("将 Demand+Meta 合并结果中的 'timestamp' 向下取整到小时...")
            sdf_merged_meta_hourly_ts = sdf_merged_meta.withColumn(
                "timestamp_hour", F.date_trunc('hour', F.col('timestamp'))
            )
            # 重命名取整后的列为 'timestamp' 以便 Join，但要小心原始时间戳可能丢失
            # 保留原始时间戳并在 join 时使用取整后的列可能更安全，但这里为了简化，直接覆盖
            # 我们也可以选择创建一个新的列 timestamp_hour 用于 join
            # 这里选择创建一个新列用于 Join，避免覆盖原始时间戳（如果需要的话）
            # 但为了与 Weather 的 'timestamp' 列名匹配进行 Join，我们还是需要一个名为 'timestamp' 的小时级列
            # 方案：1. 创建 timestamp_hour, 2. 重命名 weather 的 timestamp 为 timestamp_hour, 3. join on location_id, timestamp_hour
            # 或者：1. 创建 timestamp_hour, 2. 覆盖原始 timestamp 为 timestamp_hour
            # 这里采用创建新列然后重命名的方式进行 Join，之后可以 drop 掉多余的列
            sdf_merged_meta = sdf_merged_meta.withColumn(
                "timestamp_join_key", F.date_trunc('hour', F.col('timestamp'))
            )
            logger.info("Demand+Meta 时间戳已处理为小时级别 (timestamp_join_key)。")
            # 可选：检查一下处理后的时间戳
            # logger.info("处理后时间戳示例:")
            # sdf_merged_meta.select("timestamp", "timestamp_join_key").show(5, False)

            # 2.2 Weather 数据去重
            logger.info("对 Weather 数据按 (location_id, timestamp) 去重...")
            # weather_cols_to_select = [col for col in sdf_weather.columns if col not in ["location_id", "timestamp"]] # 不再需要，直接用原始DF
            sdf_weather_dedup = sdf_weather.dropDuplicates(
                ["location_id", "timestamp"])
            weather_original_count = sdf_weather.count()
            weather_dedup_count = sdf_weather_dedup.count()
            if weather_original_count != weather_dedup_count:
                logger.warning(
                    f"Weather 数据中存在重复的 (location_id, timestamp) 记录，已去重。原始: {weather_original_count:,}, 去重后: {weather_dedup_count:,}")
            else:
                logger.info(
                    f"Weather 数据中无重复的 (location_id, timestamp) 记录。行数: {weather_dedup_count:,}")
            # Cache deduped weather data
            sdf_weather_dedup.persist(StorageLevel.MEMORY_AND_DISK)

            # --- Diagnostic Logging: 分析 location_id 覆盖情况 ---
            logger.info("--- [诊断] 开始分析 location_id 覆盖情况 ---")
            # 提取 Demand+Meta 中的 location_id (去重)
            demand_meta_locations = sdf_merged_meta.select(
                "location_id").distinct()
            demand_meta_locations.persist()
            num_demand_meta_locations = demand_meta_locations.count()
            logger.info(
                f"[诊断] Demand+Meta 中唯一的 location_id 数量: {num_demand_meta_locations:,}")

            # 提取 Weather 中的 location_id (去重)
            weather_locations = sdf_weather_dedup.select(
                "location_id").distinct()
            weather_locations.persist()
            num_weather_locations = weather_locations.count()
            logger.info(
                f"[诊断] 去重后 Weather 中唯一的 location_id 数量: {num_weather_locations:,}")

            # 计算交集 (有多少 location_id 在两者中都存在)
            common_locations_count = demand_meta_locations.join(
                weather_locations, "location_id", "inner").count()
            logger.info(
                f"[诊断] Demand+Meta 与 Weather 共有的 location_id 数量: {common_locations_count:,}")

            # 计算只在 Demand+Meta 中存在的 location_id (这些肯定无法匹配天气)
            demand_only_locations_count = demand_meta_locations.join(
                weather_locations, "location_id", "left_anti").count()
            logger.warning(
                f"[诊断] 只存在于 Demand+Meta 中的 location_id 数量 (无法匹配天气): {demand_only_locations_count:,}")
            if demand_only_locations_count > 0 and num_demand_meta_locations > 0:
                percentage_missing_loc = (
                    demand_only_locations_count / num_demand_meta_locations) * 100
                logger.warning(
                    f"[诊断] 这占 Demand+Meta 总 location_id 的 {percentage_missing_loc:.2f}%")

            # 清理缓存
            demand_meta_locations.unpersist()
            weather_locations.unpersist()
            logger.info("--- [诊断] location_id 覆盖情况分析完毕 ---")
            # --- End Diagnostic Logging ---

            # --- Diagnostic Logging: 分析 Timestamp 范围对齐情况 ---
            # 这个诊断现在应该在取整后的时间戳上进行，或者保留原来的逻辑以查看原始范围
            # 为了简单起见，暂时保留原始逻辑，但需要注意其解释可能受时间戳处理的影响
            logger.info(
                "--- [诊断] 开始分析 Timestamp 范围对齐情况 (按 location_id, 基于原始时间戳) ---")
            try:
                # 计算 Demand+Meta 中每个 location_id 的时间范围 (使用原始 timestamp)
                logger.info("[诊断] 计算 Demand+Meta 时间范围 (原始)...")
                demand_ts_range = sdf_merged_meta.groupBy("location_id").agg(
                    F.min("timestamp").alias("min_demand_ts"),
                    F.max("timestamp").alias("max_demand_ts")
                )
                # 计算 Weather 中每个 location_id 的时间范围
                logger.info("[诊断] 计算 Weather 时间范围...")
                weather_ts_range = sdf_weather_dedup.groupBy("location_id").agg(
                    F.min("timestamp").alias("min_weather_ts"),
                    F.max("timestamp").alias("max_weather_ts")
                )

                # 合并时间范围信息 (Outer Join 以包含所有 location_id)
                logger.info("[诊断] 合并时间范围信息...")
                ts_range_comparison = demand_ts_range.join(
                    weather_ts_range,
                    "location_id",
                    "outer"  # Use outer join to see locations present in only one DF
                    # Order for consistent logging, handle nulls
                ).orderBy(F.col("location_id").asc_nulls_last())

                # 收集结果并分析 (location_id 数量不多，collect() 应该可行)
                logger.info("[诊断] 分析每个 location_id 的时间范围...")
                # Limit the collect in case location_id space is larger than expected
                comparison_results = ts_range_comparison.limit(500).collect()
                if len(comparison_results) >= 500:
                    logger.warning("[诊断] Location ID 数量较多，仅分析前 500 个的时间范围。")

                mismatches_found = False
                null_location_issues = False
                for row in comparison_results:
                    loc_id = row["location_id"]
                    min_d = row["min_demand_ts"]
                    max_d = row["max_demand_ts"]
                    min_w = row["min_weather_ts"]
                    max_w = row["max_weather_ts"]

                    # Handle potential None/Null location_id
                    if loc_id is None:
                        null_location_issues = True
                        log_msg_parts = [f"Location ID is NULL/None:"]
                        issues = ["NULL location_id detected"]
                        if min_d is not None or max_d is not None:
                            log_msg_parts.append(
                                f" Demand({min_d} to {max_d})")
                        else:
                            log_msg_parts.append(" Demand(N/A)")
                        if min_w is not None or max_w is not None:
                            # Should not happen if only demand has null loc_id
                            log_msg_parts.append(
                                f" Weather({min_w} to {max_w})")
                            issues.append(
                                "Unexpected weather data for NULL location_id")
                        else:
                            log_msg_parts.append(" Weather(N/A)")
                        log_msg_parts.append(
                            f" -> ISSUES: {'; '.join(issues)}")
                        logger.warning("".join(log_msg_parts))
                        continue  # Skip regular checks for null location id

                    # Regular location_id checks
                    log_msg_parts = [f"Location '{loc_id}':"]
                    issues = []

                    if min_d is None or max_d is None:
                        log_msg_parts.append(" Demand(N/A)")
                        # This might happen if the outer join picked up a location only in weather? Check join logic.
                        # If using outer join on ranges, this is possible. If join based on merged_meta, shouldn't happen unless loc_id was null in merged_meta.
                        issues.append(
                            "Demand data missing for this location_id in range analysis (unexpected)")
                    else:
                        log_msg_parts.append(f" Demand({min_d} to {max_d})")

                    if min_w is None or max_w is None:
                        log_msg_parts.append(" Weather(N/A)")
                        # This is expected for the location_id only in demand+meta
                        if loc_id is not None:  # Already checked for None
                            # Only log as an issue if this location_id was expected to have weather data
                            # Based on previous check, if common_locations_count includes this loc_id, it's an issue.
                            # Simplification: just note weather is missing
                            issues.append(
                                "Weather data missing for this location_id")
                    else:
                        log_msg_parts.append(f" Weather({min_w} to {max_w})")

                    # Check for range mismatches only if both ranges exist
                    if min_d and min_w and min_d < min_w:
                        # Check if difference is significant (e.g., more than an hour due to timezone/DST issues?)
                        time_diff = min_w - min_d
                        if time_diff.total_seconds() > 3600:  # More than 1 hour diff
                            issues.append(
                                f"Demand starts significantly EARLIER ({min_d} vs {min_w}, diff: {time_diff})")
                        # else: issue is minor or expected due to rounding
                    if max_d and max_w and max_d > max_w:
                        time_diff = max_d - max_w
                        if time_diff.total_seconds() > 3600:  # More than 1 hour diff
                            issues.append(
                                f"Demand ends significantly LATER ({max_d} vs {max_w}, diff: {time_diff})")
                        # else: issue is minor or expected

                    log_func = logger.info  # Default to info
                    if issues:
                        mismatches_found = True
                        log_func = logger.warning  # Log as warning if issues found
                        log_msg_parts.append(
                            f" -> ISSUES: {'; '.join(issues)}")

                    log_func("".join(log_msg_parts))

                if null_location_issues:
                    logger.error(
                        "[诊断] 检测到 NULL location_id，这通常是数据问题，可能导致部分数据无法 Join 天气。")

                if not mismatches_found and not null_location_issues:
                    logger.success(
                        "[诊断] 所有检查的 location_id 的时间戳范围看起来基本对齐或符合预期 (允许细微差异或无天气数据)。")
                elif not mismatches_found and null_location_issues:
                    logger.warning("[诊断] 时间戳范围基本对齐，但存在 NULL location_id 的问题。")
                else:
                    logger.warning(
                        "[诊断] 发现部分 location_id 的时间戳范围存在不匹配或缺失天气数据，并可能存在 NULL location_id。")

            except Exception as ts_diag_e:
                logger.error(f"[诊断] 分析 Timestamp 范围时出错: {ts_diag_e}")
                # Log traceback for timestamp errors
                logger.exception("Traceback:")
            logger.info("--- [诊断] Timestamp 范围对齐情况分析完毕 ---")
            # --- End Timestamp Diagnostic ---

            # 2.3 合并上一步结果和 Weather (保留所有 Demand 记录)
            logger.info(
                "合并结果与 Weather (left join on location_id and timestamp_hour)...")
            # Alias weather DF for clarity and to avoid column ambiguity if we didn't rename cols
            sdf_weather_aliased = sdf_weather_dedup.alias("weather")

            # Perform the join using the truncated timestamp key from demand
            # and the original timestamp from weather (which is already hourly)
            sdf_final_merged = sdf_merged_meta.join(
                sdf_weather_aliased,
                (sdf_merged_meta["location_id"] == sdf_weather_aliased["location_id"]) &
                (sdf_merged_meta["timestamp_join_key"]
                 == sdf_weather_aliased["timestamp"]),
                how="left"
            ).drop(sdf_merged_meta["timestamp_join_key"]) \
             .drop(sdf_weather_aliased["location_id"]) \
             .drop(sdf_weather_aliased["timestamp"])
            # 也删除 weather 表的 timestamp，避免与 demand 表的 timestamp 混淆 (可选，如果需要 weather 的原始时间戳则保留)
            # .drop(sdf_weather_aliased["timestamp"])

            logger.info("与 Weather 数据合并完成。")
            logger.info("最终合并数据的 Schema:")
            sdf_final_merged.printSchema()  # Schema should now only have one location_id

            # --- Diagnostic Logging: 检查合并后天气列的 Null 比例 ---
            logger.info("--- [诊断] 开始检查合并后天气列 Null 比例 ---")
            # Re-cache the final merged DF if memory allows, as count can be expensive
            # sdf_final_merged.persist(StorageLevel.MEMORY_AND_DISK) # Optional: Cache final result before count

            final_merged_count = sdf_final_merged.count()
            logger.info(f"[诊断] 最终合并后总行数: {final_merged_count:,}")

            # Check for nulls after join (excluding rows where location_id itself was null in demand)
            # This gives a better idea of join success for valid locations
            sdf_valid_locations = sdf_final_merged.where(
                F.col("location_id").isNotNull())
            valid_location_count = sdf_valid_locations.count()
            logger.info(
                f"[诊断] 其中 location_id 非 Null 的行数: {valid_location_count:,}")

            # 选择一个关键天气列进行检查
            key_weather_col = "temperature_2m"
            if key_weather_col in sdf_final_merged.columns:
                # Count nulls only for rows where location_id was not null initially
                null_weather_count_valid_loc = sdf_valid_locations.where(
                    F.col(key_weather_col).isNull()).count()

                logger.info(
                    f"[诊断] 在 location_id 非 Null 的行中，'{key_weather_col}' 为 Null 的行数: {null_weather_count_valid_loc:,}")
                if valid_location_count > 0:
                    null_percentage = (
                        null_weather_count_valid_loc / valid_location_count) * 100
                    # 预期这个比例会非常低，接近 0%
                    # Log as warning only if > 1% nulls remain
                    log_level = logger.warning if null_percentage > 1.0 else logger.info
                    log_level(
                        f"[诊断] 天气数据 Join 成功率 (基于 '{key_weather_col}', 排除 Null location_id): {100.0 - null_percentage:.2f}% (缺失率: {null_percentage:.2f}%)")
                else:
                    logger.info(
                        "[诊断] 没有 location_id 非 Null 的行，无法计算有效 Join 成功率。")

                # Also log total null count for reference
                total_null_weather_count = sdf_final_merged.where(
                    F.col(key_weather_col).isNull()).count()
                logger.info(
                    f"[诊断] 最终合并数据中 '{key_weather_col}' 总 Null 行数 (包括 Null location_id): {total_null_weather_count:,}")
                if final_merged_count > 0:
                    total_null_perc = (
                        total_null_weather_count / final_merged_count) * 100
                    logger.info(f"[诊断] 占总行数的比例: {total_null_perc:.2f}%")

            else:
                logger.error(
                    f"[诊断] 关键天气列 '{key_weather_col}' 不在最终合并结果中！无法检查 Null 比例。")

            # Unpersist the final merged DF if it was cached
            # if sdf_final_merged.is_cached: sdf_final_merged.unpersist()

            logger.info("--- [诊断] 天气列 Null 比例检查完毕 ---")
            # --- End Diagnostic Logging ---

            # 清理缓存
            # sdf_merged_meta already unpersisted or cached for join below
            if sdf_weather_dedup.is_cached:
                sdf_weather_dedup.unpersist()

        except Exception as merge_e:
            logger.exception(f"数据合并过程中出错: {merge_e}")
            if 'sdf_merged_meta' in locals() and sdf_merged_meta.is_cached:
                sdf_merged_meta.unpersist()  # Clean up cache on error
            if 'sdf_weather_dedup' in locals() and sdf_weather_dedup.is_cached:
                sdf_weather_dedup.unpersist()
            raise

        # --- 步骤 3: 保存合并后的数据 ---
        logger.info(f"--- 步骤 3: 保存合并后的数据到 {merged_output_path} ---")
        try:
            logger.info("开始写入 Parquet 文件...")
            # 可以在写入前按 unique_id 和 timestamp 排序，提高后续读取性能，但写入会变慢
            # logger.info("按 unique_id, timestamp 排序并写入...")
            # sdf_final_merged.orderBy("unique_id", "timestamp").write.mode("overwrite").parquet(str(merged_output_path))
            sdf_final_merged.write.mode(
                "overwrite").parquet(str(merged_output_path))
            logger.success(f"成功保存合并后的数据到：{merged_output_path}")
        except Exception as save_e:
            logger.exception(f"保存合并后的 Spark 数据时出错：{save_e}")
            raise

        logger.info("=========================================")
        logger.info("===     数据合并脚本 (Spark) 执行完毕    ===")
        logger.info("=========================================")

    except Exception as e:
        logger.critical(f"数据合并过程中发生严重错误: {e}")
        logger.exception("Traceback:")
    finally:
        # Ensure intermediate dataframes are unpersisted
        if 'sdf_merged_meta' in locals() and sdf_merged_meta.is_cached:
            sdf_merged_meta.unpersist()
        if 'sdf_weather_dedup' in locals() and sdf_weather_dedup.is_cached:
            sdf_weather_dedup.unpersist()
        stop_spark_session(spark)  # Use utility function

        end_run_time = time.time()
        logger.info(
            f"--- Spark 数据合并脚本总执行时间: {end_run_time - start_run_time:.2f} 秒 ---")


if __name__ == "__main__":
    try:
        # logger.info("运行 Demand 重采样...")
        # run_demand_resampling_spark() # 已完成，注释掉

        logger.info("运行数据合并...")
        run_merge_data_spark()  # 执行数据合并步骤

    except Exception as e:
        # logger.exception(f"执行过程中发生错误：{e}") # 主函数已有更详细的日志
        sys.exit(1)
