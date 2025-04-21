import os
import sys
import time
from pathlib import Path  # 使用 Pathlib

from loguru import logger

# 移除 Dask 导入
# import dask.dataframe as dd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # 导入 SparkSession 和 functions

# --- 项目设置 ---
# Determine project root dynamically
try:
    _script_path = os.path.abspath(__file__)
    project_root = Path(_script_path).parent.parent.parent  # Use Path object
except NameError:  # If __file__ is not defined (e.g., interactive)
    project_root = Path.cwd()
    # Add project root to path if running interactively might be needed
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Add src directory to sys.path to allow absolute imports from src
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- 配置日志 ---
try:
    from electricitydemand.utils.log_utils import setup_logger
except ImportError:
    print("Error: Could not import setup_logger. Ensure src is in PYTHONPATH or script is run correctly.", file=sys.stderr)
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

log_prefix = Path(__file__).stem  # Use run_preprocessing as prefix
logs_dir = project_root / 'logs'
plots_dir = project_root / 'plots'  # 虽然此脚本不用，保持一致性
data_dir = project_root / "data"  # 数据目录 (Path object)

os.makedirs(logs_dir, exist_ok=True)
# os.makedirs(plots_dir, exist_ok=True) # plots 目录在此脚本中可能不需要

if 'setup_logger' in globals():
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")

logger.info(f"项目根目录：{project_root}")
logger.info(f"数据目录：{data_dir}")
logger.info(f"日志目录：{logs_dir}")

# --- 数据文件路径 ---
demand_hourly_path = data_dir / "demand_hourly.parquet" # 重采样后的需求数据
metadata_path = data_dir / "metadata.parquet" # 元数据
weather_path = data_dir / "weather_converted.parquet" # 天气数据
merged_output_path = data_dir / "merged_data.parquet" # 合并后数据输出路径

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
        # 复用 1_run_eda.py 中的 SparkSession 配置逻辑
        import psutil
        total_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        driver_memory = max(2, int(total_memory_gb * 0.4 + 0.5))
        executor_memory = max(2, int(total_memory_gb * 0.4 + 0.5))
        logger.info(
            f"系统可用内存: {total_memory_gb:.2f}GB, 设置驱动器内存: {driver_memory}g, 执行器内存: {executor_memory}g")

        spark = SparkSession.builder \
            .appName("ElectricityDemandPreprocessing - Resample") \
            .config("spark.driver.memory", f"{driver_memory}g") \
            .config("spark.executor.memory", f"{executor_memory}g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 加载数据 (使用 Spark) ---
        demand_path = data_dir / "demand_converted.parquet" # 原始转换后数据
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
        if spark:  # Check if spark session was initialized
            try:
                if not spark.sparkContext._jsc.sc().isStopped():
                    logger.info("正在停止 SparkSession...")
                    spark.stop()
                    logger.info("SparkSession 已停止。")
                else:
                    logger.info("SparkSession 已停止。")
            except Exception as stop_e:
                logger.error(f"停止 SparkSession 时发生错误: {stop_e}")
        else:
            logger.info("SparkSession 未成功初始化或已停止。")

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
        import psutil
        total_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        # 合并操作可能需要更多内存，适当增加比例或设置固定值
        driver_memory = max(4, int(total_memory_gb * 0.5 + 0.5)) # 增加驱动内存
        executor_memory = max(4, int(total_memory_gb * 0.5 + 0.5)) # 增加执行器内存
        logger.info(
            f"系统可用内存: {total_memory_gb:.2f}GB, 设置驱动器内存: {driver_memory}g, 执行器内存: {executor_memory}g")

        spark = SparkSession.builder \
            .appName("ElectricityDemandPreprocessing - Merge") \
            .config("spark.driver.memory", f"{driver_memory}g") \
            .config("spark.executor.memory", f"{executor_memory}g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
            # .config("spark.sql.shuffle.partitions", "200") \ # 调整 shuffle 分区数
            # .config("spark.sql.autoBroadcastJoinThreshold", "-1") \ # 禁用广播连接，避免 OOM
            

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 步骤 1: 加载数据 (Spark) ---
        logger.info("--- 步骤 1: 加载所需数据 (Spark) ---")
        try:
            logger.info(f"加载小时 Demand 数据: {demand_hourly_path}")
            sdf_demand = spark.read.parquet(str(demand_hourly_path))
            logger.info("小时 Demand 数据加载成功。")
            sdf_demand.printSchema()

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
            sdf_meta.printSchema()

            logger.info(f"加载 Weather 数据: {weather_path}")
            sdf_weather = spark.read.parquet(str(weather_path))
            # 确保 weather 的 timestamp 是 TimestampType
            if "StringType" in str(sdf_weather.schema["timestamp"].dataType):
                 logger.warning("Weather 'timestamp' is StringType, attempting conversion...")
                 sdf_weather = sdf_weather.withColumn("timestamp", F.to_timestamp("timestamp"))
                 logger.info("Weather 'timestamp' converted to TimestampType.")
            logger.info("Weather 数据加载成功。")
            sdf_weather.printSchema()

        except Exception as load_e:
            logger.exception(f"加载 Parquet 文件失败: {load_e}")
            raise

        # --- 步骤 2: 合并数据 (Spark) ---
        logger.info("--- 步骤 2: 合并 Demand, Metadata, 和 Weather 数据 ---")
        try:
            # 2.1 合并 Demand 和 Metadata (保留所有 Demand 记录)
            logger.info("合并 Demand 和 Metadata (left join on unique_id)...")
            # 检查 metadata 中 unique_id 是否唯一，如果不唯一 join 会导致行数增加
            meta_distinct_count = sdf_meta.select("unique_id").distinct().count()
            meta_total_count = sdf_meta.count()
            if meta_distinct_count != meta_total_count:
                logger.warning(f"Metadata 中 unique_id 不唯一 ({meta_distinct_count} distinct vs {meta_total_count} total)! Join 可能导致行数增加。")
                # 可选：对 metadata 去重，保留第一条记录
                # from pyspark.sql.window import Window
                # window_spec = Window.partitionBy("unique_id").orderBy(F.lit(1)) # 随意指定一个排序依据
                # sdf_meta = sdf_meta.withColumn("row_num", F.row_number().over(window_spec)).filter(F.col("row_num") == 1).drop("row_num")
                # logger.info("已对 Metadata 按 unique_id 去重。")

            sdf_merged = sdf_demand.join(sdf_meta, "unique_id", "left")
            logger.info("Demand 和 Metadata 合并完成。")
            # sdf_merged.printSchema()
            # logger.info(f"合并后 (Demand+Meta) 行数: {sdf_merged.count()}") # count() is expensive

            # 2.2 合并上一步结果和 Weather (保留所有 Demand 记录)
            logger.info("合并结果与 Weather (left join on location_id and timestamp)...")
            # Weather 的 timestamp 也是小时级别的，可以直接 join
            # 需要处理 weather 中可能存在的重复 location_id + timestamp
            weather_cols_to_select = [col for col in sdf_weather.columns if col not in ["location_id", "timestamp"]]
            sdf_weather_dedup = sdf_weather.dropDuplicates(["location_id", "timestamp"])
            weather_original_count = sdf_weather.count()
            weather_dedup_count = sdf_weather_dedup.count()
            if weather_original_count != weather_dedup_count:
                logger.warning(f"Weather 数据中存在重复的 (location_id, timestamp) 记录，已去重。原始: {weather_original_count}, 去重后: {weather_dedup_count}")

            sdf_final_merged = sdf_merged.join(
                sdf_weather_dedup,
                on=["location_id", "timestamp"],
                how="left"
            )
            logger.info("与 Weather 数据合并完成。")
            sdf_final_merged.printSchema()
            # logger.info(f"最终合并后行数: {sdf_final_merged.count()}") # count() is expensive

        except Exception as merge_e:
            logger.exception(f"数据合并过程中出错: {merge_e}")
            raise

        # --- 步骤 3: 保存合并后的数据 ---
        logger.info(f"--- 步骤 3: 保存合并后的数据到 {merged_output_path} ---")
        try:
            logger.info("开始写入 Parquet 文件...")
            sdf_final_merged.write.mode("overwrite").parquet(str(merged_output_path))
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
        if spark:  # Check if spark session was initialized
            try:
                if not spark.sparkContext._jsc.sc().isStopped():
                    logger.info("正在停止 SparkSession...")
                    spark.stop()
                    logger.info("SparkSession 已停止。")
                else:
                    logger.info("SparkSession 已停止。")
            except Exception as stop_e:
                logger.error(f"停止 SparkSession 时发生错误: {stop_e}")
        else:
            logger.info("SparkSession 未成功初始化或已停止。")

        end_run_time = time.time()
        logger.info(
            f"--- Spark 数据合并脚本总执行时间: {end_run_time - start_run_time:.2f} 秒 ---")



if __name__ == "__main__":
    try:
        # logger.info("运行 Demand 重采样...")
        # run_demand_resampling_spark() # 已完成，注释掉

        logger.info("运行数据合并...")
        run_merge_data_spark() # 执行数据合并步骤

    except Exception as e:
        # logger.exception(f"执行过程中发生错误：{e}") # 主函数已有更详细的日志
        sys.exit(1)
