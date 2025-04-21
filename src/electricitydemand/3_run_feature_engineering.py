import os
import sys
import time
from pathlib import Path

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, TimestampType  # 用于检查列类型
from pyspark.sql.window import Window

# --- 项目设置 ---
try:
    _script_path = os.path.abspath(__file__)
    project_root = Path(_script_path).parent.parent.parent
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

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

log_prefix = Path(__file__).stem # 使用 3_run_feature_engineering 作为前缀
logs_dir = project_root / 'logs'
plots_dir = project_root / 'plots' # 特征工程可能也需要绘图
data_dir = project_root / "data"

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True) # 创建绘图目录

if 'setup_logger' in globals():
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")

logger.info(f"项目根目录：{project_root}")
logger.info(f"数据目录：{data_dir}")
logger.info(f"日志目录：{logs_dir}")
logger.info(f"绘图目录：{plots_dir}")

# --- 数据文件路径 ---
merged_data_path = data_dir / "merged_data.parquet" # 输入：合并后的数据
features_output_path = data_dir / "features.parquet" # 输出：特征工程后的数据

logger.info(f"输入合并数据路径: {merged_data_path}")
logger.info(f"输出特征数据路径: {features_output_path}")


# ======================================================================
# ==                 Feature Engineering Functions                   ==
# ======================================================================

def add_time_features_spark(sdf):
    """Adds time-based features to the Spark DataFrame."""
    logger.info("开始添加时间特征...")

    # 确保 timestamp 列是 TimestampType
    if not isinstance(sdf.schema["timestamp"].dataType, TimestampType):
         logger.warning("Input 'timestamp' is not TimestampType. Attempting conversion.")
         sdf = sdf.withColumn("timestamp_temp", F.to_timestamp("timestamp")) \
                  .drop("timestamp") \
                  .withColumnRenamed("timestamp_temp", "timestamp")
         if not isinstance(sdf.schema["timestamp"].dataType, TimestampType):
              logger.error("Failed to convert 'timestamp' column to TimestampType.")
              raise TypeError("Column 'timestamp' must be TimestampType for time feature extraction.")
         logger.info("'timestamp' 已转换为 TimestampType.")

    sdf_with_features = sdf.withColumn("year", F.year("timestamp")) \
                           .withColumn("month", F.month("timestamp")) \
                           .withColumn("day", F.dayofmonth("timestamp")) \
                           .withColumn("dayofweek", F.dayofweek("timestamp")) \
                           .withColumn("dayofyear", F.dayofyear("timestamp")) \
                           .withColumn("hour", F.hour("timestamp"))
                           # 可以添加更多特征，如 weekofyear 等
                           # .withColumn("weekofyear", F.weekofyear("timestamp"))

    logger.success("成功添加时间特征: year, month, day, dayofweek, dayofyear, hour")
    sdf_with_features.printSchema() # 显示添加特征后的 schema
    return sdf_with_features

def add_rolling_features_spark(sdf, target_col="y", window_sizes=[3, 6, 12, 24, 168], stats=["mean", "stddev", "min", "max"]):
    """Adds rolling window statistics for a target column using Spark Window functions."""
    if not window_sizes or not stats:
        logger.info("No window sizes or statistics specified. Skipping rolling feature creation.")
        return sdf
    if target_col not in sdf.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame. Skipping rolling features.")
        return sdf

    logger.info(f"开始添加 '{target_col}' 的滚动统计特征...")
    logger.info(f"窗口大小: {window_sizes}")
    logger.info(f"统计指标: {stats}")

    # 基础窗口定义：按 unique_id 分区，按 timestamp 升序排序
    if not isinstance(sdf.schema["timestamp"].dataType, TimestampType):
         logger.error("Cannot create rolling features: 'timestamp' column is not TimestampType.")
         raise TypeError("Column 'timestamp' must be TimestampType for rolling feature calculation.")

    base_window_spec = Window.partitionBy("unique_id").orderBy("timestamp")

    sdf_with_rolling = sdf
    stat_functions = {
        "mean": F.mean,
        "stddev": F.stddev_samp, # Use sample standard deviation
        "min": F.min,
        "max": F.max
        # Could add sum, count, etc. if needed
    }

    valid_stats = [s for s in stats if s in stat_functions]
    if not valid_stats:
        logger.warning("No valid statistics requested. Skipping rolling features.")
        return sdf
    if len(valid_stats) < len(stats):
        ignored_stats = set(stats) - set(valid_stats)
        logger.warning(f"Ignoring unsupported statistics: {ignored_stats}")


    for window_size in window_sizes:
        # 定义滚动窗口框架: 包括当前行在内的前 window_size 行
        # 对于预测 t 时刻的值，使用 t 时刻及之前的 window_size-1 行数据是合理的
        # 如果要严格预测 t 时刻，仅使用 t-1 及之前的数据，则用 rowsBetween(-window_size, -1)
        # 这里我们包含当前行：rowsBetween(-(window_size - 1), 0)
        # 注意: 这假设了每个 unique_id 的时间序列是连续的（每小时都有记录）
        rolling_window_spec = base_window_spec.rowsBetween(-(window_size - 1), 0)

        for stat_name in valid_stats:
            stat_func = stat_functions[stat_name]
            col_name = f"{target_col}_rolling_{stat_name}_{window_size}h"
            logger.debug(f"Adding rolling feature: {col_name}")
            sdf_with_rolling = sdf_with_rolling.withColumn(
                col_name,
                stat_func(target_col).over(rolling_window_spec)
            )

    logger.success(f"成功添加滚动统计特征 (指标: {valid_stats}, 窗口: {window_sizes}h)")
    sdf_with_rolling.printSchema() # 显示添加特征后的 schema
    return sdf_with_rolling

def handle_missing_values_spark(sdf, max_lag=168):
    """Handles missing values in the Spark DataFrame after feature engineering."""
    logger.info("开始处理缺失值...")

    initial_count = sdf.count()
    logger.info(f"处理前总行数: {initial_count:,}")

    # 1. 删除目标变量 y 为 null 的行
    sdf_cleaned = sdf.dropna(subset=["y"])
    count_after_y_drop = sdf_cleaned.count()
    logger.info(f"删除 'y' 为 null 的行后，剩余行数: {count_after_y_drop:,} (删除了 {initial_count - count_after_y_drop:,} 行)")
    if count_after_y_drop == 0:
        logger.warning("删除 'y' 为 null 的行后，DataFrame 为空！")
        return sdf_cleaned # Early exit if empty

    # 2. 删除每个 unique_id 的初始行 (基于最大滞后)
    # 使用 y_lag_{max_lag} 作为标记，因为它会在前 max_lag 行是 null
    lag_col_to_check = f"y_lag_{max_lag}"
    if lag_col_to_check in sdf_cleaned.columns:
        sdf_cleaned = sdf_cleaned.dropna(subset=[lag_col_to_check])
        count_after_lag_drop = sdf_cleaned.count()
        logger.info(f"删除每个 unique_id 前 {max_lag} 小时的记录后，剩余行数: {count_after_lag_drop:,} (又删除了 {count_after_y_drop - count_after_lag_drop:,} 行)")
        if count_after_lag_drop == 0:
             logger.warning(f"删除前 {max_lag} 小时记录后，DataFrame 为空！")
             return sdf_cleaned
    else:
        logger.warning(f"未找到列 '{lag_col_to_check}'，跳过基于最大滞后的初始行删除。")


    # 识别需要填充的列
    cols_to_fill_zero = []
    cols_to_fill_unknown = []
    cols_with_remaining_nulls = []

    logger.info("检查剩余列中的缺失值并确定填充策略...")
    for col_name, col_type in sdf_cleaned.dtypes:
        # 跳过关键列和目标列
        if col_name in ["unique_id", "timestamp", "y", lag_col_to_check]:
            continue

        # 检查是否有 null
        null_count = sdf_cleaned.where(F.col(col_name).isNull()).count()

        if null_count > 0:
            logger.info(f"列 '{col_name}' (类型: {col_type}) 存在 {null_count:,} 个 null 值。")
            cols_with_remaining_nulls.append(col_name)
            if isinstance(sdf_cleaned.schema[col_name].dataType, NumericType):
                # 对所有数值型列（包括天气、滚动特征等）用 0 填充
                cols_to_fill_zero.append(col_name)
                logger.info(f"  -> 将使用 0 填充。")
            elif isinstance(sdf_cleaned.schema[col_name].dataType, StringType):
                # 对字符型列（如 building_class）用 'Unknown' 填充
                if col_name == "building_class": # Specific handling for building_class
                    cols_to_fill_unknown.append(col_name)
                    logger.info(f"  -> 将使用 'Unknown' 填充。")
                else:
                     logger.warning(f"  -> 字符型列 '{col_name}' 有缺失值，但未指定填充策略，将保留 null。")
            else:
                logger.warning(f"  -> 列 '{col_name}' 类型为 {col_type}，未指定填充策略，将保留 null。")
        # else:
            # logger.debug(f"列 '{col_name}' 无缺失值。")


    # 3. 填充分类特征 (building_class)
    if cols_to_fill_unknown:
        logger.info(f"使用 'Unknown' 填充以下字符型列: {cols_to_fill_unknown}")
        sdf_cleaned = sdf_cleaned.fillna("Unknown", subset=cols_to_fill_unknown)

    # 4. 填充剩余的数值特征 (天气, 滚动特征等)
    if cols_to_fill_zero:
        logger.info(f"使用 0 填充以下数值型列: {cols_to_fill_zero}")
        sdf_cleaned = sdf_cleaned.fillna(0.0, subset=cols_to_fill_zero) # Use 0.0 for Double/Float types

    # 最终检查
    final_count = sdf_cleaned.count()
    logger.info(f"处理后最终行数: {final_count:,} (总共删除了 {initial_count - final_count:,} 行)")
    logger.info("最终缺失值检查:")
    null_counts_final = sdf_cleaned.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in sdf_cleaned.columns]).first().asDict()
    has_nulls = False
    for col, count in null_counts_final.items():
        if count > 0:
            logger.warning(f"  - 列 '{col}' 仍然包含 {count:,} 个 null 值！")
            has_nulls = True
    if not has_nulls:
        logger.success("所有检查过的列均无剩余缺失值。")
    else:
        logger.warning("数据中仍存在缺失值，请检查填充策略或源数据。")


    return sdf_cleaned

# --- 其他特征工程函数将在此处添加 ---
# def add_lag_features_spark(sdf, lags=[1, 2, 3, 24]): ...
# def encode_categorical_features_spark(sdf): ...

# ======================================================================
# ==                   Main Execution Function                      ==
# ======================================================================
def run_feature_engineering_spark():
    """Loads merged data, performs feature engineering using Spark, and saves the results."""
    logger.info("=====================================================")
    logger.info("=== 开始执行 特征工程脚本 (Spark) ===")
    logger.info("=====================================================")
    start_run_time = time.time()
    spark = None

    try:
        # --- 创建 SparkSession ---
        logger.info("创建 SparkSession...")
        import psutil
        total_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        # 特征工程可能涉及窗口函数等，内存需求可能较大
        driver_memory = max(4, int(total_memory_gb * 0.5 + 0.5))
        executor_memory = max(4, int(total_memory_gb * 0.5 + 0.5))
        logger.info(
            f"系统可用内存: {total_memory_gb:.2f}GB, 设置驱动器内存: {driver_memory}g, 执行器内存: {executor_memory}g")

        spark = SparkSession.builder \
            .appName("ElectricityDemandFeatureEngineering") \
            .config("spark.driver.memory", f"{driver_memory}g") \
            .config("spark.executor.memory", f"{executor_memory}g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.execution.pyarrow.fallback.enabled", "true") \
            .getOrCreate()

        logger.info("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # --- 步骤 1: 加载合并后的数据 ---
        logger.info(f"--- 步骤 1: 加载合并后的数据 {merged_data_path} ---")
        try:
            sdf = spark.read.parquet(str(merged_data_path))
            logger.info("合并数据加载成功。")
            logger.info("原始 Schema:")
            sdf.printSchema()
            # logger.info(f"数据行数 (估计): {sdf.count()}") # Count is expensive
        except Exception as load_e:
            logger.exception(f"加载 Parquet 文件失败: {load_e}")
            raise

        # --- 步骤 2: 执行特征工程 ---
        logger.info("--- 步骤 2: 执行特征工程 ---")

        # 2.1 添加时间特征
        sdf_features = add_time_features_spark(sdf)

        # 2.2 添加滚动特征
        logger.info("--- 步骤 2.2: 添加滚动统计特征 ---")
        window_sizes = [3, 6, 12, 24, 168]
        stats_to_compute = ["mean", "stddev", "min", "max"]
        sdf_features = add_rolling_features_spark(
            sdf_features,
            target_col="y",
            window_sizes=window_sizes,
            stats=stats_to_compute
        )

        # 2.3 处理缺失值
        logger.info("--- 步骤 2.3: 处理缺失值 ---")
        # 使用最长的滞后/滚动窗口期作为删除初始行的依据
        max_hist_window = max(window_sizes) # Find the max window used for now
        logger.info(f"将基于最大历史窗口 {max_hist_window}h 来删除初始行并填充缺失值。")
        sdf_features = handle_missing_values_spark(sdf_features, max_lag=max_hist_window)

        # --- 后续步骤将在此处添加 ---
        # logger.info("--- 步骤 2.4: 编码分类特征 ---")
        # sdf_features = encode_categorical_features_spark(sdf_features)

        # --- 步骤 3: 保存特征工程后的数据 ---
        logger.info(f"--- 步骤 3: 保存特征工程后的数据到 {features_output_path} ---")
        try:
            logger.info("开始写入 Parquet 文件...")
            # 考虑重新分区以优化后续读取或写入大小适中的文件
            # num_partitions = sdf_features.rdd.getNumPartitions() # 获取当前分区数
            # logger.info(f"当前分区数: {num_partitions}. 考虑根据数据大小调整分区。")
            # sdf_features = sdf_features.repartition(200) # 或 coalesce

            sdf_features.write.mode("overwrite").parquet(str(features_output_path))
            logger.success(f"成功保存特征工程后的数据到：{features_output_path}")
        except Exception as save_e:
            logger.exception(f"保存特征工程后的 Spark 数据时出错：{save_e}")
            raise

        logger.info("=====================================================")
        logger.info("===      特征工程脚本 (Spark) 执行完毕       ===")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"特征工程过程中发生严重错误: {e}")
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
                logger.error(f"停止 SparkSession 时发生错误: {stop_e}")
        else:
            logger.info("SparkSession 未成功初始化或已停止。")

        end_run_time = time.time()
        logger.info(
            f"--- Spark 特征工程脚本总执行时间: {end_run_time - start_run_time:.2f} 秒 ---")

if __name__ == "__main__":
    try:
        run_feature_engineering_spark()
    except Exception as e:
        # 主函数已有详细日志记录
        sys.exit(1)
