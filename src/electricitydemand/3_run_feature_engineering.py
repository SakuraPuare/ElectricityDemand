import os
import sys
import time
from pathlib import Path

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, TimestampType  # 用于检查列类型
from pyspark.sql.window import Window
from pyspark.sql.storage import StorageLevel

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

@logger.catch(reraise=True)
def handle_missing_values_spark(sdf: DataFrame, window_hours: int) -> DataFrame:
    """
    Handles missing values in the feature DataFrame using Spark.

    1. Removes rows where the target 'y' is null.
    2. Removes initial rows within the rolling window period for each time series.
    3. Identifies remaining nulls and applies specific strategies:
       - Fills rolling stddev nulls with 0.0.
       - Removes rows where location_id is null.
       - Reports any other unexpected nulls.

    Args:
        sdf (DataFrame): Input Spark DataFrame with features.
        window_hours (int): The maximum rolling window size used, to determine initial rows to drop.

    Returns:
        DataFrame: Spark DataFrame with missing values handled.
    """
    logger.info("开始处理缺失值...")
    initial_count = sdf.count()
    logger.info(f"处理前总行数: {initial_count:,}")

    # 1. 删除 target 'y' 为 null 的行
    logger.info("检查列 'y' 的缺失值...")
    sdf_no_y_null = sdf.filter(F.col("y").isNotNull())
    count_after_y_drop = sdf_no_y_null.count()
    rows_dropped_y = initial_count - count_after_y_drop
    logger.info(f"删除 'y' 为 null 的行后，剩余行数: {count_after_y_drop:,} (删除了 {rows_dropped_y:,} 行)")
    sdf = sdf_no_y_null # Update sdf

    # 2. 删除每个时间序列的初始窗口期数据
    # 使用滚动均值列来识别（如果滚动特征计算正确，应该有 null）
    # 注意：如果一个序列的初始值全部相同，stddev 可能是 0 或 null，mean 可能非 null
    # 更健壮的方法可能是使用 row_number，但这里暂时沿用基于 rolling mean 的方法
    rolling_col_for_init_drop = f"y_rolling_mean_{window_hours}h"
    if rolling_col_for_init_drop not in sdf.columns:
        logger.error(f"用于删除初始窗口的列 '{rolling_col_for_init_drop}' 不存在！跳过此步骤。")
    else:
        logger.info(f"检查列 '{rolling_col_for_init_drop}' 的缺失值以删除初始窗口期...")
        # 确保在删除前缓存，避免重复计算 filter
        sdf.persist(StorageLevel.MEMORY_AND_DISK)
        sdf_no_initial_window = sdf.filter(F.col(rolling_col_for_init_drop).isNotNull())
        count_after_init_drop = sdf_no_initial_window.count()
        rows_dropped_init = count_after_y_drop - count_after_init_drop # Compare with count after y drop
        logger.info(f"删除每个 unique_id 前 {window_hours} 小时的记录（基于 '{rolling_col_for_init_drop}' 为 null）后，剩余行数: {count_after_init_drop:,} (又删除了 {rows_dropped_init:,} 行)")
        sdf = sdf_no_initial_window # Update sdf
        # Check if the initial drop removed significant rows, if not, warn.
        if rows_dropped_init == 0 and initial_count > 0:
             logger.warning(f"基于 '{rolling_col_for_init_drop}' 的初始窗口删除未移除任何行。请检查滚动特征计算或窗口期是否足够长。")
        # Unpersist the intermediate sdf if needed, but let's keep it for the next step
        # sdf.unpersist() # Unpersist previous sdf version


    # 3. 处理 location_id 为 Null 的情况 (删除这些行)
    logger.info("检查并删除 'location_id' 为 null 的行...")
    count_before_loc_drop = sdf.count()
    sdf_valid_location = sdf.filter(F.col("location_id").isNotNull())
    final_count_after_loc_drop = sdf_valid_location.count()
    rows_dropped_loc = count_before_loc_drop - final_count_after_loc_drop
    logger.info(f"删除 'location_id' 为 null 的行后，剩余行数: {final_count_after_loc_drop:,} (删除了 {rows_dropped_loc:,} 行)")
    sdf = sdf_valid_location # Update sdf
    if rows_dropped_loc > 0:
         logger.success(f"成功删除 {rows_dropped_loc:,} 行因 location_id 为 null 的记录。")

    # 4. 处理滚动标准差 (stddev) 特征的 Null 值 (填充为 0.0)
    stddev_cols = [c for c in sdf.columns if "y_rolling_stddev" in c]
    if stddev_cols:
        logger.info("检查并填充滚动标准差列的 Null 值为 0.0...")
        null_counts_stddev_before = {}
        for col_name in stddev_cols:
             # 检查是否存在 Null (耗时操作，可选)
             # null_count = sdf.where(F.col(col_name).isNull()).count()
             # if null_count > 0:
             #     null_counts_stddev_before[col_name] = null_count
             #     logger.warning(f"  - 列 '{col_name}' 将填充 {null_count} 个 null 值。")
             sdf = sdf.withColumn(col_name, F.when(F.col(col_name).isNull(), 0.0).otherwise(F.col(col_name)))
        logger.success(f"已将以下滚动标准差列中的 Null 值填充为 0.0: {stddev_cols}")
        # 可选：再次检查填充后是否还有 Null
        # for col_name in stddev_cols:
        #     null_count_after = sdf.where(F.col(col_name).isNull()).count()
        #     if null_count_after > 0:
        #         logger.error(f"  - 警告！列 '{col_name}' 在填充后仍有 {null_count_after} 个 null 值！")

    else:
        logger.info("未找到滚动标准差特征列，跳过填充步骤。")


    # 5. 最终检查并报告剩余的 Null 值
    logger.info("最终检查剩余列中的缺失值...")
    final_count = sdf.count()
    remaining_null_cols = []
    null_check_results = {}
    for col_name in sdf.columns:
        # Check for nulls more efficiently if possible, maybe aggregate?
        # For now, use count for simplicity
        null_count = sdf.where(F.col(col_name).isNull()).count()
        if null_count > 0:
            percentage = (null_count / final_count) * 100 if final_count > 0 else 0
            col_type = sdf.schema[col_name].dataType
            null_check_results[col_name] = (null_count, percentage, col_type)
            remaining_null_cols.append(col_name)
            logger.warning(f"  - 列 '{col_name}' (类型: {col_type}) 仍存在 {null_count:,} 个 null 值 ({percentage:.4f}%).")


    if not remaining_null_cols:
        logger.success("============================== 缺失值报告 ==============================")
        logger.success("所有预期的缺失值已处理完毕，未发现其他列存在 Null。")
        logger.success("======================================================================")
    else:
        logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 缺失值警告 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.error("在执行完所有处理步骤后，以下列仍包含 Null 值:")
        for col_name in remaining_null_cols:
             n_count, perc, c_type = null_check_results[col_name]
             logger.error(f"  - 列 '{col_name}' ({c_type}): {n_count:,} 个 null ({perc:.4f}%)")
        logger.error("请检查数据源或处理逻辑，这些 Null 值可能影响后续模型训练！")
        logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 根据需要，可以选择在这里抛出异常停止执行
        # raise ValueError("发现未处理的 Null 值，请检查日志。")


    logger.info(f"处理后最终行数: {final_count:,} (总共删除了 {initial_count - final_count:,} 行)")

    # Unpersist the final sdf before returning if it was persisted
    if sdf.is_cached:
        sdf.unpersist()

    return sdf

# --- 其他特征工程函数将在此处添加 ---
# def add_lag_features_spark(sdf, lags=[1, 2, 3, 24]): ...
# def encode_categorical_features_spark(sdf): ...

# ======================================================================
# ==                   Main Execution Function                      ==
# ======================================================================
def run_feature_engineering_spark(sample_fraction: float | None = None):
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
        driver_memory = max(4, int(total_memory_gb * 0.6 + 0.5)) # 稍微增加 Driver 内存比例
        executor_memory = max(4, int(total_memory_gb * 0.6 + 0.5)) # 稍微增加 Executor 内存比例
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
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "200") \
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
            # logger.info(f"数据行数 (加载后): {sdf.count():,}") # Count is expensive, maybe remove
        except Exception as load_e:
            logger.exception(f"加载 Parquet 文件失败: {load_e}")
            raise

        # 移除抽样逻辑 (注释掉或删除)
        # if sample_fraction is not None and 0 < sample_fraction < 1:
        #     logger.warning(f"--- 注意: 已启用数据抽样，仅使用 {sample_fraction:.2%} 的数据 ---")
        #     initial_row_count = sdf.count()
        #     sdf_processed = sdf.sample(withReplacement=False, fraction=sample_fraction, seed=42)
        #     sampled_row_count = sdf_processed.count()
        #     logger.info(f"抽样前行数: {initial_row_count:,}")
        #     logger.info(f"抽样后行数: {sampled_row_count:,}")
        #     # Define output path for sampled data
        #     features_output_path_final = features_output_path.parent / f"{features_output_path.stem}_sampled_{sample_fraction:.1f}{features_output_path.suffix}"
        # else:
        #     logger.info("--- 注意: 将在完整数据集上运行特征工程 ---")
        #     sdf_processed = sdf
        #     features_output_path_final = features_output_path

        sdf_processed = sdf # 直接使用全量数据
        features_output_path_final = features_output_path # 使用原始全量输出路径


        # --- 步骤 2: 执行特征工程 ---
        logger.info("--- 步骤 2: 执行特征工程 ---")
        try:
            # --- 步骤 2.1: 添加时间特征 ---
            sdf_processed = add_time_features_spark(sdf_processed)

            # --- 步骤 2.2: 添加滚动统计特征 ---
            logger.info("--- 步骤 2.2: 添加滚动统计特征 ---")
            window_hours_list = [3, 6, 12, 24, 48, 168] # 包含最大窗口 168h
            stats_list = ['mean', 'stddev', 'min', 'max']
            sdf_processed = add_rolling_features_spark(
                sdf_processed,
                target_col='y',
                window_sizes=window_hours_list,
                stats=stats_list,
                partition_col='unique_id',
                timestamp_col='timestamp'
            )

            # --- 步骤 2.3: 处理缺失值 ---
            logger.info("--- 步骤 2.3: 处理缺失值 ---")
            max_window = max(window_hours_list) if window_hours_list else 0
            logger.info(f"将基于最大历史窗口 {max_window}h 来删除初始行并处理缺失值。")
            sdf_processed = handle_missing_values_spark(sdf_processed, window_hours=max_window)


            # --- 步骤 2.4: (待添加) 分类特征编码 ---
            logger.warning("--- 步骤 2.4: (待添加) 分类特征编码 ---")
            # TODO: Add encoding for 'building_class' (e.g., OneHotEncoder)

            # --- 步骤 2.5: (待添加) 更多特征? ---
            # Lag features, interaction features etc.

        except Exception as fe_e:
            logger.exception(f"特征工程步骤中发生错误: {fe_e}")
            raise

        # --- 步骤 3: 保存特征工程后的数据 ---
        logger.info(f"--- 步骤 3: 保存特征工程后的数据到 {features_output_path_final} ---")
        try:
            logger.info("开始写入 Parquet 文件...")
            # Consider repartitioning or sorting before write if needed
            # sdf_processed.repartition(200).write.mode("overwrite").parquet(str(features_output_path_final))
            sdf_processed.write.mode("overwrite").parquet(str(features_output_path_final))
            logger.success(f"成功保存特征工程后的数据到：{features_output_path_final}")
        except Exception as save_e:
            logger.exception(f"保存特征工程数据时出错：{save_e}")
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
        # !!! 开发时可以启用抽样 !!!
        # run_feature_engineering_spark(use_sampling=True, sample_fraction=0.1)
        # !!! 最终运行时关闭抽样 !!!
        run_feature_engineering_spark(sample_fraction=None)
    except Exception as e:
        # 主函数已有详细日志记录
        sys.exit(1)
