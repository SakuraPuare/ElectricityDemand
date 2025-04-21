import sys
import time
from pathlib import Path

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, TimestampType, DoubleType, IntegerType, LongType, FloatType  # 用于检查列类型
from pyspark.sql.window import Window
from pyspark.sql.dataframe import DataFrame
from pyspark.storagelevel import StorageLevel  # 显式导入

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
log_prefix = Path(__file__).stem  # 使用 3_run_feature_engineering 作为前缀
try:
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
except NameError:  # If setup_logger wasn't imported successfully
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

logger.info(f"项目根目录：{project_root}")
logger.info(f"数据目录：{data_dir}")
logger.info(f"日志目录：{logs_dir}")
logger.info(f"绘图目录：{plots_dir}")

# --- 数据文件路径 ---
merged_data_path = data_dir / "merged_data.parquet"  # 输入：合并后的数据
features_output_path = data_dir / "features.parquet"  # 输出：特征工程后的数据

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
        logger.warning(
            "Input 'timestamp' is not TimestampType. Attempting conversion.")
        sdf = sdf.withColumn("timestamp_temp", F.to_timestamp("timestamp")) \
                 .drop("timestamp") \
                 .withColumnRenamed("timestamp_temp", "timestamp")
        if not isinstance(sdf.schema["timestamp"].dataType, TimestampType):
            logger.error(
                "Failed to convert 'timestamp' column to TimestampType.")
            raise TypeError(
                "Column 'timestamp' must be TimestampType for time feature extraction.")
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
    sdf_with_features.printSchema()  # 显示添加特征后的 schema
    return sdf_with_features


def add_rolling_features_spark(sdf, target_col="y", window_sizes=[3, 6, 12, 24, 168], stats=["mean", "stddev", "min", "max"]):
    """Adds rolling window statistics for a target column using Spark Window functions."""
    if not window_sizes or not stats:
        logger.info(
            "No window sizes or statistics specified. Skipping rolling feature creation.")
        return sdf
    if target_col not in sdf.columns:
        logger.error(
            f"Target column '{target_col}' not found in DataFrame. Skipping rolling features.")
        return sdf

    logger.info(f"开始添加 '{target_col}' 的滚动统计特征...")
    logger.info(f"窗口大小: {window_sizes}")
    logger.info(f"统计指标: {stats}")

    # 基础窗口定义：按 unique_id 分区，按 timestamp 升序排序
    if not isinstance(sdf.schema["timestamp"].dataType, TimestampType):
        logger.error(
            "Cannot create rolling features: 'timestamp' column is not TimestampType.")
        raise TypeError(
            "Column 'timestamp' must be TimestampType for rolling feature calculation.")

    base_window_spec = Window.partitionBy("unique_id").orderBy("timestamp")

    sdf_with_rolling = sdf
    stat_functions = {
        "mean": F.mean,
        "stddev": F.stddev_samp,  # Use sample standard deviation
        "min": F.min,
        "max": F.max
        # Could add sum, count, etc. if needed
    }

    valid_stats = [s for s in stats if s in stat_functions]
    if not valid_stats:
        logger.warning(
            "No valid statistics requested. Skipping rolling features.")
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
        rolling_window_spec = base_window_spec.rowsBetween(
            -(window_size - 1), 0)

        for stat_name in valid_stats:
            stat_func = stat_functions[stat_name]
            col_name = f"{target_col}_rolling_{stat_name}_{window_size}h"
            logger.debug(f"Adding rolling feature: {col_name}")
            sdf_with_rolling = sdf_with_rolling.withColumn(
                col_name,
                stat_func(target_col).over(rolling_window_spec)
            )

    logger.success(f"成功添加滚动统计特征 (指标: {valid_stats}, 窗口: {window_sizes}h)")
    sdf_with_rolling.printSchema()  # 显示添加特征后的 schema
    return sdf_with_rolling


def handle_missing_values_spark(
    sdf: DataFrame, max_window: int, target_col: str = "y"
) -> DataFrame:
    """
    处理 Spark DataFrame 中的缺失值：
    1. 删除目标列 ('y') 为 null 的行。
    2. 基于最大滚动窗口期删除初始行（这些行的滚动特征为 null）。
    3. （可选）对其他特征列进行填充（当前未实现，因为天气数据无缺失）。

    Args:
        sdf (DataFrame): 输入的 Spark DataFrame，包含滚动特征。
        max_window (int): 计算滚动特征时使用的最大窗口大小 (小时)。
        target_col (str): 目标列名 (默认为 'y')。

    Returns:
        DataFrame: 处理缺失值后的 DataFrame。
    """
    logger.info("开始处理缺失值...")
    initial_count = sdf.count()
    logger.info(f"处理前总行数: {initial_count}")

    # 1. 删除目标列 ('y') 为 null 的行
    logger.info(f"检查列 '{target_col}' 的缺失值...")
    sdf_no_y_null = sdf.filter(F.col(target_col).isNotNull())
    count_after_y_drop = sdf_no_y_null.count()
    deleted_y_null = initial_count - count_after_y_drop
    logger.info(
        f"删除 '{target_col}' 为 null 的行后，剩余行数: {count_after_y_drop} (删除了 {deleted_y_null} 行)"
    )

    # 2. 基于最大滚动窗口删除初始行为 null 的行
    rolling_feature_col = f"{target_col}_rolling_mean_{max_window}h"
    if rolling_feature_col not in sdf_no_y_null.columns:
        logger.warning(
            f"列 '{rolling_feature_col}' 不存在，无法基于滚动窗口删除初始行。"
        )
        sdf_no_initial_window = sdf_no_y_null
    else:
        logger.info(f"检查列 '{rolling_feature_col}' 的缺失值以删除初始窗口期...")
        sdf_no_initial_window = sdf_no_y_null.filter(
            F.col(rolling_feature_col).isNotNull()
        )
        # count_after_init_drop = sdf_no_initial_window.count() # <--- OOM发生点
        # deleted_init_rows = count_after_y_drop - count_after_init_drop
        # logger.info(
        #     f"删除基于滚动窗口 '{rolling_feature_col}' 的初始 null 行后，剩余行数: {count_after_init_drop} (删除了 {deleted_init_rows} 行)"
        # ) # <--- 注释掉相关日志

    # 3. （可选）处理其他特征列的缺失值 (此处假设天气数据等无缺失)
    # TODO: 如果其他列有缺失，在这里添加填充逻辑 (e.g., ffill, mean/median imputation)
    numeric_cols = [
        f.name
        for f in sdf_no_initial_window.schema.fields
        if isinstance(f.dataType, (DoubleType, IntegerType, LongType, FloatType))
           and f.name not in ["y", "unique_id", "location_id", "cluster_size"] # 排除非特征数值列和目标列
    ]
    # logger.info(f"检查数值特征列的缺失值: {numeric_cols}")
    # null_counts = sdf_no_initial_window.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols]).first().asDict()
    # cols_with_nulls = {k: v for k, v in null_counts.items() if v > 0}
    # if cols_with_nulls:
    #     logger.warning(f"以下数值特征列存在缺失值，将使用 0 填充: {cols_with_nulls}")
    #     sdf_filled = sdf_no_initial_window.fillna(0, subset=list(cols_with_nulls.keys()))
    # else:
    #     logger.info("数值特征列无缺失值，无需填充。")
    #     sdf_filled = sdf_no_initial_window

    # 暂时不进行数值列填充，因为之前的分析显示其他数值列无缺失
    sdf_filled = sdf_no_initial_window

    # 持久化处理后的数据，便于后续计算
    logger.info("持久化处理缺失值后的 DataFrame...")
    sdf_filled.persist(StorageLevel.MEMORY_AND_DISK)
    final_count = sdf_filled.count() # 在持久化后进行 count 通常更安全
    logger.info(f"持久化完成，最终处理后的数据行数: {final_count}")
    # deleted_init_rows = count_after_y_drop - final_count # 可以推算出删除的初始行数
    # logger.info(f"基于滚动窗口 '{rolling_feature_col}' 删除的初始 null 行数: {deleted_init_rows}")


    logger.success("缺失值处理完成。")
    return sdf_filled

# --- 其他特征工程函数将在此处添加 ---
# def add_lag_features_spark(sdf, lags=[1, 2, 3, 24]): ...
# def encode_categorical_features_spark(sdf): ...

# ======================================================================
# ==                   Main Execution Function                      ==
# ======================================================================


def run_feature_engineering_spark(use_sampling: bool = False, sample_fraction: float = 0.0001):
    """
    使用 Spark 执行完整的特征工程流程。

    Args:
        use_sampling (bool): 是否对合并后的数据进行采样以进行特征工程。
        sample_fraction (float): 如果 use_sampling 为 True，使用的采样比例。
    """
    logger.info("=" * 53)
    logger.info("=== 开始执行 特征工程脚本 (Spark) ===")
    logger.info("=" * 53)

    start_time = time.time()

    merged_data_path = data_dir / "merged_data.parquet"
    if use_sampling:
        output_feature_path = data_dir / f"features_sampled_{sample_fraction}.parquet"
    else:
        output_feature_path = data_dir / "features.parquet"

    logger.info(f"输入合并数据路径: {merged_data_path}")
    logger.info(f"输出特征数据路径: {output_feature_path}")

    spark = None
    try:
        logger.info("创建 SparkSession...")
        spark = create_spark_session("ElectricityDemandFeatureEngineering")
        if not spark:
            logger.error("无法创建 SparkSession。")
            return
        logger.success("SparkSession 创建成功。")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")


        # --- 步骤 1: 加载合并后的数据 ---
        logger.info(f"--- 步骤 1: 加载合并后的数据 {merged_data_path} ---")
        if not merged_data_path.exists():
             logger.error(f"错误: 合并后的数据文件未找到于 {merged_data_path}")
             raise FileNotFoundError(f"合并后的数据文件未找到于 {merged_data_path}")

        sdf_merged = spark.read.parquet(str(merged_data_path))
        logger.info("合并数据加载成功。")
        logger.info("原始 Schema:")
        sdf_merged.printSchema()
        # sdf_merged.show(5)

        if use_sampling:
            logger.warning(f"--- 注意: 将在 {sample_fraction*100:.4f}% 的抽样数据上运行特征工程 ---")
            row_count = sdf_merged.count()
            logger.info(f"原始数据行数: {row_count}")
            sdf = sdf_merged.sample(fraction=sample_fraction, seed=42)
            sampled_count = sdf.count()
            logger.info(f"抽样后数据行数: {sampled_count} (目标比例: {sample_fraction}, 实际比例: {sampled_count/row_count if row_count > 0 else 0:.6f})")
            logger.info(f"抽样特征数据将保存到: {output_feature_path}")
        else:
            logger.info("--- 注意: 将在完整数据集上运行特征工程 ---")
            sdf = sdf_merged
            logger.info(f"全量特征数据将保存到: {output_feature_path}")


        # --- 步骤 2: 执行特征工程 ---
        logger.info("--- 步骤 2: 执行特征工程 ---")

        # --- 步骤 2.1: 添加时间特征 ---
        sdf_with_time = add_time_features_spark(sdf)

        # --- 步骤 2.2: 添加滚动统计特征 ---
        logger.info("--- 步骤 2.2: 添加滚动统计特征 ---")
        # 窗口大小 (小时): 3h, 6h, 12h, 1天, 2天, 1周
        # 注意：确保窗口单位与数据频率匹配或可转换
        # 假设数据主要是 30min 或 1h，选择小时窗口
        window_sizes_hours = [3, 6, 12, 24, 48, 168] # 168 hours = 7 days
        stats_to_compute = ["mean", "stddev", "min", "max"]
        sdf_with_rolling = add_rolling_features_spark(
            sdf_with_time,
            target_col="y",
            window_sizes=window_sizes_hours,
            stats=stats_to_compute
        )

        # --- 步骤 2.3: 处理缺失值 ---
        logger.info("--- 步骤 2.3: 处理缺失值 ---")
        max_window = max(window_sizes_hours)
        logger.info(f"将基于最大历史窗口 {max_window}h 来删除初始行并处理缺失值。")
        sdf_processed = handle_missing_values_spark(
            sdf_with_rolling, max_window=max_window, target_col="y"
        )

        # --- 步骤 2.4: （可选）添加滞后特征 ---
        # logger.info("--- 步骤 2.4: 添加滞后特征 ---")
        # lag_hours = [24, 48, 168] # 1天, 2天, 1周前的 y 值
        # sdf_with_lags = add_lag_features_spark(
        #     sdf_processed,
        #     target_col="y",
        #     lag_hours=lag_hours,
        #     partition_col="unique_id",
        #     order_col="timestamp"
        # )
        # logger.success("成功添加滞后特征。")
        # sdf_final = sdf_with_lags
        sdf_final = sdf_processed # 如果不加滞后特征

        # --- 步骤 2.5: （可选）其他特征 (如交互特征、天气特征的滚动/滞后) ---
        # logger.info("--- 步骤 2.5: 添加其他特征 ---")
        # 例如: 添加温度的滚动平均
        # sdf_final = add_rolling_features_spark(sdf_final, target_col="temperature_2m", window_sizes_hours=[6, 12, 24], stats=["mean"], partition_col="unique_id", order_col="timestamp")
        # logger.success("成功添加温度滚动特征。")


        # --- 步骤 3: 显示最终 Schema 和一些样本数据 ---
        logger.info("--- 步骤 3: 显示最终 Schema 和样本数据 ---")
        logger.info("最终特征 Schema:")
        sdf_final.printSchema()
        logger.info("最终特征数据样本 (前 10 行):")
        sdf_final.show(10, truncate=False)

        # --- 步骤 4: 保存特征数据 ---
        logger.info(f"--- 步骤 4: 保存特征数据到 {output_feature_path} ---")
        start_write = time.time()
        (
            sdf_final.write.mode("overwrite")
            .partitionBy("year", "month") # 按年月分区写入，提高后续读取效率
            .parquet(str(output_feature_path))
        )
        write_time = time.time() - start_write
        logger.success(f"特征数据成功保存到 {output_feature_path} (耗时: {write_time:.2f} 秒)")

        # 验证写入的数据
        # logger.info("验证写入的数据...")
        # try:
        #     df_check = spark.read.parquet(str(output_feature_path))
        #     final_count = df_check.count()
        #     logger.info(f"成功读取已保存的特征数据，总行数: {final_count}")
        # except Exception as e:
        #     logger.error(f"验证写入数据时出错: {e}")


    except Exception as e:
        logger.error(f"特征工程步骤中发生错误: {e}")
        logger.exception("详细错误信息:") # 记录完整的 traceback
        # 尝试取消持久化的数据
        if 'sdf_filled' in locals() and sdf_processed.is_cached:
            try:
                sdf_processed.unpersist()
                logger.info("已取消持久化的 sdf_processed。")
            except Exception as unpersist_e:
                logger.warning(f"取消持久化 sdf_processed 时出错: {unpersist_e}")

        raise # 重新抛出异常，以便上层或日志钩子捕获

    finally:
        if spark:
            spark.stop()
            logger.info("SparkSession 已停止。")

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"--- Spark 特征工程脚本总执行时间: {total_time:.2f} 秒 ---")


if __name__ == "__main__":
    setup_logger("3_run_feature_engineering")

    logger.info(f"项目根目录：{project_root}")
    logger.info(f"数据目录：{data_dir}")
    logger.info(f"日志目录：{logs_dir}")
    logger.info(f"绘图目录：{plots_dir}")

    # --- 控制运行哪个部分 ---
    # run_eda = False
    # run_merge = False
    run_feature_eng = True
    use_sampling_fe = False # 控制特征工程是否使用抽样
    sampling_fraction_fe = 0.001 # 特征工程抽样比例 (如果 use_sampling_fe=True)

    # if run_eda:
    #     # 在小样本上运行 EDA
    #     run_eda_spark(sample_fraction=0.005, unique_id_sample_count=100)
    # if run_merge:
    #     # 合并完整数据
    #     run_merge_data_spark()
    if run_feature_eng:
        # 在完整数据或抽样数据上运行特征工程
        run_feature_engineering_spark(use_sampling=use_sampling_fe, sample_fraction=sampling_fraction_fe)
