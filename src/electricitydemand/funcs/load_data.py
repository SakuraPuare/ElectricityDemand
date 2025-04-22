import os
import sys

from loguru import logger  # 提前导入 logger

# 引入 Spark 相关库
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType
from pyspark.sql.utils import AnalysisException  # 用于捕获 Spark SQL 错误

# --- 项目设置 (路径和日志) ---
project_root = None
try:
    # 尝试标准的相对导入 (当作为包运行时)
    if __package__ and __package__.startswith('src.'):
        _script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(_script_path)))
        from .log_utils import setup_logger  # 相对导入
    else:
        raise ImportError(
            "Not running as a package or package structure mismatch.")

except (ImportError, ValueError, AttributeError, NameError):
    # 直接运行脚本或环境特殊的 fallback 逻辑
    try:
        _script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(_script_path)))
    except NameError:  # 如果 __file__ 未定义 (例如，交互式环境)
        project_root = os.getcwd()

    # 如果是直接运行，将项目根目录添加到 sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 现在绝对导入应该可用
    from src.electricitydemand.utils.log_utils import setup_logger

# --- 配置日志 ---
log_prefix = os.path.splitext(os.path.basename(__file__))[0]  # 从文件名自动获取前缀
logs_dir = os.path.join(project_root, 'logs')
# 设置日志级别为 INFO
setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
logger.info(f"项目根目录：{project_root}")
logger.info(f"日志目录：{logs_dir}")

# --- 数据文件路径 ---
data_dir = os.path.join(project_root, "data")  # 使用 project_root 确保路径正确
demand_path = os.path.join(data_dir, "demand_converted.parquet")
metadata_path = os.path.join(data_dir, "metadata.parquet")
weather_path = os.path.join(data_dir, "weather_converted.parquet")
logger.info(f"数据目录：{data_dir}")


# 修改函数以使用 SparkSession 加载数据


def load_datasets(spark: SparkSession):
    """使用 Spark 加载 Demand, Metadata, 和 Weather 数据集."""
    logger.info("开始使用 Spark 加载数据集...")
    try:
        ddf_demand = spark.read.parquet(demand_path)
        logger.info(f"成功加载 Demand 数据 (Spark): {demand_path}")

        ddf_metadata = spark.read.parquet(metadata_path)
        logger.info(f"成功加载 Metadata 数据 (Spark): {metadata_path}")

        ddf_weather = spark.read.parquet(weather_path)
        logger.info(f"成功加载 Weather 数据 (Spark): {weather_path}")

        logger.info("所有数据集加载完成 (Spark)。")
        return ddf_demand, ddf_metadata, ddf_weather
    except AnalysisException as e:  # 捕获 Spark 文件未找到等错误
        logger.error(f"Spark 数据文件加载失败：{e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载 Spark 数据集时发生错误：{e}")
        sys.exit(1)


# 修改函数以使用 Spark DataFrame API


def log_basic_info(ddf_demand, ddf_metadata, ddf_weather):
    """记录 Spark 数据集的基本信息：分区数、列名、行数、样本数据和模式。"""
    logger.info("--- 记录 Spark 数据集基本信息 ---")

    # 分区数 (可能因操作而变化) 和列名
    try:
        logger.info(
            f"Demand Spark DataFrame partitions: {ddf_demand.rdd.getNumPartitions()}, columns: {ddf_demand.columns}")
        logger.info(
            f"Metadata Spark DataFrame partitions: {ddf_metadata.rdd.getNumPartitions()}, columns: {ddf_metadata.columns}")
        logger.info(
            f"Weather Spark DataFrame partitions: {ddf_weather.rdd.getNumPartitions()}, columns: {ddf_weather.columns}")
    except Exception as e:
        logger.warning(f"获取分区数时出错 (可能是空 DataFrame?): {e}")

    # 计算行数 (Spark 的 .count() 是一个 Action，会触发计算)
    logger.info("--- 计算行数 (可能需要一些时间) ---")
    num_demand_rows = ddf_demand.count()
    num_metadata_rows = ddf_metadata.count()
    num_weather_rows = ddf_weather.count()
    logger.info(f"Demand 数据行数：{num_demand_rows:,}")
    logger.info(f"Metadata 数据行数：{num_metadata_rows:,}")
    logger.info(f"Weather 数据行数：{num_weather_rows:,}")

    # 查看数据样本 (使用 show)
    logger.info("--- 查看数据样本 (前 5 行) ---")
    logger.info("Demand head:")
    ddf_demand.show(5, truncate=False)  # 显示前 5 行，不截断列内容
    logger.info("Metadata head:")
    ddf_metadata.show(5, truncate=False)
    logger.info("Weather head:")
    ddf_weather.show(5, truncate=False)

    # 查看数据模式 (Schema)
    logger.info("--- 查看数据模式 (Schema) ---")
    logger.info("Demand schema:")
    ddf_demand.printSchema()
    logger.info("Metadata schema:")
    ddf_metadata.printSchema()
    logger.info("Weather schema:")
    ddf_weather.printSchema()

    return num_demand_rows, num_metadata_rows, num_weather_rows  # 返回行数供后续使用


# 修改函数以使用 Spark API


def check_missing_values(ddf_demand, ddf_metadata, ddf_weather, num_demand_rows, num_metadata_rows, num_weather_rows):
    """使用 Spark 检查并记录每个数据帧中的缺失值及其比例。"""
    logger.info("--- 使用 Spark 检查缺失值 ---")

    def calculate_missing(df, total_rows, df_name):
        if total_rows == 0:
            logger.warning(f"{df_name} 行数为 0，跳过缺失值检查。")
            return
        logger.info(f"{df_name} 缺失值统计：")
        # 构建聚合表达式列表
        missing_exprs = []
        # 获取 DataFrame 的 schema 以便检查类型
        schema = df.schema
        for field in schema.fields:
            col_name = field.name
            col_type = field.dataType
            # 检查列类型是否为数值型 (FloatType 或 DoubleType)
            is_numeric_for_nan = isinstance(
                col_type, (FloatType, DoubleType))  # 需要从 pyspark.sql.types 导入

            # 根据列类型构建条件
            if is_numeric_for_nan:
                condition = F.col(col_name).isNull() | F.isnan(col_name)
            else:
                condition = F.col(col_name).isNull()

            missing_exprs.append(
                F.sum(F.when(condition, 1).otherwise(0)).alias(
                    f"{col_name}_missing_count")
            )

        # .first() 触发计算并返回 Row 对象
        missing_counts_df = df.agg(*missing_exprs).first()

        if missing_counts_df:
            missing_info = []
            for c in df.columns:
                count_col_name = f"{c}_missing_count"
                # 检查 count_col_name 是否实际存在于 Row 对象中 (以防万一)
                if count_col_name in missing_counts_df.asDict():
                    count = missing_counts_df[count_col_name]
                    percentage = (count / total_rows *
                                  100) if total_rows > 0 else 0
                    missing_info.append(f"  {c}: {count} ({percentage:.2f}%)")
                else:
                    logger.warning(
                        f"列 '{c}' 的缺失计数列 '{count_col_name}' 未在聚合结果中找到。")
            logger.info("\n" + "\n".join(missing_info))
        else:
            logger.warning(f"无法计算 {df_name} 的缺失值统计。")

    calculate_missing(ddf_demand, num_demand_rows, "Demand")
    calculate_missing(ddf_metadata, num_metadata_rows, "Metadata")
    calculate_missing(ddf_weather, num_weather_rows, "Weather")


# 修改函数以使用 Spark API (注意：精确计数在 Spark 中可能很昂贵)


def check_duplicates(ddf_demand, ddf_metadata, ddf_weather):
    """使用 Spark 检查并记录数据帧中的重复值。"""
    logger.info("--- 使用 Spark 检查重复值 ---")

    def check_df_duplicates(df, key_columns, df_name):
        logger.info(f"检查 {df_name} 基于 {key_columns} 的重复值...")

        # --- 新增：检查 DataFrame 是否为空 ---
        if df.isEmpty():
            logger.info(f"{df_name} DataFrame 为空，跳过重复值检查。")
            return
        # --- 结束新增 ---

        duplicate_counts = df.groupBy(
            key_columns).count().where(F.col('count') > 1)
        num_duplicate_groups = duplicate_counts.count()  # 计算有多少组 key 是重复的
        if num_duplicate_groups > 0:
            try:
                # --- 使用 try-except 包裹可能因空 DataFrame 导致 first() 出错的操作 ---
                total_duplicate_rows_result = duplicate_counts.agg(
                    F.sum('count')).first()
                if total_duplicate_rows_result and total_duplicate_rows_result[0] is not None:
                    # 计算重复组的总行数
                    total_duplicate_rows = total_duplicate_rows_result[0]
                    logger.warning(
                        f"{df_name} 数据中发现 {num_duplicate_groups} 组基于 {key_columns} 的重复记录。")
                    logger.warning(f"重复组的总行数约为：{total_duplicate_rows}")
                    logger.warning("重复组样本 (前 5 组):")
                    duplicate_counts.show(5, truncate=False)
                else:
                    # This might happen if duplicate_counts becomes empty after the initial count due to race conditions or complex plans
                    logger.info(
                        f"{df_name} 数据中未发现基于 {key_columns} 的重复记录 (聚合结果为空)。")
            except Exception as e:
                logger.error(f"计算 {df_name} 重复总行数时出错：{e}")
        else:
            logger.info(f"{df_name} 数据中未发现基于 {key_columns} 的重复记录。")

    # Demand 重复值 (基于 unique_id 和 timestamp)
    if 'unique_id' in ddf_demand.columns and 'timestamp' in ddf_demand.columns:
        check_df_duplicates(ddf_demand, ['unique_id', 'timestamp'], "Demand")
    else:
        logger.warning(
            "Demand DataFrame 缺少 'unique_id' 或 'timestamp' 列，跳过重复检查。")

    # Metadata 重复值 (基于 unique_id)
    if 'unique_id' in ddf_metadata.columns:
        check_df_duplicates(ddf_metadata, ['unique_id'], "Metadata")
    else:
        logger.warning("Metadata DataFrame 缺少 'unique_id' 列，跳过重复检查。")

    # Weather 重复值 (基于 location_id 和 timestamp)
    if 'location_id' in ddf_weather.columns and 'timestamp' in ddf_weather.columns:
        check_df_duplicates(
            ddf_weather, ['location_id', 'timestamp'], "Weather")
    else:
        logger.warning(
            "Weather DataFrame 缺少 'location_id' 或 'timestamp' 列，跳过重复检查。")


# 修改函数以使用 Spark API
def log_time_ranges(ddf_demand, ddf_weather):
    """使用 Spark 计算并记录 Demand 和 Weather 数据集的时间戳范围。"""
    logger.info("--- 使用 Spark 计算时间范围 ---")
    try:
        if 'timestamp' in ddf_demand.columns:
            demand_range = ddf_demand.agg(
                F.min('timestamp').alias('min_ts'),
                F.max('timestamp').alias('max_ts')
            ).first()
            if demand_range:
                logger.info(
                    f"Demand 时间范围：从 {demand_range['min_ts']} 到 {demand_range['max_ts']}")
            else:
                logger.warning("无法计算 Demand 时间范围。")
        else:
            logger.warning("Demand DataFrame 缺少 'timestamp' 列，无法计算时间范围。")
    except Exception as e:
        logger.exception(f"计算 Demand 时间范围时出错：{e}")

    try:
        if 'timestamp' in ddf_weather.columns:
            weather_range = ddf_weather.agg(
                F.min('timestamp').alias('min_ts'),
                F.max('timestamp').alias('max_ts')
            ).first()
            if weather_range:
                logger.info(
                    f"Weather 时间范围：从 {weather_range['min_ts']} 到 {weather_range['max_ts']}")
            else:
                logger.warning("无法计算 Weather 时间范围。")
        else:
            logger.warning("Weather DataFrame 缺少 'timestamp' 列，无法计算时间范围。")
    except Exception as e:
        logger.exception(f"计算 Weather 时间范围时出错：{e}")


# 修改 main 函数以创建 SparkSession


def main():
    """主执行函数，编排 Spark 数据加载和分析步骤。"""
    spark = None  # Initialize spark session variable
    try:
        # 创建 SparkSession
        logger.info("创建 SparkSession...")
        spark = SparkSession.builder \
            .appName("ElectricityDemand_LoadData") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
            .config("spark.sql.parquet.int64AsTimestampNanos", "true") \
            .getOrCreate()
        # .master("local[*]") 表示使用所有可用的本地核心
        # 调整 memory 配置根据你的机器资源
        # 添加 rebaseMode 配置以兼容旧的 Parquet 文件
        # 添加 int64AsTimestampNanos 配置以解决 INT64 (TIMESTAMP(NANOS,false)) 问题

        logger.info("SparkSession 创建成功。")
        # 打印 Spark UI 地址
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # 步骤 1: 加载数据集
        ddf_demand, ddf_metadata, ddf_weather = load_datasets(spark)

        # 步骤 2: 记录基本信息
        num_demand_rows, num_metadata_rows, num_weather_rows = log_basic_info(
            ddf_demand, ddf_metadata, ddf_weather)

        # 步骤 3: 检查缺失值
        check_missing_values(ddf_demand, ddf_metadata, ddf_weather,
                             num_demand_rows, num_metadata_rows, num_weather_rows)

        # 步骤 4: 检查重复值
        check_duplicates(ddf_demand, ddf_metadata, ddf_weather)

        # 步骤 5: 记录时间范围
        log_time_ranges(ddf_demand, ddf_weather)

        logger.info("Spark 数据加载和初步检查脚本执行完毕。")
    except Exception as e:
        logger.exception(f"在主执行流程中发生严重错误：{e}")
        sys.exit(1)
    finally:
        if spark:
            logger.info("正在停止 SparkSession...")
            spark.stop()
            logger.info("SparkSession 已停止。")


if __name__ == "__main__":
    main()
