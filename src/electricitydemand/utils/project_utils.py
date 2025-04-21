import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

from loguru import logger
from pyspark.sql import SparkSession
import psutil
from pyspark import SparkConf


def get_project_root() -> Path:
    """Dynamically detects the project root directory."""
    try:
        # Assumes the script is within the project structure
        _script_path = Path(os.path.abspath(__file__)).resolve()
        # Navigate up until a known marker is found (e.g., 'pyproject.toml', '.git')
        # Or assume a fixed structure (e.g., utils is 2 levels down from root)
        project_root = _script_path.parent.parent.parent.parent
        # Add a check for a marker file if structure varies
        if not (project_root / 'pyproject.toml').exists() and not (project_root / '.git').exists():
            logger.warning(
                f"Could not reliably confirm project root at {project_root}. Assuming structure.")
    except NameError:
        # Fallback for interactive use or environments where __file__ is not defined
        project_root = Path.cwd()
        logger.warning(
            f"__file__ not defined, using current working directory as project root: {project_root}")
    return project_root


def setup_project_paths(project_root: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Sets up common project directory paths and adds src to sys.path.

    Args:
        project_root: The root directory of the project.

    Returns:
        A tuple containing paths for: (src_path, data_dir, logs_dir, plots_dir)
    """
    src_path = project_root / 'src'
    data_dir = project_root / 'data'
    logs_dir = project_root / 'logs'
    plots_dir = project_root / 'plots'  # Common directory, even if not used by all

    # Add src to sys.path if not already present
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        logger.debug(f"Added '{src_path}' to sys.path")

    # Ensure common directories exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    # data_dir might be created by download script, or assumed to exist

    return src_path, data_dir, logs_dir, plots_dir

def create_spark_session(
    app_name="SparkApplication",
    master="local[*]",
    log_level="WARN",
    driver_memory="8g",
    executor_memory="16g"
):
    """
    创建并配置一个基本的 SparkSession 实例，包含一些性能优化配置。

    设置应用名称、master URL、日志级别、本地临时目录，
    并添加了 Kryo 序列化、Arrow 优化、自适应查询执行 (AQE)
    和一些 shuffle 相关配置。现在可以显式配置 Driver 和 Executor 内存。

    Args:
        app_name (str): Spark 应用的名称。
        master (str): Spark master URL (e.g., "local[*]", "yarn").
        log_level (str): Spark 日志级别 (e.g., "INFO", "WARN", "ERROR").
        driver_memory (str): Spark Driver 的内存 (e.g., "2g", "4g").
        executor_memory (str): 每个 Spark Executor 的内存 (e.g., "4g", "8g").

    Returns:
        SparkSession: 配置好的 SparkSession 实例，如果出错则返回 None。
    """
    logger.info(f"创建 SparkSession: {app_name} (master: {master})...")
    logger.info(f"配置 Driver 内存: {driver_memory}")
    logger.info(f"配置 Executor 内存: {executor_memory}")

    try:
        # --- 构建 SparkConf ---
        conf = SparkConf().setAppName(app_name).setMaster(master)

        # --- 配置内存 ---
        conf.set("spark.driver.memory", driver_memory)
        conf.set("spark.executor.memory", executor_memory)

        # 配置 Spark 本地临时目录
        local_tmp_dir = "/home/ubuntu/data/tmp"
        logger.info(f"配置 Spark 本地临时目录: {local_tmp_dir}")
        try:
            os.makedirs(local_tmp_dir, exist_ok=True)
            logger.debug(f"确认本地临时目录 {local_tmp_dir} 已存在或已创建。")
        except OSError as e:
            logger.warning(f"无法创建或确认本地临时目录 {local_tmp_dir}: {e}. 请确保该目录存在且 Spark 有权访问。")
        conf.set("spark.local.dir", local_tmp_dir)

        # --- 添加性能和稳定性配置 ---
        # 序列化
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.kryoserializer.buffer.max", "1024m")

        # Arrow 优化
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.execution.pyarrow.fallback.enabled", "true")

        # 自适应查询执行 (AQE)
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "2048m")
        conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

        # Shuffle 相关配置
        shuffle_partitions = 500
        conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
        conf.set("spark.default.parallelism", str(shuffle_partitions))
        logger.info(f"设置 spark.sql.shuffle.partitions 和 spark.default.parallelism 为: {shuffle_partitions}")

        # --- 启用并配置堆外内存 ---
        conf.set("spark.memory.offHeap.enabled", "true")
        # Estimate based on cores; adjust if needed. Example: 2GB per executor regardless of core count.
        # Or calculate based on typical core count, e.g., 4 cores * 1g = 4g
        off_heap_size = "32g" # Example: Allocate 4GB off-heap per executor
        conf.set("spark.memory.offHeap.size", off_heap_size)
        logger.info(f"配置 Executor 堆外内存 (spark.memory.offHeap.size): {off_heap_size}")

        # 网络和心跳超时
        conf.set("spark.network.timeout", "800s")
        conf.set("spark.executor.heartbeatInterval", "60s")

        # 设置日志级别
        conf.set("spark.logConf", "true")

        # --- 创建 SparkSession ---
        spark_session = SparkSession.builder.config(conf=conf).getOrCreate()
        spark_session.sparkContext.setLogLevel(log_level) # 设置 PySpark Driver 日志级别

        logger.success("SparkSession 创建成功 (含性能优化)")
        logger.info(f"Spark Web UI: {spark_session.sparkContext.uiWebUrl}")

        # 打印关键配置信息
        logger.info("Spark 配置:")
        key_configs = [
            "spark.app.name", "spark.master", "spark.driver.memory", "spark.executor.memory",
            "spark.memory.offHeap.enabled", "spark.memory.offHeap.size",
            "spark.serializer", "spark.kryoserializer.buffer.max",
            "spark.sql.execution.arrow.pyspark.enabled",
            "spark.sql.adaptive.enabled", "spark.sql.adaptive.advisoryPartitionSizeInBytes",
            "spark.sql.shuffle.partitions",
            "spark.default.parallelism", "spark.local.dir", "spark.network.timeout"
        ]
        all_conf = spark_session.sparkContext.getConf().getAll()
        conf_dict = dict(all_conf)
        for key in key_configs:
            value = conf_dict.get(key, "未显式设置 (使用默认值)")
            logger.info(f"  {key} = {value}")

        return spark_session

    except Exception as e:
        logger.error(f"创建 SparkSession 失败: {e}")
        logger.exception("详细错误信息:")
        return None

def stop_spark_session(spark: Optional[SparkSession]):
    """Safely stops the given SparkSession if it's active."""
    if spark:
        try:
            if not spark.sparkContext._jsc.sc().isStopped():
                logger.info("Stopping SparkSession...")
                spark.stop()
                logger.success("SparkSession stopped.")
            else:
                logger.info("SparkSession was already stopped.")
        except Exception as e:
            logger.error(f"Error stopping SparkSession: {e}", exc_info=True)
    else:
        logger.info("No active SparkSession found to stop.")
