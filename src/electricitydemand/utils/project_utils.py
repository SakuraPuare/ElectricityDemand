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
    driver_memory_gb=None,  # Allow override
    executor_memory_gb=None, # Allow override
    executor_cores=None,     # Allow override
    num_executors=None,      # Allow override
    max_ram_gb=None,
    log_level="WARN"
):
    """
    创建并配置一个优化的 SparkSession 实例。

    尝试根据系统资源（CPU 核心数、可用内存）和 Spark 最佳实践来自动配置，
    优先考虑本地模式下的 CPU 和内存利用率。
    允许通过函数参数或环境变量覆盖自动配置。

    环境变量覆盖 (优先级最高):
    - SPARK_DRIVER_MEMORY_GB
    - SPARK_EXECUTOR_MEMORY_GB
    - SPARK_EXECUTOR_CORES
    - SPARK_NUM_EXECUTORS
    - SPARK_MAX_RAM_GB

    Args:
        app_name (str): Spark 应用的名称。
        master (str): Spark master URL (e.g., "local[*]", "yarn").
        driver_memory_gb (int, optional): 手动指定 Driver 内存 (GB)。
        executor_memory_gb (int, optional): 手动指定 Executor 内存 (GB)。
        executor_cores (int, optional): 手动指定每个 Executor 的核心数。
        num_executors (int, optional): 手动指定 Executor 数量。
        max_ram_gb (int, optional): 手动指定此 SparkSession 可使用的最大系统内存 (GB)。
        log_level (str): Spark 日志级别 (e.g., "INFO", "WARN", "ERROR").

    Returns:
        SparkSession: 配置好的 SparkSession 实例，如果出错则返回 None。
    """
    logger.info(f"创建SparkSession: {app_name}，优先考虑CPU利用率...")

    try:
        # --- 1. 获取系统资源 ---
        total_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        svmem = psutil.virtual_memory()
        total_ram_gb = svmem.total / (1024**3)
        available_ram_gb = svmem.available / (1024**3)

        # 允许通过环境变量或参数限制最大可用 RAM
        max_ram_gb_env = os.getenv("SPARK_MAX_RAM_GB")
        if max_ram_gb_env:
            try:
                max_ram_gb = int(max_ram_gb_env)
                logger.info(f"从环境变量读取最大内存限制: {max_ram_gb} GB")
            except ValueError:
                logger.warning(f"无效的环境变量 SPARK_MAX_RAM_GB: '{max_ram_gb_env}'. 忽略.")
        if max_ram_gb is None:
            max_ram_gb = total_ram_gb # 默认使用全部
        elif max_ram_gb > total_ram_gb:
            logger.warning(f"指定的 max_ram_gb ({max_ram_gb}GB) 大于系统总内存 ({total_ram_gb:.2f}GB)。将使用系统总内存。")
            max_ram_gb = total_ram_gb

        # 调整可用内存基于限制
        effective_available_ram_gb = min(available_ram_gb, max_ram_gb)
        logger.info(f"系统资源: {physical_cores} 物理核心, {total_cores} 逻辑核心, {total_ram_gb:.2f} GB 总内存, {available_ram_gb:.2f} GB 可用内存 (上限: {max_ram_gb:.2f}GB, 生效可用: {effective_available_ram_gb:.2f}GB)")

        # --- 2. 确定 Driver 和 Executor 配置 (允许覆盖) ---
        # 覆盖逻辑：环境变量 > 函数参数 > 自动计算

        # Driver Cores (通常不需要太多，除非 Driver 任务重)
        driver_cores = 4 if total_cores >= 8 else max(1, total_cores // 4) # 保留大部分给 executor

        # Driver Memory (上次 OOM 的关键点，给更多)
        # 尝试分配可用内存的 15-25%，或至少 8GB，但不超过一个硬上限 (e.g., 64GB)
        auto_driver_memory_gb = max(8, int(effective_available_ram_gb * 0.20))
        auto_driver_memory_gb = min(auto_driver_memory_gb, 64) # 硬上限防止过大
        driver_memory_gb = int(os.getenv("SPARK_DRIVER_MEMORY_GB", driver_memory_gb or auto_driver_memory_gb))

        # Executor Cores (推荐 3-5 个核心，以平衡并行度和内存)
        auto_executor_cores = min(5, max(3, physical_cores // 4 if physical_cores else total_cores // 4))
        executor_cores = int(os.getenv("SPARK_EXECUTOR_CORES", executor_cores or auto_executor_cores))

        # 计算可用资源给 Executors
        available_cores_for_executors = total_cores - driver_cores
        available_ram_for_executors_gb = effective_available_ram_gb - driver_memory_gb - (total_ram_gb * 0.1) # 减去 Driver 内存和 10% OS 预留

        if available_cores_for_executors <= 0 or available_ram_for_executors_gb <= 1:
            logger.error("没有足够的 CPU 核心或内存分配给 Executors。请检查 Driver 配置或系统资源。")
            return None

        # Executor 数量
        auto_num_executors = available_cores_for_executors // executor_cores
        num_executors = int(os.getenv("SPARK_NUM_EXECUTORS", num_executors or auto_num_executors))
        num_executors = max(1, num_executors) # 至少一个 executor

        # Executor Memory (均分剩余可用内存给 Executors)
        auto_executor_total_memory_gb = available_ram_for_executors_gb / num_executors
        # 为 MemoryOverhead 留出空间 (e.g., 15-20% of total executor memory, or at least 2GB)
        overhead_factor = 0.18 # 增加 overhead 比例
        min_overhead_gb = 2
        # 计算 Executor Heap Memory
        auto_executor_memory_gb_float = auto_executor_total_memory_gb * (1 - overhead_factor)
        auto_executor_memory_gb = max(1, int(auto_executor_memory_gb_float)) # 至少 1GB heap

        # 计算 Memory Overhead
        calculated_overhead_gb = max(min_overhead_gb, int(auto_executor_total_memory_gb * overhead_factor))

        # 最终确定 Executor Memory 和 Overhead
        executor_memory_gb = int(os.getenv("SPARK_EXECUTOR_MEMORY_GB", executor_memory_gb or auto_executor_memory_gb))
        executor_memory_overhead_mb = calculated_overhead_gb * 1024


        # 计算建议的 shuffle 分区数 (可以基于总核心数调整)
        total_executor_cores = num_executors * executor_cores
        shuffle_partitions = max(200, total_executor_cores * 3) # 经验值，可以调整

        logger.info(f"计算资源 -> Driver: {driver_memory_gb}GB/{driver_cores} 核心 | Executors: {num_executors} 个 * ({executor_memory_gb}GB Heap/{executor_cores} 核心), Overhead: {executor_memory_overhead_mb}MB | Shuffle Partitions: {shuffle_partitions}")


        # --- 3. 构建 SparkConf ---
        conf = SparkConf().setAppName(app_name).setMaster(master)

        # 基本配置
        conf.set("spark.driver.memory", f"{driver_memory_gb}g")
        conf.set("spark.driver.cores", str(driver_cores))
        conf.set("spark.executor.memory", f"{executor_memory_gb}g")
        conf.set("spark.executor.cores", str(executor_cores))
        conf.set("spark.executor.instances", str(num_executors))
        conf.set("spark.executor.memoryOverhead", f"{executor_memory_overhead_mb}m")
        conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
        conf.set("spark.default.parallelism", str(shuffle_partitions)) # 保持一致性

        # 性能调优和稳定性配置
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") # 通常比 Java 序列化更快
        conf.set("spark.kryoserializer.buffer.max", "512m") # 增加 Kryo 缓冲区
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true") # 使用 Arrow 优化 PySpark 数据传输
        conf.set("spark.sql.execution.pyarrow.fallback.enabled", "true") # Arrow 不可用时回退
        conf.set("spark.sql.adaptive.enabled", "true") # 启用 AQE
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true") # 自动合并小分区
        conf.set("spark.sql.adaptive.skewJoin.enabled", "true") # 处理倾斜 Join
        # conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64m") # AQE 建议的分区大小 (可调)
        # conf.set("spark.sql.files.maxPartitionBytes", "128m") # 读取文件时每个分区的最大字节数 (可调)
        # conf.set("spark.sql.autoBroadcastJoinThreshold", "100m") # 自动广播小表的阈值 (可调)
        conf.set("spark.memory.fraction", "0.6") # Spark 统一内存管理中执行和存储内存的比例
        conf.set("spark.memory.storageFraction", "0.3") # 上述内存中用于存储(缓存)的比例 (相应减少执行内存比例)
        conf.set("spark.sql.broadcastTimeout", "900") # 广播超时时间（秒）
        conf.set("spark.network.timeout", "800s") # 网络超时，处理慢节点
        conf.set("spark.executor.heartbeatInterval", "60s") # Executor 心跳间隔
        conf.set("spark.shuffle.file.buffer", "1m") # Shuffle 文件缓冲区大小
        conf.set("spark.unsafe.sorter.spill.reader.buffer.size", "1m") # 溢写排序读取缓冲区

        # 本地模式特定优化 (如果 master 是 local)
        if master.startswith("local"):
            conf.set("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+DisableExplicitGC -XX:+HeapDumpOnOutOfMemoryError") # G1 GC, 禁止代码中显式GC, OOM时dump堆
            conf.set("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+DisableExplicitGC -XX:+HeapDumpOnOutOfMemoryError") # 同上
        else:
             # 集群模式可能需要不同的 GC 或 Java 选项
             pass

        # 设置日志级别
        conf.set("spark.logConf", "true") # 在日志中打印有效配置

        # --- 4. 创建 SparkSession ---
        spark_session = SparkSession.builder.config(conf=conf).getOrCreate()
        spark_session.sparkContext.setLogLevel(log_level) # 设置 PySpark Driver 日志级别

        logger.success("SparkSession 创建成功")
        logger.info(f"Spark Web UI: {spark_session.sparkContext.uiWebUrl}")

        # 打印关键配置信息
        logger.info("Spark 配置:")
        key_configs = [
            "spark.app.name", "spark.master", "spark.driver.memory", "spark.driver.cores",
            "spark.executor.memory", "spark.executor.cores", "spark.executor.instances",
            "spark.executor.memoryOverhead", "spark.default.parallelism", "spark.sql.shuffle.partitions",
            "spark.driver.maxResultSize", "spark.sql.adaptive.enabled"
        ]
        all_conf = spark_session.sparkContext.getConf().getAll()
        conf_dict = dict(all_conf)
        for key in key_configs:
            if key in conf_dict:
                logger.info(f"  {key} = {conf_dict[key]}")

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
