import os
import sys
import psutil
from pathlib import Path
from typing import Tuple, Optional, Dict

from loguru import logger
from pyspark.sql import SparkSession


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
    app_name: str = "SparkApplication",
    driver_mem_gb: Optional[int] = None,
    executor_mem_gb: Optional[int] = None,
    driver_mem_ratio: float = 0.2,  # 进一步降低比例
    executor_mem_ratio: float = 0.5,  # 调整比例
    default_mem_gb: int = 6,
    driver_cores: Optional[int] = None,
    executor_cores: Optional[int] = None,
    dynamic_allocation: bool = False,  # 关闭动态分配
    additional_configs: Optional[Dict[str, str]] = None,
    max_memory_per_node_gb: int = 160,
    enable_offheap: bool = False,  # 关闭堆外内存
    offheap_size_gb: int = 16  # 减小堆外内存大小
) -> SparkSession:
    """
    创建高度优化的SparkSession，着重于CPU利用率而非内存使用。
    通过分区控制和任务分散来减轻内存压力。
    """
    logger.info(f"创建SparkSession: {app_name}，优先考虑CPU利用率...")

    # 获取系统资源
    try:
        total_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

        # 限制可用内存上限
        available_memory_gb = min(available_memory_gb, max_memory_per_node_gb)

        logger.info(f"系统资源: {physical_cores} 物理核心, "
                  f"{total_cores} 逻辑核心, "
                  f"{total_memory_gb:.2f} GB 总内存, "
                  f"{available_memory_gb:.2f} GB 可用内存 (上限: {max_memory_per_node_gb}GB)")

        # 更严格的内存保留
        system_reserve_gb = max(32, total_memory_gb * 0.3)  # 提高保留比例
        usable_memory_gb = max(available_memory_gb - system_reserve_gb, total_memory_gb * 0.5)

        # 计算内存
        if driver_mem_gb is None:
            driver_mem_gb = max(default_mem_gb, int(usable_memory_gb * driver_mem_ratio))
            driver_mem_gb = min(driver_mem_gb, 24)  # 减少驱动器内存上限

        if executor_mem_gb is None:
            executor_mem_gb = max(default_mem_gb, int(usable_memory_gb * executor_mem_ratio))
            executor_mem_gb = min(executor_mem_gb, 32)  # 减少执行器内存上限

        # 计算更小的核心数
        if driver_cores is None:
            driver_cores = max(2, min(4, physical_cores // 10))

        if executor_cores is None:
            executor_cores = max(1, min(3, physical_cores // 12))  # 更小的执行器

        # 计算更多数量但更小的执行器
        recommended_executors = max(4, (total_cores - driver_cores) // executor_cores)

        # 大幅增加分区数，减少每个任务的数据量
        default_parallelism = max(recommended_executors * executor_cores * 10, 
                                total_cores * 8)

        # 增加内存开销
        overhead_factor = 0.3
        overhead_mb = max(768, int(executor_mem_gb * 1024 * overhead_factor))

        # 计算堆外内存
        offheap_mb = offheap_size_gb * 1024 if enable_offheap else 0

        logger.info(
            f"计算资源 -> 驱动: {driver_mem_gb}GB/{driver_cores} 核心, "
            f"执行器: {executor_mem_gb}GB/{executor_cores} 核心, "
            f"建议执行器数量: {recommended_executors}, "
            f"堆外内存: {offheap_size_gb if enable_offheap else 0}GB, "
            f"内存开销: {overhead_mb}MB, "
            f"默认并行度: {default_parallelism}")
    except Exception as e:
        logger.warning(f"获取系统资源失败: {e}. 使用默认值。")
        driver_mem_gb = driver_mem_gb or 12
        executor_mem_gb = executor_mem_gb or 20
        driver_cores = driver_cores or 2
        executor_cores = executor_cores or 2
        recommended_executors = 12
        default_parallelism = 500  # 大幅增加默认并行度
        overhead_mb = 6144
        offheap_mb = 0  # 不使用堆外内存

    # 构建Spark配置
    builder = SparkSession.builder.appName(app_name) \
        .config("spark.driver.memory", f"{driver_mem_gb}g") \
        .config("spark.executor.memory", f"{executor_mem_gb}g") \
        .config("spark.driver.cores", driver_cores) \
        .config("spark.executor.cores", executor_cores) \
        .config("spark.default.parallelism", default_parallelism) \
        .config("spark.sql.shuffle.partitions", default_parallelism * 2) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.memory.fraction", "0.6")  \
        .config("spark.memory.storageFraction", "0.1")  \
        .config("spark.executor.memoryOverhead", f"{overhead_mb}m") \
        .config("spark.driver.maxResultSize", f"{max(1, driver_mem_gb // 6)}g") \
        .config("spark.network.timeout", "1800s")  \
        .config("spark.executor.heartbeatInterval", "180s")  \
        .config("spark.speculation", "false") \
        .config("spark.rdd.compress", "true") \
        .config("spark.io.compression.codec", "lz4") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "512m") \
        .config("spark.sql.autoBroadcastJoinThreshold", "32m")  \
        .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
        .config("spark.sql.inMemoryColumnarStorage.batchSize", "5000")  \
        .config("spark.sql.windowExec.buffer.spill.threshold", "1024")  \
        .config("spark.unsafe.sorter.spill.reader.buffer.size", "512k") \
        .config("spark.shuffle.file.buffer", "512k") \
        .config("spark.sql.files.maxPartitionBytes", "64m")  \
        .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "16") \
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "32m") \
        .config("spark.cleaner.periodicGC.interval", "30s") \
        .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+DisableExplicitGC -XX:+HeapDumpOnOutOfMemoryError") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+DisableExplicitGC -XX:+HeapDumpOnOutOfMemoryError") \
        .config("spark.locality.wait", "0s") \
        .config("spark.sql.broadcastTimeout", "900") \
        .config("spark.sql.autoBroadcastJoinThreshold", "8m")

    # 启用堆外内存配置
    if enable_offheap:
        builder = builder \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", f"{offheap_mb}m")

    # 动态分配设置
    if dynamic_allocation:
        builder = builder \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.shuffle.service.enabled", "false") \
            .config("spark.dynamicAllocation.initialExecutors", str(max(2, recommended_executors // 2))) \
            .config("spark.dynamicAllocation.minExecutors", "2") \
            .config("spark.dynamicAllocation.maxExecutors", str(max(8, recommended_executors * 2))) \
            .config("spark.dynamicAllocation.executorIdleTimeout", "60s") \
            .config("spark.dynamicAllocation.schedulerBacklogTimeout", "30s") \
            .config("spark.dynamicAllocation.sustainedSchedulerBacklogTimeout", "15s")
    else:
        # 固定更多数量但更小的执行器
        builder = builder.config("spark.executor.instances", str(recommended_executors * 2))

    # 添加额外配置
    if additional_configs:
        for key, value in additional_configs.items():
            builder = builder.config(key, value)
            logger.info(f"添加配置: {key}={value}")

    try:
        spark = builder.getOrCreate()
        logger.success("SparkSession 创建成功")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # 记录最终配置
        spark_conf = spark.sparkContext.getConf().getAll()
        logger.info("Spark 配置:")
        for item in spark_conf:
            logger.info(f"  {item[0]} = {item[1]}")

        return spark
    except Exception as e:
        logger.exception("创建 SparkSession 失败!")
        raise
    
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
