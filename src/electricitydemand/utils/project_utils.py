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
    driver_mem_ratio: float = 0.5,
    executor_mem_ratio: float = 0.5,
    default_mem_gb: int = 4,
    driver_cores: Optional[int] = None,
    executor_cores: Optional[int] = None,
    dynamic_allocation: bool = True,
    additional_configs: Optional[Dict[str, str]] = None
) -> SparkSession:
    """
    Creates or gets a SparkSession with dynamic resource allocation based on system specs.

    Args:
        app_name: Name of the Spark application.
        driver_mem_gb: Explicit driver memory in GB. Overrides ratio calculation.
        executor_mem_gb: Explicit executor memory in GB. Overrides ratio calculation.
        driver_mem_ratio: Ratio of available system memory for the driver (if not explicit).
        executor_mem_ratio: Ratio of available system memory for executors (if not explicit).
        default_mem_gb: Default memory in GB if calculation results are too low.
        driver_cores: Number of cores for driver. If None, calculated automatically.
        executor_cores: Number of cores per executor. If None, calculated automatically.
        dynamic_allocation: Whether to enable dynamic allocation for executors.
        additional_configs: Dictionary of additional Spark configurations.

    Returns:
        An active SparkSession.
    """
    logger.info(f"Creating SparkSession for app: {app_name}...")

    # Get system resources
    try:
        total_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

        logger.info(f"System resources: {physical_cores} physical cores, "
                  f"{total_cores} logical cores, "
                  f"{total_memory_gb:.2f} GB total memory, "
                  f"{available_memory_gb:.2f} GB available memory")

        # Reserve memory for OS and other processes (10% of total or at least 2GB)
        system_reserve_gb = max(2, total_memory_gb * 0.1)
        usable_memory_gb = available_memory_gb - system_reserve_gb

        # Calculate memory if not provided
        if driver_mem_gb is None:
            driver_mem_gb = max(default_mem_gb, int(usable_memory_gb * driver_mem_ratio))

        if executor_mem_gb is None:
            executor_mem_gb = max(default_mem_gb, int(usable_memory_gb * executor_mem_ratio))

        # Calculate cores if not provided
        if driver_cores is None:
            driver_cores = max(1, min(4, physical_cores // 2))  # 1-4 cores for driver

        if executor_cores is None:
            executor_cores = max(1, min(4, physical_cores // 2))  # 1-4 cores per executor

        # Calculate number of executors based on available resources
        default_parallelism = max(2, total_cores - driver_cores)

        logger.info(
            f"Calculated resources -> Driver: {driver_mem_gb}GB/{driver_cores} cores, "
            f"Executor: {executor_mem_gb}GB/{executor_cores} cores, "
            f"Default parallelism: {default_parallelism}")
    except Exception as e:
        logger.warning(f"Failed to get system resources: {e}. Using defaults.")
        driver_mem_gb = driver_mem_gb or default_mem_gb
        executor_mem_gb = executor_mem_gb or default_mem_gb
        driver_cores = driver_cores or 2
        executor_cores = executor_cores or 2
        default_parallelism = 4

    # Build Spark configuration
    builder = SparkSession.builder.appName(app_name) \
        .config("spark.driver.memory", f"{driver_mem_gb}g") \
        .config("spark.executor.memory", f"{executor_mem_gb}g") \
        .config("spark.driver.cores", driver_cores) \
        .config("spark.executor.cores", executor_cores) \
        .config("spark.default.parallelism", default_parallelism) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3")

    # Enable dynamic allocation if requested
    if dynamic_allocation:
        builder = builder \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.shuffle.service.enabled", "true") \
            .config("spark.dynamicAllocation.initialExecutors", "1") \
            .config("spark.dynamicAllocation.minExecutors", "1") \
            .config("spark.dynamicAllocation.maxExecutors", max(2, total_cores // executor_cores))

    # Add off-heap memory configuration
    overhead_factor = 0.1  # 10% of executor memory for off-heap
    overhead_mb = int(executor_mem_gb * 1024 * overhead_factor)
    builder = builder.config("spark.executor.memoryOverhead", f"{overhead_mb}m")

    # Add any extra configurations
    if additional_configs:
        for key, value in additional_configs.items():
            builder = builder.config(key, value)
            logger.info(f"Adding Spark config: {key}={value}")

    try:
        spark = builder.getOrCreate()

        # Set level of parallelism for shuffles
        spark.conf.set("spark.sql.shuffle.partitions", default_parallelism * 2)

        logger.success("SparkSession created successfully.")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

        # Log final configuration
        spark_conf = spark.sparkContext.getConf().getAll()
        logger.info("Spark configuration:")
        for item in spark_conf:
            logger.info(f"  {item[0]} = {item[1]}")

        return spark
    except Exception as e:
        logger.exception("Failed to create SparkSession!")
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
