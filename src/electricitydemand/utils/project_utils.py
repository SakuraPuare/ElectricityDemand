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
    additional_configs: Optional[Dict[str, str]] = None
) -> SparkSession:
    """
    Creates or gets a SparkSession with dynamic memory allocation.

    Args:
        app_name: Name of the Spark application.
        driver_mem_gb: Explicit driver memory in GB. Overrides ratio calculation.
        executor_mem_gb: Explicit executor memory in GB. Overrides ratio calculation.
        driver_mem_ratio: Ratio of available system memory for the driver (if not explicit).
        executor_mem_ratio: Ratio of available system memory for executors (if not explicit).
        default_mem_gb: Default memory in GB if calculation results are too low.
        additional_configs: Dictionary of additional Spark configurations.

    Returns:
        An active SparkSession.
    """
    logger.info(f"Creating SparkSession for app: {app_name}...")

    if driver_mem_gb is None or executor_mem_gb is None:
        try:
            total_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
            logger.info(f"System available memory: {total_memory_gb:.2f} GB")
            if driver_mem_gb is None:
                driver_mem_gb = max(default_mem_gb, int(
                    total_memory_gb * driver_mem_ratio + 0.5))
            if executor_mem_gb is None:
                executor_mem_gb = max(default_mem_gb, int(
                    total_memory_gb * executor_mem_ratio + 0.5))
            logger.info(
                f"Calculated memory -> Driver: {driver_mem_gb}g, Executor: {executor_mem_gb}g")
        except Exception as e:
            logger.warning(
                f"Failed to get system memory: {e}. Using default memory: {default_mem_gb}g")
            driver_mem_gb = driver_mem_gb or default_mem_gb
            executor_mem_gb = executor_mem_gb or default_mem_gb

    else:
        logger.info(
            f"Using explicit memory -> Driver: {driver_mem_gb}g, Executor: {executor_mem_gb}g")

    builder = SparkSession.builder.appName(app_name) \
        .config("spark.driver.memory", f"{driver_mem_gb}g") \
        .config("spark.executor.memory", f"{executor_mem_gb}g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # Add any extra configurations
    if additional_configs:
        for key, value in additional_configs.items():
            builder = builder.config(key, value)
            logger.info(f"Adding Spark config: {key}={value}")

    try:
        spark = builder.getOrCreate()
        logger.success("SparkSession created successfully.")
        logger.info(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")
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
