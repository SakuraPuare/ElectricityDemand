import pandas as pd
import dask.dataframe as dd # 导入 Dask DataFrame
from dask.diagnostics import ProgressBar # 用于显示 Dask 进度
from loguru import logger
import sys
import os
import numpy as np # 导入 numpy
from dask.distributed import Client, LocalCluster # Import Client/LocalCluster

# --- Path Setup ---
# 确保可以导入 src 目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (假设 preprocess.py 在 src/electricity_demand_analysis/preprocessing/ 下)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# 将项目根目录添加到 Python 路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from project ---
try:
    # Use the updated loader that returns Dask DataFrames directly
    from src.electricity_demand_analysis.data.loader import load_electricity_data
except ImportError as e:
    logger.error(f"Failed to import load_electricity_data: {e}")
    logger.error("Ensure the script is run from the project root or the path is correctly configured.")
    sys.exit(1)

# --- Logging Setup ---
log_file_path = os.path.join(project_root, "logs", "preprocessing_dask.log") # 使用新的日志文件
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logger.add(log_file_path, rotation="10 MB", level="INFO")

# --- Utility Functions ---

def optimize_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes dtypes for Pandas DataFrames (like metadata)."""
    logger.info("Optimizing Pandas dtypes...")
    original_mem = df.memory_usage(deep=True).sum()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        min_val, max_val = df[col].min(), df[col].max()
        if not pd.isna(min_val): # Check before comparison
             # Ensure max_val is also not NaN before comparison
             if not pd.isna(max_val) and min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                 df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Check if nunique is feasible before computing
            if len(df[col]) < 1_000_000 or df[col].nunique(dropna=False) / len(df[col]) < 0.5: # Adjusted heuristic
                df[col] = df[col].astype('category')
                logger.debug(f"Converted column '{col}' to category.")
        except Exception: # Catch more general exceptions like unhashable types
            logger.warning(f"Could not check nunique or convert column '{col}' to category.")
    new_mem = df.memory_usage(deep=True).sum()
    logger.info(f"Pandas dtypes optimized. Memory reduced from {original_mem / 1e6:.2f} MB to {new_mem / 1e6:.2f} MB")
    return df

# --- Preprocessing Functions using Dask ---

def merge_dataframes_dask_stage1(
    demand_ddf: dd.DataFrame,
    metadata_df: pd.DataFrame
) -> dd.DataFrame | None:
    """Merges demand and metadata."""
    logger.info("--- Stage 1: Merging Demand + Metadata ---")
    metadata_df = optimize_pandas_dtypes(metadata_df.copy())
    try:
        metadata_cols_to_keep = ['unique_id', 'location_id', 'building_class', 'freq', 'timezone', 'dataset']
        # Ensure merge key 'unique_id' in metadata_df is not category before merge if demand_ddf['unique_id'] is object/string
        if 'unique_id' in metadata_df.columns and pd.api.types.is_categorical_dtype(metadata_df['unique_id']):
             logger.debug("Converting metadata 'unique_id' from category to object for merge.")
             metadata_df['unique_id'] = metadata_df['unique_id'].astype(object)
        # Convert demand unique_id to category *if* metadata one is category (choose one)
        # Let's make demand_ddf unique_id category as it's larger
        if 'unique_id' in demand_ddf.columns and not pd.api.types.is_categorical_dtype(demand_ddf['unique_id']):
            logger.debug("Converting demand 'unique_id' to category before merge.")
            demand_ddf['unique_id'] = demand_ddf['unique_id'].astype('category')
        if 'unique_id' in metadata_df.columns and not pd.api.types.is_categorical_dtype(metadata_df['unique_id']):
            logger.debug("Converting metadata 'unique_id' to category before merge.")
            metadata_df['unique_id'] = metadata_df['unique_id'].astype('category')


        merged_ddf = dd.merge(
            demand_ddf,
            metadata_df[metadata_cols_to_keep],
            on='unique_id',
            how='left'
        )
        # Add timestamp_hour column needed for the next stage
        logger.info("Adding 'timestamp_hour' column...")
        merged_ddf['timestamp_hour'] = merged_ddf['timestamp'].dt.floor('H')
        logger.info("Stage 1 merge graph constructed.")
        return merged_ddf
    except Exception as e:
        logger.error(f"Error in Stage 1 merge: {e}", exc_info=True)
        return None

def merge_dataframes_dask_stage2(
    intermediate_ddf: dd.DataFrame,
    weather_ddf: dd.DataFrame
) -> dd.DataFrame | None:
    """Merges intermediate (demand+meta) with weather."""
    logger.info("--- Stage 2: Merging Intermediate + Weather ---")
    try:
        # Prepare weather keys/columns
        logger.info("Preparing weather keys and columns...")
        weather_ddf = weather_ddf.rename(columns={'timestamp': 'timestamp_hour'})
        weather_cols_to_rename = {
            col: f"weather_{col}" for col in weather_ddf.columns
            if col not in ['location_id', 'timestamp_hour']
        }
        weather_ddf = weather_ddf.rename(columns=weather_cols_to_rename)
        weather_cols_to_merge = ['location_id', 'timestamp_hour'] + list(weather_cols_to_rename.values())
        # Ensure dtypes match for merge keys
        if not pd.api.types.is_categorical_dtype(intermediate_ddf['location_id']):
             logger.warning("Intermediate Dask DF 'location_id' is not category. Converting.")
             intermediate_ddf['location_id'] = intermediate_ddf['location_id'].astype('category')
        if not pd.api.types.is_categorical_dtype(weather_ddf['location_id']):
             logger.warning("Weather Dask DF 'location_id' is not category. Converting.")
             weather_ddf['location_id'] = weather_ddf['location_id'].astype('category')

        weather_subset_ddf = weather_ddf[weather_cols_to_merge]
        logger.info("Keys prepared for weather merge.")

        # Perform merge without setting index
        logger.info("Performing merge without indices on 'location_id', 'timestamp_hour'...")
        final_merged_ddf = dd.merge(
            intermediate_ddf,
            weather_subset_ddf,
            on=['location_id', 'timestamp_hour'],
            how='left'
        )

        # Drop the temporary timestamp_hour column
        logger.info("Dropping temporary 'timestamp_hour' column...")
        final_merged_ddf = final_merged_ddf.drop(columns=['timestamp_hour'])
        logger.info("Stage 2 merge graph constructed.")
        return final_merged_ddf

    except Exception as e:
        logger.error(f"Error in Stage 2 merge: {e}", exc_info=True)
        return None


# --- Main Execution using Dask (Iterative) ---

def run_preprocessing_dask():
    """Loads data and runs the preprocessing steps using Dask in stages."""
    logger.info("--- Starting Dask Preprocessing Pipeline (Iterative Stages) ---")

    # --- Start Dask Client ---
    n_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 2)
    mem_limit = '6GB' # ADJUST THIS based on system RAM / n_workers
    logger.info(f"Starting Dask LocalCluster with {n_workers} workers, {mem_limit} memory limit each.")
    cluster = None
    client = None
    try:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=2, memory_limit=mem_limit, dashboard_address=':8788')
        client = Client(cluster)
        logger.info(f"Dask client started: {client.dashboard_link}")
    except Exception as e:
        logger.error(f"Failed to start Dask client: {e}. Running without client.", exc_info=True)

    # Define intermediate and final file paths
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    intermediate_path = os.path.join(output_dir, "intermediate_demand_meta.parquet")
    final_output_path = os.path.join(output_dir, "merged_data_final.parquet")

    intermediate_ddf = None # Initialize

    try: # Wrap main logic in try block to ensure client shutdown

        # === Stage 1: Load Demand/Meta and Merge ===
        logger.info("=== Entering Stage 1: Demand + Metadata Merge ===")
        logger.info("Loading data for Stage 1...")
        load_npartitions = n_workers * 4
        # Load data - ensure loader returns appropriate Dask DFs
        loaded_data_stage1 = load_electricity_data(return_format="dask", npartitions=load_npartitions)
        if not loaded_data_stage1:
            logger.error("Data loading failed for Stage 1. Exiting preprocessing.")
            return # Exit early
        demand_ddf, metadata_df, _ = loaded_data_stage1 # Ignore weather for now
        logger.info("Data loaded successfully for Stage 1.")

        # Merge Demand and Metadata
        intermediate_ddf_graph = merge_dataframes_dask_stage1(demand_ddf, metadata_df)
        del demand_ddf, metadata_df # Release memory

        if intermediate_ddf_graph is None:
            logger.error("Stage 1 merge failed. Exiting.")
            return

        # Save intermediate result
        logger.info(f"Attempting to save Stage 1 intermediate result to {intermediate_path}...")
        try:
            with ProgressBar():
                intermediate_ddf_graph.to_parquet(
                    intermediate_path,
                    engine='pyarrow',
                    write_index=False,
                    schema='infer',
                    overwrite=True
                )
            logger.success(f"Stage 1 intermediate data saved successfully to: {intermediate_path}")
        except Exception as e:
            logger.error(f"Failed to save Stage 1 intermediate data: {e}", exc_info=True)
            return # Cannot proceed without intermediate data

        # Clear intermediate graph from memory (important!)
        del intermediate_ddf_graph
        if client:
             logger.info("Restarting Dask workers to clear memory...")
             try:
                  client.restart() # Restart workers to clear memory completely
                  logger.info("Dask workers restarted after Stage 1.")
             except Exception as restart_err:
                  logger.warning(f"Could not restart workers: {restart_err}. Memory might not be fully cleared.")
        logger.info("=== Stage 1 Completed ===")


        # === Stage 2: Load Intermediate/Weather and Merge ===
        logger.info("=== Entering Stage 2: Intermediate + Weather Merge ===")

        # Load intermediate data
        logger.info(f"Loading intermediate data from {intermediate_path}...")
        try:
            intermediate_ddf = dd.read_parquet(intermediate_path, engine='pyarrow')
             # Ensure intermediate dtypes are reasonable, especially merge keys
            intermediate_ddf['location_id'] = intermediate_ddf['location_id'].astype('category')
            intermediate_ddf['timestamp_hour'] = dd.to_datetime(intermediate_ddf['timestamp_hour'])
            logger.info(f"Intermediate data loaded (Partitions: {intermediate_ddf.npartitions}). Dtypes:\n{intermediate_ddf.dtypes}")
        except Exception as e:
             logger.error(f"Failed to load intermediate data from {intermediate_path}: {e}", exc_info=True)
             return

        # Load weather data again (it's small and cached locally now)
        logger.info("Loading weather data again...")
        loaded_data_stage2 = load_electricity_data(return_format="dask", npartitions=max(1, n_workers))
        if not loaded_data_stage2:
            logger.error("Weather data loading failed for Stage 2. Exiting.")
            return
        _, _, weather_ddf = loaded_data_stage2 # Only need weather now
        logger.info("Weather data loaded successfully for Stage 2.")

        # Merge Intermediate and Weather
        final_merged_ddf = merge_dataframes_dask_stage2(intermediate_ddf, weather_ddf)
        del intermediate_ddf, weather_ddf # Release memory

        if final_merged_ddf is None:
             logger.error("Stage 2 merge failed. Exiting.")
             return

        # Save final result
        logger.info(f"Attempting to save final merged DataFrame to {final_output_path}...")
        try:
            with ProgressBar():
                final_merged_ddf.to_parquet(
                    final_output_path,
                    engine='pyarrow',
                    write_index=False,
                    schema='infer',
                    overwrite=True
                )
            logger.success(f"Final merged data saved successfully to directory: {final_output_path}")
        except Exception as e:
            logger.error(f"Failed to save final merged data: {e}", exc_info=True)

        logger.info("=== Stage 2 Completed ===")

    finally:
        # --- Shutdown Dask Client ---
        if client:
            logger.info("Shutting down Dask client and cluster...")
            try:
                client.close()
                if cluster:
                    cluster.close()
                logger.info("Dask client and cluster shut down.")
            except Exception as ce:
                logger.error(f"Error shutting down Dask client/cluster: {ce}")

    logger.success("--- Dask Preprocessing Pipeline (Iterative Stages) Completed ---")


if __name__ == "__main__":
    run_preprocessing_dask()