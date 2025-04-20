import dask.dataframe as dd
import pandas as pd
from loguru import logger
import os
import sys
from dask.diagnostics import ProgressBar

# 确保可以导入 src 目录下的模块
# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (假设 preprocessor.py 在 src/electricity_demand_analysis/data/ 下)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)

# 从 data.loader 导入数据加载函数
try:
    from src.electricity_demand_analysis.data.loader import load_electricity_data
except ImportError as e:
    logger.error(f"Failed to import load_electricity_data: {e}")
    logger.error("Ensure the script is run from the project root or the path is correctly configured.")
    sys.exit(1) # 如果无法导入，则退出

# 配置 loguru
log_file_path = os.path.join(project_root, "logs", "data_preprocessing.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # 确保日志目录存在
logger.add(log_file_path, rotation="10 MB", level="INFO") # 单独的预处理日志

def compute_with_progress(dask_obj, desc="Computing"):
    """Helper function to compute Dask object with progress bar."""
    logger.info(f"Starting computation: {desc}")
    with ProgressBar(dt=1.0): # Update progress every 1 second
        result = dask_obj.compute()
    logger.success(f"Computation finished: {desc}")
    return result

def merge_data(npartitions: int | None = None) -> dd.DataFrame | None:
    """
    Loads demand, metadata, and weather data using Dask and merges them.

    Args:
        npartitions: Number of partitions for the final Dask DataFrame.

    Returns:
        A merged Dask DataFrame containing demand, metadata, and weather information,
        or None if loading or merging fails.
    """
    logger.info("--- Starting Data Merging Process ---")

    # --- 1. Load Data ---
    logger.info("Loading data using Dask...")
    loaded_data = load_electricity_data(return_format="dask", npartitions=npartitions)
    if not loaded_data:
        logger.error("Data loading failed. Cannot proceed with merge.")
        return None
    demand_ddf, metadata_df, weather_ddf = loaded_data
    logger.info("Data loaded successfully.")
    logger.info(f"Initial shapes: Demand ~{len(demand_ddf)} (estimated), Metadata {metadata_df.shape}, Weather ~{len(weather_ddf)} (estimated)")
    logger.debug(f"Demand partitions: {demand_ddf.npartitions}, Weather partitions: {weather_ddf.npartitions}")
    logger.debug(f"Demand dtypes:\n{demand_ddf.dtypes}")
    logger.debug(f"Metadata dtypes:\n{metadata_df.dtypes}")
    logger.debug(f"Weather dtypes:\n{weather_ddf.dtypes}")

    # --- 2. Merge Demand and Metadata ---
    logger.info("Merging Demand (Dask) with Metadata (Pandas) on 'unique_id'...")
    # Prepare metadata for merge (select columns, ensure dtype compatibility)
    metadata_subset = metadata_df[['unique_id', 'location_id', 'building_class', 'timezone', 'freq']].copy()
    # Ensure 'unique_id' in both Dask and Pandas are compatible (object/string is safest for merge)
    try:
        if demand_ddf['unique_id'].dtype != 'object':
             logger.debug("Converting demand_ddf['unique_id'] to object for merge.")
             demand_ddf['unique_id'] = demand_ddf['unique_id'].astype('object')
        if metadata_subset['unique_id'].dtype != 'object':
             logger.debug("Converting metadata_subset['unique_id'] to object for merge.")
             metadata_subset['unique_id'] = metadata_subset['unique_id'].astype('object')

        # Perform the merge (Dask DataFrame with Pandas DataFrame)
        # Dask automatically broadcasts the smaller Pandas DataFrame
        merged_demand_meta_ddf = dd.merge(
            demand_ddf,
            metadata_subset,
            on='unique_id',
            how='left' # Keep all demand records
        )
        logger.success("Demand and Metadata merge initiated.")
        logger.debug(f"Merged Demand-Metadata dtypes:\n{merged_demand_meta_ddf.dtypes}")

    except Exception as e:
        logger.error(f"Error merging Demand and Metadata: {e}", exc_info=True)
        logger.error("Check 'unique_id' column existence and dtype compatibility.")
        return None

    # --- 3. Merge Result with Weather ---
    logger.info("Merging result with Weather (Dask) on 'location_id' and 'timestamp'...")
    # Prepare weather data for merge
    # Ensure timestamp is a column, not index (it should be based on loader)
    if 'timestamp' not in weather_ddf.columns:
        logger.warning("Timestamp not found as column in weather_ddf, attempting reset_index()...")
        weather_ddf = weather_ddf.reset_index()
        if 'timestamp' not in weather_ddf.columns:
             logger.error("Failed to find 'timestamp' column in weather_ddf even after reset_index(). Cannot merge.")
             return None

    # Ensure 'location_id' and 'timestamp' types are compatible
    try:
        if merged_demand_meta_ddf['location_id'].dtype != 'object':
             logger.debug("Converting merged_demand_meta_ddf['location_id'] to object.")
             merged_demand_meta_ddf['location_id'] = merged_demand_meta_ddf['location_id'].astype('object')
        if weather_ddf['location_id'].dtype != 'object':
             logger.debug("Converting weather_ddf['location_id'] to object.")
             weather_ddf['location_id'] = weather_ddf['location_id'].astype('object')

        # Ensure timestamp types match (datetime64[ns] is standard)
        if merged_demand_meta_ddf['timestamp'].dtype != weather_ddf['timestamp'].dtype:
            logger.warning(f"Timestamp dtypes differ: "
                           f"Demand/Meta ({merged_demand_meta_ddf['timestamp'].dtype}) vs "
                           f"Weather ({weather_ddf['timestamp'].dtype}). Attempting conversion...")
            # Assuming both should be datetime64[ns]
            merged_demand_meta_ddf['timestamp'] = merged_demand_meta_ddf['timestamp'].astype('datetime64[ns]')
            weather_ddf['timestamp'] = weather_ddf['timestamp'].astype('datetime64[ns]')
            logger.info("Converted timestamp columns to datetime64[ns].")


        # Select relevant weather columns to avoid large intermediate dataframe
        weather_cols_to_merge = [col for col in weather_ddf.columns if col not in ['unique_id']] # Avoid duplicate cols if any
        logger.debug(f"Weather columns selected for final merge: {weather_cols_to_merge}")

        # Perform the final merge
        final_merged_ddf = dd.merge(
            merged_demand_meta_ddf,
            weather_ddf[weather_cols_to_merge],
            on=['location_id', 'timestamp'],
            how='left' # Keep all demand records, match weather where available
        )
        logger.success("Demand/Metadata and Weather merge initiated.")
        logger.debug(f"Final merged dtypes:\n{final_merged_ddf.dtypes}")

    except Exception as e:
        logger.error(f"Error merging with Weather data: {e}", exc_info=True)
        logger.error("Check 'location_id', 'timestamp' columns existence and dtype compatibility.")
        return None

    # --- 4. Final Checks & Return ---
    # Optional: Persist the final DataFrame if it will be used multiple times soon
    # logger.info("Persisting the final merged DataFrame...")
    # final_merged_ddf = final_merged_ddf.persist()
    # logger.info("Final DataFrame persisted.")

    # Log estimated size after merge
    logger.info(f"Final merged DataFrame partitions: {final_merged_ddf.npartitions}")
    logger.info(f"Final estimated length (may be inaccurate before compute): ~{len(final_merged_ddf)}")

    logger.success("Data merging process definition complete.")
    return final_merged_ddf


# --- Usage Example ---
if __name__ == "__main__":
    logger.info("\n--- Testing Data Merging ---")

    # Optional: Setup Dask client for dashboard and better resource management
    # from dask.distributed import Client, LocalCluster
    # try:
    #     cluster = LocalCluster(n_workers=os.cpu_count() // 2, threads_per_worker=2, memory_limit='4GB')
    #     client = Client(cluster)
    #     logger.info(f"Dask dashboard link: {client.dashboard_link}")
    # except Exception as client_err:
    #     logger.warning(f"Could not start Dask client/cluster: {client_err}. Running sequentially.")
    #     client = None

    # Define number of partitions (can be tuned)
    num_partitions = os.cpu_count() * 2 if os.cpu_count() else 4

    merged_ddf = merge_data(npartitions=num_partitions)

    if merged_ddf is not None:
        logger.info("\n--- Inspecting Merged Dask DataFrame ---")
        logger.info(f"Number of partitions: {merged_ddf.npartitions}")
        logger.info(f"Columns: {merged_ddf.columns.tolist()}")
        logger.info(f"dtypes:\n{merged_ddf.dtypes}")

        # --- Example Computations on Merged Data ---
        logger.info("\n--- Performing Example Computations on Merged Data ---")

        # 1. Compute length
        total_rows = 0 # Initialize in case computation fails
        try:
            logger.info("Computing length of the merged DataFrame...")
            total_rows = compute_with_progress(merged_ddf.index.size, desc="Total row count")
            logger.info(f"Total rows in merged DataFrame: {total_rows}")
        except Exception as e:
            logger.error(f"Failed to compute length: {e}", exc_info=True)
            # Fallback using len() - might trigger computation earlier than desired sometimes
            try:
                 logger.info("Attempting length computation via len()...")
                 total_rows = compute_with_progress(len(merged_ddf), desc="Total row count (len)")
                 logger.info(f"Total rows in merged DataFrame (via len): {total_rows}")
            except Exception as e2:
                 logger.error(f"Fallback length computation failed: {e2}", exc_info=True)


        # 2. Compute head
        try:
            logger.info("Computing head of the merged DataFrame...")
            # Call .head() directly; it computes and returns Pandas. Wrap with ProgressBar.
            with ProgressBar(dt=1.0):
                head_df = merged_ddf.head(5)
            logger.success("Computation finished: Head computation")
            logger.info(f"Head of merged DataFrame:\n{head_df}")
        except Exception as e:
            # Add exc_info=True for more detailed traceback
            logger.error(f"Failed to compute head: {e}", exc_info=True)

        # 3. Check for nulls in key columns after merge (example: weather data nulls)
        try:
            logger.info("Checking nulls in 'temperature_2m' after merge...")
            # Make sure temperature_2m exists before trying to access it
            if 'temperature_2m' in merged_ddf.columns:
                null_temp_count = compute_with_progress(
                    merged_ddf['temperature_2m'].isnull().sum(),
                    desc="Null temperature count"
                )
                if total_rows > 0: # Check if total_rows was successfully computed
                    null_perc = (null_temp_count / total_rows) * 100
                    logger.info(f"Null count in 'temperature_2m': {null_temp_count} ({null_perc:.2f}%)")
                else:
                     logger.info(f"Null count in 'temperature_2m': {null_temp_count} (percentage unavailable)")
                logger.info("Note: Nulls here indicate demand entries without matching weather data (e.g., timestamp mismatch or missing location).")
            else:
                logger.warning("Column 'temperature_2m' not found in the merged DataFrame. Skipping null check.")
        except Exception as e:
            logger.error(f"Failed to check nulls for temperature: {e}", exc_info=True)

        logger.success("Data merging test finished successfully.")
    else:
        logger.error("Data merging test failed.")

    # Optional: Shutdown Dask client
    # if 'client' in locals() and client:
    #     logger.info("Shutting down Dask client and cluster...")
    #     client.close()
    #     if 'cluster' in locals() and cluster: # Check if cluster exists
    #         cluster.close()
    #     logger.info("Dask client and cluster shut down.")
