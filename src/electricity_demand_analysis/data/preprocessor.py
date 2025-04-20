from datetime import datetime
import dask.dataframe as dd
import pandas as pd
from loguru import logger
import os
import sys
from dask.diagnostics import ProgressBar
import numpy as np # Import numpy for numeric types check
from dask.base import is_dask_collection # Import to check if object is Dask type
import pytz # Import pytz for timezone handling

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
log_file_path = os.path.join(project_root, "logs", f"data_preprocessing_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # 确保日志目录存在
logger.add(log_file_path, rotation="10 MB", level="INFO") # 单独的预处理日志

def compute_with_progress(dask_obj, desc="Computing"):
    """
    Helper function to compute Dask object with progress bar.
    Handles cases where the input might already be computed.
    """
    # Check if it's a Dask collection and has a compute method
    if is_dask_collection(dask_obj) and hasattr(dask_obj, 'compute'):
        logger.info(f"Starting computation: {desc}")
        with ProgressBar(dt=1.0): # Update progress every 1 second
            result = dask_obj.compute()
        logger.success(f"Computation finished: {desc}")
        return result
    else:
        # If it's not a Dask collection or doesn't have compute, assume it's already computed
        logger.info(f"Skipping computation (already computed?): {desc}")
        return dask_obj

def merge_data(
    demand_ddf: dd.DataFrame,
    metadata_df: pd.DataFrame,
    weather_ddf: dd.DataFrame,
    npartitions: int | None = None
) -> dd.DataFrame | None:
    """
    Merges demand, metadata, and weather data using Dask.

    Args:
        demand_ddf: Dask DataFrame for demand data.
        metadata_df: Pandas DataFrame for metadata.
        weather_ddf: Dask DataFrame for weather data.
        npartitions: Number of partitions for the final Dask DataFrame.

    Returns:
        A merged Dask DataFrame containing demand, metadata, and weather information,
        or None if merging fails.
    """
    logger.info("--- Starting Data Merging Process ---")

    # --- 1. Log Initial Shapes & Dtypes (Data already loaded) ---
    logger.info("Using pre-loaded dataframes.")
    try:
        demand_len = compute_with_progress(len(demand_ddf), desc="Initial demand length")
        weather_len = compute_with_progress(len(weather_ddf), desc="Initial weather length")
        logger.info(f"Initial shapes: Demand ~{demand_len}, Metadata {metadata_df.shape}, Weather ~{weather_len}")
    except Exception as e:
         logger.warning(f"Could not compute initial lengths: {e}")
         logger.info(f"Initial partitions: Demand {demand_ddf.npartitions}, Weather {weather_ddf.npartitions}")

    logger.debug(f"Demand partitions: {demand_ddf.npartitions}, Weather partitions: {weather_ddf.npartitions}")
    logger.debug(f"Demand dtypes:\n{demand_ddf.dtypes}")
    logger.debug(f"Metadata dtypes:\n{metadata_df.dtypes}")
    logger.debug(f"Weather dtypes:\n{weather_ddf.dtypes}")

    # --- 2. Merge Demand and Metadata ---
    logger.info("Merging Demand (Dask) with Metadata (Pandas) on 'unique_id'...")
    # Keep original metadata columns needed later or for context
    metadata_cols_to_keep = ['unique_id', 'location_id', 'building_class', 'timezone', 'freq', 'latitude', 'longitude', 'location']
    # Ensure location_id is also kept as it's needed for the next merge
    if 'location_id' not in metadata_df.columns:
        logger.error("Critical: 'location_id' missing from metadata_df. Cannot proceed.")
        return None
    metadata_subset = metadata_df[[col for col in metadata_cols_to_keep if col in metadata_df.columns]].copy()

    # Ensure 'unique_id' in both Dask and Pandas are compatible (object/string is safest for merge)
    try:
        if demand_ddf['unique_id'].dtype != 'object':
             logger.debug("Converting demand_ddf['unique_id'] to object for merge.")
             demand_ddf['unique_id'] = demand_ddf['unique_id'].astype('object')
        if metadata_subset['unique_id'].dtype != 'object':
             logger.debug("Converting metadata_subset['unique_id'] to object for merge.")
             metadata_subset['unique_id'] = metadata_subset['unique_id'].astype('object')

        # Perform the merge (Dask DataFrame with Pandas DataFrame)
        merged_demand_meta_ddf = dd.merge(
            demand_ddf,
            metadata_subset,
            on='unique_id',
            how='left' # Keep all demand records
        )
        logger.success("Demand and Metadata merge initiated.")
        logger.debug(f"Merged Demand-Metadata dtypes:\n{merged_demand_meta_ddf.dtypes}")
        # Clean up intermediate reference if possible
        del demand_ddf

    except Exception as e:
        logger.error(f"Error merging Demand and Metadata: {e}", exc_info=True)
        logger.error("Check 'unique_id' column existence and dtype compatibility.")
        return None

    # --- 3. Merge Result with Weather ---
    logger.info("Merging result with Weather (Dask) on 'location_id' and 'timestamp'...")

    # Prepare weather data for merge
    if 'timestamp' not in weather_ddf.columns:
        logger.warning("Timestamp not found as column in weather_ddf, attempting reset_index()...")
        weather_ddf = weather_ddf.reset_index()
        if 'timestamp' not in weather_ddf.columns:
             logger.error("Failed to find 'timestamp' column in weather_ddf even after reset_index(). Cannot merge.")
             return None

    # --- Timezone Handling (Crucial Step - Conceptual Outline) ---
    # TODO: Implement robust timezone conversion before merging.
    # Timestamps are currently 'local'. They need to be unified, preferably to UTC.
    # Approach Idea:
    # 1. For merged_demand_meta_ddf: Use 'timezone' column (from metadata).
    #    - Define a function for map_partitions that takes a partition (Pandas DF),
    #      iterates through unique timezones within the partition, and applies
    #      `tz_localize(timezone).tz_convert('UTC')`. Handle errors (e.g., invalid timezone strings).
    #    - Apply this function using `merged_demand_meta_ddf.map_partitions(...)`.
    #    - Store the result in a new 'timestamp_utc' column.
    # 2. For weather_ddf: Need to associate 'location_id' with 'timezone'.
    #    - Create a mapping: `location_tz_map = metadata_df.set_index('location_id')['timezone'].to_dict()`
    #    - Define a function for map_partitions for weather_ddf. This function needs access
    #      to `location_tz_map`. It iterates rows/groups by location_id, looks up the timezone,
    #      and converts 'timestamp' to UTC. Store in 'timestamp_utc'.
    #    - Apply using `weather_ddf.map_partitions(...)`.
    # 3. Merge on 'location_id' and the new 'timestamp_utc' column.
    # -------------------------------------------------------------
    logger.warning("Skipping timezone conversion for now. Merge is on original local timestamps, which might lead to incorrect matches if timezones differ.")

    # Temporary: Log samples before merge for debugging
    try:
        logger.info("Sample data before weather merge:")
        logger.info("Demand/Meta sample (first 5 partitions' heads):")
        sample_demand_meta = compute_with_progress(merged_demand_meta_ddf[['location_id', 'timestamp']].head(5*merged_demand_meta_ddf.npartitions, compute=False).compute(), desc="Sample Demand/Meta")
        logger.info(f"\n{sample_demand_meta}")

        logger.info("Weather sample (first 5 partitions' heads):")
        sample_weather = compute_with_progress(weather_ddf[['location_id', 'timestamp']].head(5*weather_ddf.npartitions, compute=False).compute(), desc="Sample Weather")
        logger.info(f"\n{sample_weather}")
    except Exception as e:
        logger.warning(f"Could not retrieve samples for debugging: {e}")


    # Ensure 'location_id' and 'timestamp' types are compatible
    try:
        if merged_demand_meta_ddf['location_id'].dtype != 'object' and merged_demand_meta_ddf['location_id'].dtype != 'string':
             logger.debug(f"Converting merged_demand_meta_ddf['location_id'] (dtype: {merged_demand_meta_ddf['location_id'].dtype}) to object.")
             merged_demand_meta_ddf['location_id'] = merged_demand_meta_ddf['location_id'].astype('object')
        if weather_ddf['location_id'].dtype != 'object' and weather_ddf['location_id'].dtype != 'string':
             logger.debug(f"Converting weather_ddf['location_id'] (dtype: {weather_ddf['location_id'].dtype}) to object.")
             weather_ddf['location_id'] = weather_ddf['location_id'].astype('object')

        # Ensure timestamp types match (datetime64[ns] is standard)
        if merged_demand_meta_ddf['timestamp'].dtype != weather_ddf['timestamp'].dtype:
            logger.warning(f"Timestamp dtypes differ: "
                           f"Demand/Meta ({merged_demand_meta_ddf['timestamp'].dtype}) vs "
                           f"Weather ({weather_ddf['timestamp'].dtype}). Attempting conversion to datetime64[ns]...")
            # Assuming both should be datetime64[ns]
            merged_demand_meta_ddf['timestamp'] = dd.to_datetime(merged_demand_meta_ddf['timestamp'])
            weather_ddf['timestamp'] = dd.to_datetime(weather_ddf['timestamp'])
            logger.info("Converted timestamp columns to datetime64[ns].")


        # Select relevant weather columns for the merge
        cols_already_present_in_left = set(merged_demand_meta_ddf.columns)
        merge_keys = {'location_id', 'timestamp'}

        # Select columns from weather_ddf for the right side of the merge:
        # 1. Must include the merge keys.
        # 2. Must include columns NOT present in the left dataframe.
        weather_cols_to_select_for_right_df = list(merge_keys) + [
            col for col in weather_ddf.columns
            if col not in cols_already_present_in_left # Select only NEW columns
        ]
        # Ensure uniqueness just in case merge_keys were somehow already not in left
        weather_cols_to_select_for_right_df = sorted(list(set(weather_cols_to_select_for_right_df)))

        logger.debug(f"Weather columns selected for the right side of the final merge: {weather_cols_to_select_for_right_df}")


        # Perform the final merge
        final_merged_ddf = dd.merge(
            merged_demand_meta_ddf,
            weather_ddf[weather_cols_to_select_for_right_df], # Use the correctly selected columns
            on=list(merge_keys), # Specify merge keys here
            how='left' # Keep all demand records, match weather where available
        )
        logger.success("Demand/Metadata and Weather merge initiated.")
        logger.debug(f"Final merged dtypes:\n{final_merged_ddf.dtypes}")

        # Clean up intermediate references
        del merged_demand_meta_ddf
        del weather_ddf

    except Exception as e:
        logger.error(f"Error merging with Weather data: {e}", exc_info=True)
        logger.error("Check 'location_id', 'timestamp' columns existence, dtype compatibility, and potential timezone issues.")
        return None

    # --- 4. Optional Persist & Final Checks ---
    # Persisting after merge might be beneficial if merge is expensive and successful
    # logger.info("Persisting the final merged DataFrame...")
    # with ProgressBar(dt=1.0):
    #      final_merged_ddf = final_merged_ddf.persist()
    # logger.info("Final DataFrame persisted after merge.")

    # Log estimated size after merge
    logger.info(f"Final merged DataFrame partitions: {final_merged_ddf.npartitions}")
    try:
        final_len = compute_with_progress(len(final_merged_ddf), desc="Final merged length")
        logger.info(f"Final merged length: {final_len}")
    except Exception as e:
        logger.warning(f"Could not compute final merged length: {e}")


    logger.success("Data merging process definition complete.")
    return final_merged_ddf

def analyze_missing_values(ddf: dd.DataFrame, stage: str = "Before Imputation"):
    """Analyzes and logs missing values in a Dask DataFrame."""
    logger.info(f"--- Analyzing Missing Values ({stage}) ---")
    try:
        logger.info("Calculating total row count for percentage calculation...")
        # Use len() - compute_with_progress will handle if it's int or Dask scalar
        total_rows = compute_with_progress(len(ddf), desc=f"Total row count ({stage})")
        if not isinstance(total_rows, int) or total_rows == 0:
             logger.warning(f"Invalid total row count ({total_rows}). Skipping missing value analysis.")
             return None, 0 # Return None for null counts, 0 for total_rows

        logger.info("Calculating null counts per column...")
        # Ensure ddf.isnull().sum() is passed correctly
        null_counts_dask = ddf.isnull().sum()
        null_counts = compute_with_progress(null_counts_dask, desc=f"Null counts ({stage})")

        # Check if null_counts computation was successful and is a Pandas Series
        if isinstance(null_counts, pd.Series):
            null_percentages = (null_counts / total_rows) * 100
            missing_info = pd.DataFrame({
                'Missing Count': null_counts,
                'Missing Percentage': null_percentages
            })
            missing_info = missing_info[missing_info['Missing Count'] > 0] # Show only columns with missing values
            missing_info = missing_info.sort_values(by='Missing Percentage', ascending=False)
            if not missing_info.empty:
                logger.info(f"Missing value summary ({stage}):\n{missing_info}")
            else:
                 logger.info("No missing values found.")
            return null_counts, total_rows # Return pandas Series and int
        else:
            logger.error(f"Null counts computation did not return a Pandas Series. Type: {type(null_counts)}")
            return None, total_rows # Indicate failure but keep total_rows if valid
    except Exception as e:
        logger.error(f"Error during missing value analysis ({stage}): {e}", exc_info=True)
        return None, 0 # Indicate failure

# Helper function for imputing numeric columns within a partition
def _impute_numeric_partition(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Applies interpolate, bfill, ffill to specified numeric columns."""
    if not columns:
        return df

    df = df.copy()
    try:
        df[columns] = df[columns].interpolate(method='linear', axis=0, limit_direction='both')
        df[columns] = df[columns].bfill(axis=0)
        df[columns] = df[columns].ffill(axis=0)
    except Exception as e:
        logger.error(f"Error during numeric imputation (interp/bfill/ffill) in partition: {e}")
        raise e
    return df

# Helper function for filling specific columns with a value within a partition
def _fillna_value_partition(df: pd.DataFrame, columns: list[str], value) -> pd.DataFrame:
    """Applies fillna(value) to specified columns within a Pandas DataFrame partition."""
    if not columns:
        return df
    df = df.copy()
    try:
        cols_in_df = [col for col in columns if col in df.columns]
        if cols_in_df:
            df[cols_in_df] = df[cols_in_df].fillna(value)
    except Exception as e:
        logger.error(f"Error during fillna({value}) in partition for columns {columns}: {e}")
        raise e
    return df

def impute_missing_values(ddf: dd.DataFrame, weather_column_names: list[str]) -> dd.DataFrame:
    """
    Imputes missing values in the merged DataFrame.

    Args:
        ddf: The merged Dask DataFrame.
        weather_column_names: List of column names originating from the weather data.

    Returns:
        Dask DataFrame with missing values imputed.
    """
    logger.info("--- Starting Missing Value Imputation ---")

    # 1. Analyze initial missing values
    initial_null_counts, total_rows = analyze_missing_values(ddf, stage="Before Imputation")
    if not isinstance(initial_null_counts, pd.Series) or total_rows == 0:
        logger.warning("Skipping imputation due to issues in initial analysis or empty dataframe.")
        return ddf

    # --- CRITICAL CHECK: Assess Merge Success ---
    weather_nan_check_cols = [col for col in weather_column_names if col in initial_null_counts and col not in ['location_id', 'timestamp']]
    if not weather_nan_check_cols:
         logger.warning("No weather columns found in the dataframe to check for NaNs.")
    else:
        avg_weather_missing_pct = initial_null_counts.loc[weather_nan_check_cols].sum() / (len(weather_nan_check_cols) * total_rows) * 100 if weather_nan_check_cols else 0
        logger.info(f"Average missing percentage across weather columns: {avg_weather_missing_pct:.2f}%")
        if avg_weather_missing_pct > 95: # Threshold for likely merge failure
            logger.critical(f"Extremely high average missing percentage ({avg_weather_missing_pct:.2f}%) in weather columns.")
            logger.critical("This strongly indicates the merge with weather data failed (likely due to non-matching keys or timezone issues).")
            logger.critical("Skipping imputation as it would be meaningless. Please investigate the merge step.")
            # Optionally raise an error or return the dataframe as is
            # raise ValueError("Weather data merge likely failed, too many NaNs.")
            return ddf # Return the unimputed dataframe

    imputed_ddf = ddf.copy()
    y_missing_count_flag = False
    cols_that_were_imputed = [] # Keep track of columns we attempt to impute

    # 2. Handle missing 'y' (Demand)
    y_missing_count = initial_null_counts.get('y', 0)
    if y_missing_count > 0:
        y_missing_perc = (y_missing_count / total_rows) * 100
        logger.warning(f"Found {y_missing_count} ({y_missing_perc:.4f}%) missing values in target variable 'y'.")
        logger.info("Dropping rows where 'y' is NaN.")
        imputed_ddf = imputed_ddf.dropna(subset=['y'])
        y_missing_count_flag = True
    else:
        logger.info("No missing values found in 'y'.")

    # 3. Impute Numerical Weather Features (interpolate, ffill, bfill only)
    exclude_cols = ['unique_id', 'location_id', 'timestamp', 'building_class', 'timezone', 'freq', 'weather_code', 'is_day', 'y', 'location']
    # Ensure imputed_ddf dtypes are used for check
    numeric_cols = [col for col in imputed_ddf.columns if pd.api.types.is_numeric_dtype(imputed_ddf[col].dtype) and col not in exclude_cols]
    numeric_weather_cols = [col for col in numeric_cols if col in weather_column_names]
    logger.info(f"Identified numerical weather columns potentially needing imputation: {numeric_weather_cols}")

    if numeric_weather_cols:
        # Check which *actually* have missing values based on initial analysis
        cols_to_impute_numeric = [
            col for col in numeric_weather_cols
            if initial_null_counts.get(col, 0) > 0 # Use .get for safety
        ]
        if cols_to_impute_numeric:
            logger.info(f"Applying numerical imputation (interp/ffill/bfill) via map_partitions to: {cols_to_impute_numeric}")
            meta = imputed_ddf._meta
            imputed_ddf = imputed_ddf.map_partitions(
                _impute_numeric_partition, columns=cols_to_impute_numeric, meta=meta )
            logger.info("map_partitions task for numeric imputation defined.")
            cols_that_were_imputed.extend(cols_to_impute_numeric) # Add to tracking list
        else:
            logger.info("No missing values found in identified numerical weather columns requiring imputation.")
    else:
         logger.info("No numerical weather columns identified for imputation.")

    # 4. Impute Categorical/Metadata Features
    categorical_cols_to_impute = ['building_class', 'timezone', 'freq']
    logger.info(f"Checking categorical/metadata columns {categorical_cols_to_impute} for NaNs...")
    for col in categorical_cols_to_impute:
         if col in imputed_ddf.columns and initial_null_counts.get(col, 0) > 0:
            logger.info(f"Filling NaNs in '{col}' with 'Unknown'.")
            imputed_ddf[col] = imputed_ddf[col].fillna('Unknown')
         elif col not in imputed_ddf.columns:
            logger.warning(f"Column '{col}' not found for imputation check.")

    # 5. Impute other specific columns (weather_code, is_day) using ffill/bfill only
    # Check for weather_code
    if 'weather_code' in imputed_ddf.columns and initial_null_counts.get('weather_code', 0) > 0:
        logger.info("Applying ffill then bfill to 'weather_code'...")
        imputed_ddf['weather_code'] = imputed_ddf['weather_code'].ffill()
        imputed_ddf['weather_code'] = imputed_ddf['weather_code'].bfill()
        cols_that_were_imputed.append('weather_code') # Add to tracking list

    # Check for is_day
    if 'is_day' in imputed_ddf.columns and initial_null_counts.get('is_day', 0) > 0:
        logger.info("Applying ffill then bfill to 'is_day'...")
        imputed_ddf['is_day'] = imputed_ddf['is_day'].ffill()
        imputed_ddf['is_day'] = imputed_ddf['is_day'].bfill()
        cols_that_were_imputed.append('is_day') # Add to tracking list

    # 5b. Final fillna(0) for all imputed numeric/specific columns using map_partitions
    cols_to_fillna_zero = list(set(cols_that_were_imputed)) # Get unique list of imputed cols
    if cols_to_fillna_zero:
        logger.info(f"Applying final fillna(0) via map_partitions as fallback to: {cols_to_fillna_zero}")
        meta = imputed_ddf._meta
        imputed_ddf = imputed_ddf.map_partitions(
            _fillna_value_partition,
            columns=cols_to_fillna_zero,
            value=0,
            meta=meta
        )
        logger.info("map_partitions task for final fillna(0) defined.")
    else:
        logger.info("No columns required final fillna(0) fallback.")

    # --- Persist after ALL imputation steps ---
    logger.info("Persisting DataFrame after imputation steps before final analysis...")
    try:
        with ProgressBar(dt=1.0):
             # Ensure computation happens if persist doesn't block
             imputed_ddf = imputed_ddf.persist()
        logger.success("DataFrame persisted successfully.") # Log success
    except Exception as e:
        logger.error(f"Failed to persist DataFrame after imputation: {e}", exc_info=True)
        # Decide how to proceed: return the non-persisted df or None/raise error
        logger.warning("Returning the non-persisted DataFrame due to persist error.")
        return imputed_ddf # Return the potentially large, non-persisted graph

    # 6. Final check for missing values
    if y_missing_count_flag:
        logger.info("Calculating total rows after dropping missing 'y' (post-persist)...")
        final_total_rows = compute_with_progress(len(imputed_ddf), desc="Final row count check")
    else:
        final_total_rows = total_rows
        logger.info(f"Using original row count for final analysis: {final_total_rows}")
    final_null_counts, _ = analyze_missing_values(imputed_ddf, stage="After Imputation")
    if isinstance(final_null_counts, pd.Series):
        remaining_missing = final_null_counts[final_null_counts > 0]
        if not remaining_missing.empty:
            logger.warning(f"Columns with remaining missing values after imputation:\n{remaining_missing}")
        else:
            logger.success("No missing values remain after imputation (based on computed check).")
    else:
         logger.warning("Could not verify remaining missing values due to analysis error after imputation.")

    logger.success("Missing value imputation process definition complete.")
    return imputed_ddf

# --- Usage Example ---
if __name__ == "__main__":
    logger.info("\n--- Testing Data Preprocessing Pipeline ---")

    # Optional: Setup Dask client for dashboard and better resource management
    # from dask.distributed import Client, LocalCluster
    # try:
    #     cluster = LocalCluster(n_workers=os.cpu_count() // 2, threads_per_worker=2, memory_limit='4GB')
    #     client = Client(cluster)
    #     logger.info(f"Dask dashboard link: {client.dashboard_link}")
    # except Exception as client_err:
    #     logger.warning(f"Could not start Dask client/cluster: {client_err}. Running sequentially.")
    #     client = None

    # Define number of partitions
    num_partitions = os.cpu_count() * 2 if os.cpu_count() else 4 # Adjusted default

    # --- Load Data Once with Time Filter ---
    test_start_date = '2013-03-01' # Choose a start date likely to have data
    test_end_date = '2013-03-15'   # Choose an end date
    logger.info(f"Loading initial data with time filter ({test_start_date} to {test_end_date})...")

    loaded_data = load_electricity_data(
        return_format="dask",
        npartitions=num_partitions,
        start_date=test_start_date, # <-- Pass start date
        end_date=test_end_date     # <-- Pass end date
        # Removed max_rows_per_source
    )
    if not loaded_data:
        logger.error("Initial data loading failed. Exiting.")
        sys.exit(1)

    demand_ddf, metadata_df, weather_ddf = loaded_data
    # Get weather column names from the actual loaded dataframe
    weather_cols = weather_ddf.columns.tolist() if weather_ddf is not None else []
    logger.info("Initial data loaded successfully.")
    # Log loaded shapes again
    try:
        # Compute lengths of the *filtered* dataframes
        demand_len_main = compute_with_progress(len(demand_ddf), desc="Loaded filtered demand length")
        weather_len_main = compute_with_progress(len(weather_ddf), desc="Loaded filtered weather length")
        logger.info(f"Loaded filtered shapes: Demand ~{demand_len_main}, Metadata {metadata_df.shape}, Weather ~{weather_len_main}")
    except Exception as e:
        logger.warning(f"Could not compute loaded filtered lengths: {e}")


    # 1. Merge Data - Pass loaded dataframes
    logger.info("\n--- Starting Data Merging Step ---") # Added log separator
    merged_ddf = merge_data(
        demand_ddf=demand_ddf,
        metadata_df=metadata_df,
        weather_ddf=weather_ddf,
        npartitions=num_partitions # Pass the potentially recalculated npartitions
    )
    # Delete original references if merge was successful and they are no longer needed
    if merged_ddf is not None:
         logger.info("Deleting original demand and weather Dask dataframe references after successful merge setup.")
         del demand_ddf, weather_ddf # Keep metadata_df potentially for timezone lookup later
         # del metadata_df # Can delete this too if definitely not needed later in this scope

    if merged_ddf is not None:
        logger.info("\n--- Starting Imputation Step ---")
        # 2. Impute Missing Values
        imputed_ddf = impute_missing_values(merged_ddf, weather_column_names=weather_cols)

        # Make sure imputed_ddf is not None before proceeding
        if imputed_ddf is not None:
            logger.info("\n--- Inspecting Imputed Dask DataFrame ---")
            logger.info(f"Number of partitions: {imputed_ddf.npartitions}")

            # --- Example Computations on Imputed Data ---
            logger.info("\n--- Performing Example Computations on Imputed Data ---")

            # Compute head
            try:
                logger.info("Computing head of the imputed DataFrame...")
                head_df_imputed = compute_with_progress(imputed_ddf.head(5), desc="Head computation (imputed)")
                if isinstance(head_df_imputed, pd.DataFrame):
                    logger.info(f"Head of imputed DataFrame:\n{head_df_imputed}")
                else:
                    logger.warning(f"Head computation returned unexpected type: {type(head_df_imputed)}")
            except Exception as e:
                logger.error(f"Failed to compute head on imputed data: {e}", exc_info=True)

            # Verify no NaNs remain in 'y' (if it wasn't dropped entirely)
            if 'y' in imputed_ddf.columns:
                 try:
                     logger.info("Verifying no NaNs remain in 'y'...")
                     y_nan_count_after = compute_with_progress(
                         imputed_ddf['y'].isnull().sum(),
                         desc="NaN count in 'y' after imputation"
                     )
                     if isinstance(y_nan_count_after, (int, np.integer)) and y_nan_count_after == 0:
                         logger.success("Confirmed: No NaN values remaining in 'y'.")
                     else:
                         logger.error(f"Verification Error: {y_nan_count_after} NaN values found in 'y' after imputation.")
                 except Exception as e:
                      logger.error(f"Failed to verify NaNs in 'y': {e}", exc_info=True)
            else:
                 logger.info("Column 'y' not found in final dataframe (might have been dropped or not selected).")


            # Verify NaNs in a key imputed weather column (only if merge seemed successful)
            # Find a weather column that likely exists and was numeric
            potential_weather_col = 'temperature_2m' # Example
            if potential_weather_col in imputed_ddf.columns and potential_weather_col in weather_cols:
                 try:
                     logger.info(f"Verifying NaNs remain in '{potential_weather_col}' (should be 0 if imputation worked)...")
                     temp_nan_count_after = compute_with_progress(
                         imputed_ddf[potential_weather_col].isnull().sum(),
                         desc=f"NaN count in '{potential_weather_col}' after imputation"
                     )
                     if isinstance(temp_nan_count_after, (int, np.integer)) and temp_nan_count_after == 0:
                         logger.success(f"Confirmed: No NaN values remaining in '{potential_weather_col}'.")
                     elif isinstance(temp_nan_count_after, (int, np.integer)):
                          logger.warning(f"Verification Warning: {temp_nan_count_after} NaN values found in '{potential_weather_col}' after imputation.")
                     else:
                          logger.error(f"Verification Error: Unexpected result type {type(temp_nan_count_after)} for NaN count in '{potential_weather_col}'.")

                 except Exception as e:
                      logger.error(f"Failed to verify NaNs in '{potential_weather_col}': {e}", exc_info=True)
            else:
                 logger.info(f"Skipping NaN verification for '{potential_weather_col}' as it's not present.")


            logger.success("Data preprocessing test (Merge + Impute) finished.")
        else:
             logger.error("Imputation step failed or was skipped. Cannot proceed with final checks.")
    else:
        logger.error("Data merging failed, skipping imputation test.")

    # Optional: Shutdown Dask client
    # if 'client' in locals() and client:
    #     logger.info("Shutting down Dask client and cluster...")
    #     client.close()
    #     if 'cluster' in locals() and cluster: # Check if cluster exists
    #         cluster.close()
    #     logger.info("Dask client and cluster shut down.")
