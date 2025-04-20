from datasets import load_dataset, Features, Value # Removed Sequence for simplicity now
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from loguru import logger
import os
import numpy as np

# Import huggingface_hub utilities
try:
    from huggingface_hub import HfFolder, hf_hub_download # Import hf_hub_download
    # Check if user is logged in, needed for private datasets or potentially rate limits
    hf_token = HfFolder.get_token()
    if not hf_token:
        logger.warning("Hugging Face token not found. Login using 'huggingface-cli login'.")
        # Depending on dataset visibility, download might still work or fail later.
except ImportError:
    logger.error("huggingface_hub not found. Please install it: pip install huggingface_hub")
    exit()

# Make sure fsspec is installed for local file reading by Dask
try:
    import fsspec
except ImportError:
    logger.error("fsspec not found. Please install it: pip install fsspec")
    exit()

# 配置 loguru
logger.add("data_loading.log", rotation="500 MB") # 将日志记录到文件

def _optimize_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to optimize dtypes for smaller Pandas dataframes like metadata."""
    logger.debug("Optimizing Pandas dtypes...")
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        # Check range before downcasting int
        min_val, max_val = df[col].min(), df[col].max()
        if pd.isna(min_val) or pd.isna(max_val): # Skip if column is all NaN
             continue
        if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            df[col] = pd.to_numeric(df[col], downcast='integer') # Tries int32, int16 etc.
        # else keep as int64
    for col in df.select_dtypes(include=['object']).columns:
        try:
             # Only convert if nunique is feasible to compute and ratio is low
             if df[col].nunique() / len(df[col]) < 0.5:
                 df[col] = df[col].astype('category')
                 logger.debug(f"Converted column '{col}' to category.")
        except TypeError: # Handle potential unhashable types in object columns
             logger.warning(f"Could not check nunique or convert column '{col}' to category due to unhashable types.")
    logger.debug("Pandas dtype optimization finished.")
    return df

def load_electricity_data(
    dataset_name: str = "EDS-lab/electricity-demand",
    return_format: str = "pandas",
    npartitions: int | None = None, # Desired number of partitions for Dask DataFrames. Adjust based on core count and I/O speed for potential performance tuning.
    optimize_meta_weather: bool = True,
    start_date: str | pd.Timestamp | None = None, # <-- 新增参数
    end_date: str | pd.Timestamp | None = None,   # <-- 新增参数
    demand_storage_options: dict | None = None, # For potential authentication if needed
    weather_storage_options: dict | None = None
) -> tuple | None:
    """
    Loads demand, metadata, and weather data.

    Uses huggingface_hub to download/cache Parquet files locally first,
    then loads them using dd.read_parquet (for Dask) or pd.read_parquet (via datasets for Pandas).
    Metadata is always loaded via 'datasets' library and returned as Pandas.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub (e.g., 'Org/DatasetName').
        return_format: Output format ('pandas' or 'dask').
        npartitions: Desired number of partitions for Dask DataFrames.
                     Defaults to os.cpu_count() * 2. Increasing this might improve speed
                     on machines with many cores and fast I/O, but excessive partitions
                     can add overhead. Experimentation might be needed.
        optimize_meta_weather: Whether to optimize dtypes for metadata (Pandas)
                               and weather (Pandas only).
        start_date: (Optional) Start date for filtering demand and weather data (inclusive).
                    Example: '2013-01-01'.
        end_date: (Optional) End date for filtering demand and weather data (inclusive).
                  Example: '2013-01-31'.
        demand_storage_options: Storage options for dask/fsspec for demand parquet URL.
        weather_storage_options: Storage options for dask/fsspec for weather parquet URL.

    Returns:
        - If return_format='pandas': tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None
        - If return_format='dask': tuple[dd.DataFrame, pd.DataFrame, dd.DataFrame] | None
          (Metadata remains Pandas)
    """
    if start_date or end_date:
         logger.info(f"Applying time filter: Start={start_date}, End={end_date}")

    logger.info(f"Attempting to load data from '{dataset_name}' (format: {return_format})...")

    try:
        # --- Load Metadata (Always Pandas via datasets) ---
        logger.info("Loading metadata via datasets library...")
        metadata_features = Features({
            'unique_id': Value('string'), 'dataset': Value('string'), 'building_id': Value('string'),
            'location_id': Value('string'), 'latitude': Value('float32'), 'longitude': Value('float32'),
            'location': Value('string'), 'timezone': Value('string'), 'freq': Value('string'),
            'building_class': Value('string'), 'cluster_size': Value('int32'),
        })
        # Getting token isn't strictly necessary here if logged in, but good practice
        hf_token = HfFolder.get_token()
        logger.debug(f"Using Hugging Face token: {'Yes' if hf_token else 'No'}")
        logger.debug(f"Calling load_dataset for metadata: dataset_name='{dataset_name}', name='metadata', split='train'")
        try:
            metadata_ds = load_dataset(
                dataset_name, name="metadata", split="train",
                features=metadata_features, trust_remote_code=True,
                token=hf_token
            )
            logger.debug("load_dataset for metadata successful.")
        except Exception as e:
            logger.error(f"Failed to load metadata dataset using load_dataset: {e}", exc_info=True)
            logger.error("Check dataset name ('{dataset_name}'), 'metadata' configuration, split 'train', and HF token/permissions.")
            return None # Exit early if metadata fails

        metadata_df = metadata_ds.to_pandas()
        logger.debug("Converted metadata dataset to Pandas DataFrame.")
        if optimize_meta_weather:
            metadata_df = _optimize_pandas_dtypes(metadata_df)
        logger.success(f"Metadata loaded ({len(metadata_df)} rows) and processed as Pandas DataFrame.") # Added row count log
        del metadata_ds

        # --- Define relative paths within the HF dataset repo ---
        demand_filename = "data/demand.parquet"
        weather_filename = "data/weather.parquet"
        logger.debug(f"Target demand file within repo: '{demand_filename}'")
        logger.debug(f"Target weather file within repo: '{weather_filename}'")


        # --- Load Demand and Weather ---
        if return_format == "dask":
            if npartitions is None:
                # Default logic: good starting point
                npartitions = os.cpu_count() * 2 if os.cpu_count() else 4
                logger.info(f"Defaulting npartitions to {npartitions} for Dask DataFrames.")
            # Ensure npartitions is at least 1
            npartitions = max(1, npartitions)

            logger.info(f"Loading Demand and Weather as Dask DataFrames ({npartitions} partitions)...")

            # --- Download/Cache and get local path for Demand ---
            logger.info(f"Ensuring '{demand_filename}' is cached locally via hf_hub_download...")
            logger.debug(f"Calling hf_hub_download for demand: repo_id='{dataset_name}', filename='{demand_filename}'")
            try:
                demand_local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=demand_filename,
                    repo_type='dataset',
                    token=hf_token # Pass token for potential private datasets
                )
                logger.success(f"Demand file is available locally at: {demand_local_path}")
            except Exception as download_err:
                logger.error(f"Failed to download/cache '{demand_filename}' using hf_hub_download: {download_err}", exc_info=True)
                logger.error(f"Check repository '{dataset_name}', filename '{demand_filename}', connection, and permissions.")
                return None

            # --- Download/Cache and get local path for Weather ---
            logger.info(f"Ensuring '{weather_filename}' is cached locally via hf_hub_download...")
            logger.debug(f"Calling hf_hub_download for weather: repo_id='{dataset_name}', filename='{weather_filename}'")
            try:
                weather_local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=weather_filename,
                    repo_type='dataset',
                    token=hf_token
                )
                logger.success(f"Weather file is available locally at: {weather_local_path}")
            except Exception as download_err:
                logger.error(f"Failed to download/cache '{weather_filename}' using hf_hub_download: {download_err}", exc_info=True)
                logger.error(f"Check repository '{dataset_name}', filename '{weather_filename}', connection, and permissions.")
                return None

            # --- Read Parquet files from local cache using Dask ---
            demand_dtypes = {'unique_id': 'category', 'y': 'float32'}
            logger.info(f"Reading Demand Dask DataFrame from local path: {demand_local_path}")
            try:
                demand_ddf = dd.read_parquet(
                    demand_local_path,
                    engine='pyarrow',
                )
                # Ensure timestamp column exists before filtering
                if 'timestamp' not in demand_ddf.columns:
                    logger.error("Critical: 'timestamp' column not found in demand data.")
                    return None
                demand_ddf['timestamp'] = dd.to_datetime(demand_ddf['timestamp']) # Ensure datetime type

                # Apply time filter *after* reading, *before* repartition/head
                if start_date:
                    demand_ddf = demand_ddf[demand_ddf['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    # Add 1 day to end_date to make it inclusive for date-only strings
                    effective_end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                    demand_ddf = demand_ddf[demand_ddf['timestamp'] < effective_end_date]

                # Apply specific dtypes *after* filtering
                demand_ddf = demand_ddf.astype(demand_dtypes)

                demand_ddf = demand_ddf.repartition(npartitions=npartitions)
                logger.success(f"Demand Dask DataFrame created/filtered and repartitioned to {demand_ddf.npartitions} partitions.")
            except Exception as read_err:
                logger.error(f"Failed to read/process demand parquet file '{demand_local_path}': {read_err}", exc_info=True)
                return None


            logger.info(f"Reading Weather Dask DataFrame from local path: {weather_local_path}")
            try:
                weather_ddf = dd.read_parquet(
                    weather_local_path,
                    engine='pyarrow'
                )
                if 'timestamp' not in weather_ddf.columns:
                     logger.error("Critical: 'timestamp' column not found in weather data.")
                     return None
                weather_ddf['timestamp'] = dd.to_datetime(weather_ddf['timestamp']) # Ensure datetime type

                # Apply time filter *after* reading, *before* repartition/head
                if start_date:
                    weather_ddf = weather_ddf[weather_ddf['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    effective_end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                    weather_ddf = weather_ddf[weather_ddf['timestamp'] < effective_end_date]

                weather_ddf = weather_ddf.repartition(npartitions=npartitions)
                logger.info(f"Weather Dask DataFrame read/filtered and repartitioned to {weather_ddf.npartitions} partitions.")
            except Exception as read_err:
                 logger.error(f"Failed to read/process weather parquet file '{weather_local_path}': {read_err}", exc_info=True)
                 return None

            # Apply type conversions *after* loading and filtering
            logger.info("Applying dtype conversions to Weather Dask DataFrame...")
            weather_dtypes_apply = {}
            if 'location_id' in weather_ddf.columns:
                 weather_dtypes_apply['location_id'] = 'category'
            for col, dtype in weather_ddf.dtypes.items():
                if col == 'location_id': continue
                if 'float' in str(dtype): weather_dtypes_apply[col] = 'float32'
                elif 'int' in str(dtype): weather_dtypes_apply[col] = 'int32'
                # Keep timestamp columns as they are for now
                elif 'datetime' in str(dtype): continue
                # Potentially handle other types if needed
            if weather_dtypes_apply:
                 logger.debug(f"Applying dtypes to weather: {weather_dtypes_apply}")
                 weather_ddf = weather_ddf.astype(weather_dtypes_apply)
                 logger.info("Applied specified dtypes to weather Dask DataFrame.")
                 logger.debug(f"Weather dtypes after conversion:\n{weather_ddf.dtypes}")
            else:
                 logger.info("No specific dtype conversions identified or applied to weather Dask DataFrame.")
            logger.success("Weather Dask DataFrame processing finished.")


            return demand_ddf, metadata_df, weather_ddf

        elif return_format == "pandas":
            # For Pandas, we can still use load_dataset which handles caching internally
            logger.info("Loading demand data via datasets library (handles caching)...")
            logger.debug(f"Calling load_dataset for demand: dataset_name='{dataset_name}', name='demand', split='train'")
            try:
                demand_ds = load_dataset(
                    dataset_name, name="demand", split="train",
                    trust_remote_code=True, token=hf_token
                )
                logger.debug("load_dataset for demand successful.")
            except Exception as e:
                logger.error(f"Failed to load demand dataset using load_dataset: {e}", exc_info=True)
                logger.error("Check dataset name ('{dataset_name}'), 'demand' configuration, split 'train', and HF token/permissions.")
                return None # Exit early if demand fails

            demand_df = demand_ds.to_pandas()
            logger.debug("Converted demand dataset to Pandas DataFrame.")
            del demand_ds
            # Ensure datetime type
            if 'timestamp' in demand_df.columns:
                 demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'])
                 # Apply time filter
                 if start_date:
                     demand_df = demand_df[demand_df['timestamp'] >= pd.to_datetime(start_date)]
                 if end_date:
                     effective_end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                     demand_df = demand_df[demand_df['timestamp'] < effective_end_date]
            else:
                logger.error("Critical: 'timestamp' column not found in demand data (Pandas).")
                return None

            logger.success(f"Demand data loaded and filtered as Pandas ({len(demand_df)} rows).")

            logger.info("Loading weather data via datasets library (handles caching)...")
            logger.debug(f"Calling load_dataset for weather: dataset_name='{dataset_name}', name='weather', split='train'")
            try:
                weather_ds = load_dataset(
                    dataset_name, name="weather", split="train",
                    trust_remote_code=True, token=hf_token
                )
                logger.debug("load_dataset for weather successful.")
            except Exception as e:
                logger.error(f"Failed to load weather dataset using load_dataset: {e}", exc_info=True)
                logger.error("Check dataset name ('{dataset_name}'), 'weather' configuration, split 'train', and HF token/permissions.")
                return None # Exit early if weather fails

            weather_df = weather_ds.to_pandas()
            logger.debug("Converted weather dataset to Pandas DataFrame.")
            del weather_ds
            # Ensure datetime type
            if 'timestamp' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
                # Apply time filter
                if start_date:
                    weather_df = weather_df[weather_df['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    effective_end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                    weather_df = weather_df[weather_df['timestamp'] < effective_end_date]
            else:
                 logger.error("Critical: 'timestamp' column not found in weather data (Pandas).")
                 return None

            logger.success(f"Weather data loaded and filtered as Pandas ({len(weather_df)} rows).")

            if optimize_meta_weather:
                 logger.info("Optimizing Weather DataFrame dtypes (Pandas)...")
                 weather_df = _optimize_pandas_dtypes(weather_df)

            logger.warning("Optimizing large Demand DataFrame dtypes in Pandas format can be memory intensive.")
            try:
                # Optimize demand dtypes after loading and filtering
                logger.debug("Optimizing Demand DataFrame dtypes (Pandas)...")
                demand_df['y'] = pd.to_numeric(demand_df['y'], downcast='float')
                # unique_id might not be efficient as category if limited rows have many unique values
                if len(demand_df) > 0 and demand_df['unique_id'].nunique() / len(demand_df) < 0.8: # Adjust threshold if needed
                     demand_df['unique_id'] = demand_df['unique_id'].astype('category')
                else:
                     logger.debug("Skipping unique_id to category conversion due to high cardinality in limited data.")
                # Convert timestamp if it's not already datetime
                if not pd.api.types.is_datetime64_any_dtype(demand_df['timestamp']):
                    logger.debug("Converting demand 'timestamp' column to datetime...")
                    demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'])
                logger.debug("Demand DataFrame dtypes optimized.")
            except Exception as opt_err:
                logger.error(f"Could not optimize demand_df dtypes in Pandas mode: {opt_err}", exc_info=True)

            return demand_df, metadata_df, weather_df

        else:
            logger.error(f"Invalid return_format specified: '{return_format}'. Choose 'pandas' or 'dask'.")
            return None

    except ImportError as ie:
         # Ensure huggingface_hub is mentioned here
         logger.error(f"Import error: {ie}. Ensure 'datasets', 'pandas', 'pyarrow', 'dask', 'fsspec', 'huggingface_hub' are installed.", exc_info=True)
         return None
    except Exception as e:
        # Generic catch-all, should be hit less often now
        logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        logger.error("Check internet connection, dataset name, permissions, and memory.")
        return None

# --- Usage Example ---
if __name__ == "__main__":
    # --- Test with Time Filter ---
    test_start_date = '2013-03-01' # Example start date
    test_end_date = '2013-03-15'   # Example end date
    logger.info(f"\n--- Testing Dask Loader with Time Filter ({test_start_date} to {test_end_date}) ---")
    loaded_data_dask_filtered = load_electricity_data(
        return_format="dask",
        start_date=test_start_date,
        end_date=test_end_date
    )
    if loaded_data_dask_filtered:
        demand_ddf_filt, metadata_df_filt, weather_ddf_filt = loaded_data_dask_filtered
        logger.info("\n--- Dask Demand DataFrame Info (Filtered) ---")
        logger.info(f"Partitions: {demand_ddf_filt.npartitions}")
        logger.info(f"dtypes:\n{demand_ddf_filt.dtypes}")
        logger.info("\n--- Computing Dask Length (Filtered) ---")
        with ProgressBar():
            dask_len_filt = len(demand_ddf_filt) # Compute length of the filtered data
        logger.info(f"Length: {dask_len_filt}")
        logger.success("Dask filtered loading test finished.")
    else:
        logger.error("Failed to load filtered dataset as Dask.")

    # --- Optional: Test Full Dask Load (Original Test) ---
    logger.info("\n--- Testing Full Dask Loader ---")
    loaded_data_dask = load_electricity_data(return_format="dask")
    if loaded_data_dask:
        demand_ddf, metadata_df_dask, weather_ddf = loaded_data_dask
        logger.info("\n--- Dask Demand DataFrame Info (Full) ---")
        logger.info(f"Partitions: {demand_ddf.npartitions}")
        logger.info(f"dtypes:\n{demand_ddf.dtypes}")
        logger.info("\n--- Dask Weather DataFrame Info (Full) ---")
        logger.info(f"Partitions: {weather_ddf.npartitions}")
        logger.info(f"dtypes:\n{weather_ddf.dtypes}")
        logger.info("\n--- Pandas Metadata DataFrame (from Dask load) Info ---")
        metadata_df_dask.info(verbose=False, show_counts=True)

        # Example Computation: Length
        logger.info("\n--- Computing Dask Length (Example) ---")
        logger.info("Computing Dask Demand DataFrame length...")
        with ProgressBar():
            dask_len = len(demand_ddf)
        logger.info(f"Length: {dask_len}")

        # Example Computation: Head
        logger.info("\n--- Computing Dask Head (Example) ---")
        logger.info("Computing Dask Demand DataFrame head...")
        with ProgressBar():
            dask_head = demand_ddf.head()
        logger.info(f"Head:\n{dask_head}")

        logger.success("Full Dask loading test finished.")
    else:
        logger.error("Failed to load full dataset as Dask.")

    # --- Optional: Test Pandas Loader (uses datasets cache) ---
    # logger.info("\n--- Testing Pandas Loader (uses datasets cache) ---")
    # loaded_data_pd = load_electricity_data(return_format="pandas")
    # if loaded_data_pd:
    #     # ... info checks ...
    #     logger.success("Pandas loading test finished.")
    # else:
    #     logger.error("Failed to load dataset as Pandas.")
