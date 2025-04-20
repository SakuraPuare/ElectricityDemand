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
    npartitions: int | None = None,
    optimize_meta_weather: bool = True,
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
        optimize_meta_weather: Whether to optimize dtypes for metadata (Pandas)
                               and weather (Pandas only).
        demand_storage_options: Storage options for dask/fsspec for demand parquet URL.
        weather_storage_options: Storage options for dask/fsspec for weather parquet URL.

    Returns:
        - If return_format='pandas': tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None
        - If return_format='dask': tuple[dd.DataFrame, pd.DataFrame, dd.DataFrame] | None
          (Metadata remains Pandas)
    """
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
        metadata_ds = load_dataset(
            dataset_name, name="metadata", split="train",
            features=metadata_features, trust_remote_code=True,
            token=hf_token
        )
        metadata_df = metadata_ds.to_pandas()
        if optimize_meta_weather:
            metadata_df = _optimize_pandas_dtypes(metadata_df)
        logger.success("Metadata loaded and processed as Pandas DataFrame.")
        del metadata_ds

        # --- Define relative paths within the HF dataset repo ---
        demand_filename = "data/demand.parquet"
        weather_filename = "data/weather.parquet"

        # --- Load Demand and Weather ---
        if return_format == "dask":
            if npartitions is None:
                npartitions = os.cpu_count() * 2 if os.cpu_count() else 4
            logger.info(f"Loading Demand and Weather as Dask DataFrames ({npartitions} partitions)...")

            # --- Download/Cache and get local path for Demand ---
            logger.info(f"Ensuring '{demand_filename}' is cached locally...")
            try:
                demand_local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=demand_filename,
                    repo_type='dataset',
                    token=hf_token # Pass token for potential private datasets
                )
                logger.info(f"Using cached Demand file: {demand_local_path}")
            except Exception as download_err:
                logger.error(f"Failed to download/cache {demand_filename}: {download_err}", exc_info=True)
                return None

            # --- Download/Cache and get local path for Weather ---
            logger.info(f"Ensuring '{weather_filename}' is cached locally...")
            try:
                weather_local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=weather_filename,
                    repo_type='dataset',
                    token=hf_token
                )
                logger.info(f"Using cached Weather file: {weather_local_path}")
            except Exception as download_err:
                logger.error(f"Failed to download/cache {weather_filename}: {download_err}", exc_info=True)
                return None

            # --- Read Parquet files from local cache using Dask ---
            demand_dtypes = {'unique_id': 'category', 'y': 'float32'}
            logger.info(f"Reading Demand Dask DataFrame from local path: {demand_local_path}")
            demand_ddf = dd.read_parquet(
                demand_local_path,
                engine='pyarrow',
                dtype=demand_dtypes
                # No storage_options needed for local files
            )
            logger.success(f"Demand Dask DataFrame created (Initial partitions: {demand_ddf.npartitions}).")


            logger.info(f"Reading Weather Dask DataFrame from local path: {weather_local_path}")
            weather_ddf = dd.read_parquet(
                weather_local_path,
                engine='pyarrow'
                # Let Dask infer types initially for local read
            )
            logger.info(f"Weather Dask DataFrame read with inferred types (Partitions: {weather_ddf.npartitions}).")
            logger.info(f"Inferred weather dtypes:\n{weather_ddf.dtypes}")

            # Apply type conversions *after* loading
            logger.info("Applying dtype conversions to Weather Dask DataFrame...")
            weather_dtypes_apply = {}
            if 'location_id' in weather_ddf.columns:
                 weather_dtypes_apply['location_id'] = 'category'
            for col, dtype in weather_ddf.dtypes.items():
                if col == 'location_id': continue
                if 'float' in str(dtype): weather_dtypes_apply[col] = 'float32'
                elif 'int' in str(dtype): weather_dtypes_apply[col] = 'int32'
            if weather_dtypes_apply:
                 weather_ddf = weather_ddf.astype(weather_dtypes_apply)
                 logger.info("Applied specified dtypes to weather Dask DataFrame.")
            else:
                 logger.info("No further specific dtype conversions applied to weather Dask DataFrame.")
            logger.success("Weather Dask DataFrame processing finished.")

            return demand_ddf, metadata_df, weather_ddf

        elif return_format == "pandas":
            # For Pandas, we can still use load_dataset which handles caching internally
            logger.info("Loading demand data via datasets library (handles caching)...")
            demand_ds = load_dataset(
                dataset_name, name="demand", split="train",
                trust_remote_code=True, token=hf_token
            )
            demand_df = demand_ds.to_pandas()
            del demand_ds
            logger.success("Demand data loaded as Pandas.")

            logger.info("Loading weather data via datasets library (handles caching)...")
            weather_ds = load_dataset(
                dataset_name, name="weather", split="train",
                trust_remote_code=True, token=hf_token
            )
            weather_df = weather_ds.to_pandas()
            del weather_ds
            logger.success("Weather data loaded as Pandas.")

            if optimize_meta_weather:
                 logger.info("Optimizing Weather DataFrame dtypes (Pandas)...")
                 weather_df = _optimize_pandas_dtypes(weather_df)

            logger.warning("Optimizing large Demand DataFrame dtypes in Pandas format can be memory intensive.")
            try:
                demand_df['y'] = pd.to_numeric(demand_df['y'], downcast='float')
                demand_df['unique_id'] = demand_df['unique_id'].astype('category')
            except Exception as opt_err:
                logger.error(f"Could not optimize demand_df dtypes in Pandas mode: {opt_err}")

            return demand_df, metadata_df, weather_df

        else:
            logger.error(f"Invalid return_format specified: '{return_format}'. Choose 'pandas' or 'dask'.")
            return None

    except ImportError as ie:
         # Ensure huggingface_hub is mentioned here
         logger.error(f"Import error: {ie}. Ensure 'datasets', 'pandas', 'pyarrow', 'dask', 'fsspec', 'huggingface_hub' are installed.")
         return None
    except Exception as e:
        # Handle potential huggingface_hub download errors if needed, though caught above now
        logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        logger.error("Check internet connection, dataset name, permissions, and memory.")
        return None

# --- Usage Example ---
if __name__ == "__main__":
    logger.info("\n--- Testing Dask Loader with Caching ---")
    # Ensure libraries are reasonably up-to-date
    logger.info("Consider updating libraries: pip install --upgrade pandas pyarrow 'dask[dataframe]' huggingface_hub datasets")
    loaded_data_dask = load_electricity_data(return_format="dask")
    if loaded_data_dask:
        demand_ddf, metadata_df_dask, weather_ddf = loaded_data_dask
        logger.info("\n--- Dask Demand DataFrame Info ---")
        logger.info(f"Partitions: {demand_ddf.npartitions}")
        logger.info(f"dtypes:\n{demand_ddf.dtypes}")
        logger.info("\n--- Dask Weather DataFrame Info ---")
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

        logger.success("Dask loading test finished.")
    else:
        logger.error("Failed to load dataset as Dask.")

    # --- Optional: Test Pandas Loader (uses datasets cache) ---
    # logger.info("\n--- Testing Pandas Loader (uses datasets cache) ---")
    # loaded_data_pd = load_electricity_data(return_format="pandas")
    # if loaded_data_pd:
    #     # ... info checks ...
    #     logger.success("Pandas loading test finished.")
    # else:
    #     logger.error("Failed to load dataset as Pandas.")
