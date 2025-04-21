import sys
import time
from pathlib import Path
import joblib  # 用于保存模型

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split # 临时占位，后面会用时间分割
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- 项目设置 ---
try:
    from electricitydemand.utils.project_utils import get_project_root, setup_project_paths, create_spark_session, stop_spark_session
    from electricitydemand.utils.log_utils import setup_logger
except ImportError as e:
    print(f"Error importing project utils: {e}", file=sys.stderr)
    sys.exit(1)

project_root = get_project_root()
src_path, data_dir, logs_dir, plots_dir = setup_project_paths(project_root)
models_dir = project_root / "models" # 创建模型保存目录
models_dir.mkdir(parents=True, exist_ok=True)


# --- 配置日志 ---
log_prefix = Path(__file__).stem # 使用 4_run_model_training 作为前缀
try:
    setup_logger(log_file_prefix=log_prefix, logs_dir=logs_dir, level="INFO")
except NameError:
    logger.add(sys.stderr, level="INFO")
    logger.warning("Using basic stderr logging due to import error.")

logger.info(f"项目根目录：{project_root}")
logger.info(f"数据目录：{data_dir}")
logger.info(f"日志目录：{logs_dir}")
logger.info(f"绘图目录：{plots_dir}")
logger.info(f"模型目录：{models_dir}")

# --- 数据文件路径 ---
# 检查是否存在抽样特征文件，优先使用它
sample_fraction_str = "0p01" # 假设我们用了 1% 抽样，需要根据实际情况调整
features_sampled_path = data_dir / f"features_sampled_{sample_fraction_str}.parquet"
features_full_path = data_dir / "features.parquet"

if features_sampled_path.exists():
    features_input_path = features_sampled_path
    logger.info(f"使用抽样特征数据: {features_input_path}")
else:
    features_input_path = features_full_path
    logger.info(f"使用全量特征数据: {features_input_path}")

# ======================================================================
# ==                      Helper Functions                          ==
# ======================================================================

def time_based_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """按时间戳分割 Pandas DataFrame"""
    logger.info(f"开始基于时间戳分割数据，测试集比例: {test_ratio:.1%}")
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame 中缺少 'timestamp' 列，无法进行时间分割。")
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logger.warning("Timestamp 列不是 datetime 类型，尝试转换...")
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            logger.error(f"转换 timestamp 列为 datetime 失败: {e}")
            raise

    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    split_index = int(len(df_sorted) * (1 - test_ratio))

    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    logger.info(f"训练集时间范围: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    logger.info(f"测试集时间范围: {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")
    logger.info(f"训练集大小: {len(train_df):,} 行")
    logger.info(f"测试集大小: {len(test_df):,} 行")

    return train_df, test_df

def evaluate_model(y_true, y_pred, model_name="Model"):
    """计算并记录模型的评估指标"""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    logger.success(f"--- {model_name} 评估结果 ---")
    logger.success(f"  均方根误差 (RMSE): {rmse:.4f}")
    logger.success(f"  平均绝对误差 (MAE): {mae:.4f}")
    logger.success("---------------------------")
    return rmse, mae

# ======================================================================
# ==                   Main Execution Function                      ==
# ======================================================================

def run_model_training():
    """加载特征数据，训练模型，评估并保存模型"""
    logger.info("=====================================================")
    logger.info("=== 开始执行 模型训练脚本 ===")
    logger.info("=====================================================")
    start_run_time = time.time()
    spark = None

    try:
        # --- 创建 SparkSession (仅用于加载数据) ---
        logger.info("创建 SparkSession (用于加载数据)...")
        spark = create_spark_session(app_name="ElectricityDemandModelTrainingLoader")
        logger.info("SparkSession 创建成功。")

        # --- 步骤 1: 加载特征数据 ---
        logger.info(f"--- 步骤 1: 加载特征数据 {features_input_path} ---")
        sdf = spark.read.parquet(str(features_input_path))
        logger.info("特征数据加载成功。")
        logger.info("Schema:")
        sdf.printSchema()
        row_count = sdf.count()
        logger.info(f"数据总行数: {row_count:,}")

        # --- 步骤 2: 转换为 Pandas DataFrame ---
        # 注意：如果数据量非常大，直接 toPandas() 可能导致 Driver OOM
        # 对于大数据集，需要考虑分布式训练 (Spark MLlib) 或采样后转 Pandas
        logger.info("--- 步骤 2: 将 Spark DataFrame 转换为 Pandas DataFrame ---")
        if row_count > 10_000_000: # 设置一个阈值，超过则警告
             logger.warning(f"数据行数 ({row_count:,}) 较多，转换为 Pandas 可能需要大量内存并耗时较长！")
        start_pandas_conv = time.time()
        pdf = sdf.toPandas()
        end_pandas_conv = time.time()
        logger.info(f"转换为 Pandas DataFrame 耗时: {end_pandas_conv - start_pandas_conv:.2f} 秒")
        logger.info(f"Pandas DataFrame 内存占用: {pdf.memory_usage(deep=True).sum() / (1024**3):.2f} GB") # 估算内存占用

        # --- 步骤 3: 数据预处理和分割 ---
        logger.info("--- 步骤 3: 数据预处理和时间分割 ---")
        # 选择特征列 (排除非特征列和目标列)
        # 假设 'y' 是目标列, 'unique_id', 'timestamp' 不是特征
        exclude_cols = ['y', 'unique_id', 'timestamp', 'location_id'] # 可能还有其他非数值列
        # 动态选择特征列，排除非数值类型和上面指定的列
        feature_cols = [c for c in pdf.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(pdf[c])]
        target_col = 'y'

        if not feature_cols:
             logger.error("未能自动识别任何数值类型的特征列！请检查数据和 exclude_cols 设置。")
             raise ValueError("No numeric feature columns found.")

        logger.info(f"选定的特征列 ({len(feature_cols)}): {feature_cols[:5]}...{feature_cols[-5:]}") # 显示部分特征名
        logger.info(f"目标列: {target_col}")

        # 按时间分割
        train_pdf, test_pdf = time_based_split(pdf, test_ratio=0.2)

        X_train = train_pdf[feature_cols]
        y_train = train_pdf[target_col]
        X_test = test_pdf[feature_cols]
        y_test = test_pdf[target_col]

        # --- 步骤 4: 训练和评估模型 ---
        logger.info("--- 步骤 4: 训练和评估模型 ---")

        # --- 4.1: 线性回归 (基线) ---
        logger.info("--- 开始训练 线性回归 模型 ---")
        start_lr = time.time()
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        end_lr = time.time()
        logger.info(f"线性回归训练耗时: {end_lr - start_lr:.2f} 秒")

        # 评估
        y_pred_lr = lr_model.predict(X_test)
        evaluate_model(y_test, y_pred_lr, "线性回归")

        # 保存模型
        lr_model_path = models_dir / "linear_regression_model.joblib"
        joblib.dump(lr_model, lr_model_path)
        logger.info(f"线性回归模型已保存到: {lr_model_path}")


        # --- 4.2: LightGBM ---
        logger.info("--- 开始训练 LightGBM 模型 ---")
        start_lgb = time.time()
        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1) # 使用所有可用核心
        # 可以添加 early_stopping_rounds 来防止过拟合，需要一个验证集
        # 这里我们先在整个训练集上训练，在测试集上评估
        lgb_model.fit(X_train, y_train)
        end_lgb = time.time()
        logger.info(f"LightGBM 训练耗时: {end_lgb - start_lgb:.2f} 秒")

        # 评估
        y_pred_lgb = lgb_model.predict(X_test)
        evaluate_model(y_test, y_pred_lgb, "LightGBM")

        # 保存模型
        lgb_model_path = models_dir / "lightgbm_model.joblib"
        joblib.dump(lgb_model, lgb_model_path)
        logger.info(f"LightGBM 模型已保存到: {lgb_model_path}")

        # --- (可选) 步骤 5: 特征重要性分析 (LightGBM) ---
        if hasattr(lgb_model, 'feature_importances_'):
            logger.info("--- LightGBM 特征重要性 ---")
            feature_importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"\nTop 10 Features:\n{feature_importance_df.head(10)}")
            # 可以将完整的重要性保存到文件
            importance_path = plots_dir / "lgbm_feature_importance.csv"
            feature_importance_df.to_csv(importance_path, index=False)
            logger.info(f"特征重要性已保存到: {importance_path}")


        logger.info("=====================================================")
        logger.info("===         模型训练脚本 执行完毕         ===")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"模型训练过程中发生严重错误: {e}")
        logger.exception("Traceback:")
    finally:
        if spark:
            try:
                stop_spark_session(spark)
            except Exception as stop_e:
                logger.error(f"停止 SparkSession 时发生错误: {stop_e}")
        else:
            logger.info("SparkSession 未成功初始化或已停止。")

        end_run_time = time.time()
        logger.info(
            f"--- 模型训练脚本总执行时间: {end_run_time - start_run_time:.2f} 秒 ---")

if __name__ == "__main__":
    try:
        run_model_training()
    except Exception as e:
        # 主函数已有详细日志记录
        sys.exit(1)
