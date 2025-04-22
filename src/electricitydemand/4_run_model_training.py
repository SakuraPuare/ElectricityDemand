import sys
import time
from pathlib import Path

from loguru import logger
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType  # 用于识别数值列

# --- 项目设置 ---
try:
    from electricitydemand.utils.log_utils import setup_logger
    from electricitydemand.utils.project_utils import (
        create_spark_session,
        get_project_root,
        setup_project_paths,
        stop_spark_session,
    )
except ImportError as e:
    print(f"Error importing project utils: {e}", file=sys.stderr)
    sys.exit(1)

project_root = get_project_root()
src_path, data_dir, logs_dir, plots_dir = setup_project_paths(project_root)
models_dir = project_root / "models"  # 创建模型保存目录
models_dir.mkdir(parents=True, exist_ok=True)

# --- 配置日志 ---
log_prefix = Path(__file__).stem  # 使用 4_run_model_training 作为前缀
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
sample_fraction_str = "0p01"  # 假设我们用了 1% 抽样，需要根据实际情况调整
features_sampled_path = data_dir / f"features_sampled_{sample_fraction_str}.parquet"
features_full_path = data_dir / "features.parquet"

if features_sampled_path.exists():
    features_input_path = features_sampled_path
    logger.info(f"使用抽样特征数据：{features_input_path}")
else:
    features_input_path = features_full_path
    logger.info(f"使用全量特征数据：{features_input_path}")


# ======================================================================
# ==                      Helper Functions                          ==
# ======================================================================

# 不再需要 Pandas 的时间分割函数
# def time_based_split(df: pd.DataFrame, test_ratio: float = 0.2): ...

# 使用 MLlib 的评估器，不再需要这个函数
# def evaluate_model(y_true, y_pred, model_name="Model"): ...


def spark_time_based_split(sdf: SparkSession.DataFrame, timestamp_col: str = "timestamp", test_ratio: float = 0.2):
    """按时间戳分割 Spark DataFrame"""
    logger.info(f"开始基于时间戳 '{timestamp_col}' 分割 Spark DataFrame，测试集比例：{test_ratio:.1%}")
    if timestamp_col not in sdf.columns:
        raise ValueError(f"DataFrame 中缺少 '{timestamp_col}' 列，无法进行时间分割。")

    # 确保时间戳列是正确类型 (或者 Spark 能理解的排序类型)
    # 如果已经是 TimestampType 则无需转换

    # 找到分割点时间戳
    # 注意：这里假设数据量可以接受计算分位数，如果极大，可能需要抽样估算
    try:
        split_timestamp = sdf.approxQuantile(timestamp_col, [1.0 - test_ratio], 0.01)[0]  # 0.01 是相对误差
        logger.info(f"计算得到的时间分割点：{split_timestamp}")
    except Exception as e:
        logger.error(f"计算时间分割点时出错：{e}. 尝试获取最大最小值进行估算。")
        # 备选方案：按时间范围比例分割 (可能不太精确)
        min_max_time = sdf.agg(F.min(timestamp_col).alias("min_t"), F.max(timestamp_col).alias("max_t")).first()
        if not min_max_time or min_max_time["min_t"] is None or min_max_time["max_t"] is None:
            raise ValueError("无法获取时间戳范围用于分割。")
        total_duration = (min_max_time["max_t"] - min_max_time["min_t"]).total_seconds()
        split_timestamp = min_max_time["min_t"] + F.expr(f"INTERVAL {int(total_duration * (1 - test_ratio))} SECONDS")
        logger.info(f"备选方案 - 估算的时间分割点：{split_timestamp}")

    train_sdf = sdf.filter(F.col(timestamp_col) < split_timestamp)
    test_sdf = sdf.filter(F.col(timestamp_col) >= split_timestamp)

    # 记录分割后的大小 (可能需要触发计算)
    # train_count = train_sdf.count()
    # test_count = test_sdf.count()
    # logger.info(f"分割后 - 训练集行数：{train_count:,}")
    # logger.info(f"分割后 - 测试集行数：{test_count:,}")
    # 为避免不必要的 count()，暂时注释掉，可以在需要时取消注释

    logger.info("Spark DataFrame 时间分割完成。")
    return train_sdf, test_sdf


# ======================================================================
# ==                   Main Execution Function                      ==
# ======================================================================

def run_model_training_spark():
    """使用 Spark MLlib 加载特征数据，训练模型，评估并保存模型"""
    logger.info("=====================================================")
    logger.info("=== 开始执行 Spark MLlib 模型训练脚本 ===")
    logger.info("=====================================================")
    start_run_time = time.time()
    spark = None

    try:
        # --- 创建 SparkSession ---
        logger.info("创建 SparkSession...")
        # 可能需要根据集群调整配置
        spark = create_spark_session(app_name="ElectricityDemandSparkMLTraining")
        logger.info("SparkSession 创建成功。")

        # --- 步骤 1: 加载特征数据 ---
        logger.info(f"--- 步骤 1: 加载特征数据 {features_input_path} ---")
        sdf = spark.read.parquet(str(features_input_path))
        logger.info("特征数据加载成功。")
        logger.info("Schema:")
        sdf.printSchema()
        # row_count = sdf.count() # 避免立即计算全量 count
        # logger.info(f"数据总行数：{row_count:,}") # 可以在需要时取消注释

        # --- 步骤 2: 特征准备 ---
        logger.info("--- 步骤 2: 特征准备 (VectorAssembler) ---")
        target_col = 'y'
        # 排除非特征列和目标列
        exclude_cols = [target_col, 'unique_id', 'timestamp', 'location_id']
        # 动态识别数值类型的特征列
        numeric_feature_cols = [f.name for f in sdf.schema.fields if
                                isinstance(f.dataType, NumericType) and f.name not in exclude_cols]

        if not numeric_feature_cols:
            logger.error("未能自动识别任何数值类型的特征列！请检查数据和 exclude_cols 设置。")
            raise ValueError("No numeric feature columns found.")

        logger.info(
            f"识别出的数值特征列 ({len(numeric_feature_cols)}): {numeric_feature_cols[:5]}...{numeric_feature_cols[-5:]}")
        logger.info(f"目标列：{target_col}")

        assembler = VectorAssembler(
            inputCols=numeric_feature_cols,
            outputCol="features",
            handleInvalid="skip"  # 或 'keep', 'error' - 处理特征中的无效值 (NaN, Null)
        )

        # 应用 VectorAssembler (这是一个转换操作，延迟执行)
        sdf_assembled = assembler.transform(sdf)
        logger.info("特征已合并到 'features' 列。")
        # 选择后续需要的列，减少数据量
        sdf_final = sdf_assembled.select('timestamp', 'features',
                                         F.col(target_col).alias('label'))  # MLlib 通常需要目标列名为 'label'
        logger.info("已选择 'timestamp', 'features', 'label' 列。")

        # --- 步骤 3: 时间分割 ---
        logger.info("--- 步骤 3: 基于时间戳分割数据 ---")
        train_sdf, test_sdf = spark_time_based_split(sdf_final, timestamp_col='timestamp', test_ratio=0.2)

        # 可选：缓存训练集和测试集以加速后续模型训练（如果内存允许）
        # train_sdf.persist(StorageLevel.MEMORY_AND_DISK)
        # test_sdf.persist(StorageLevel.MEMORY_AND_DISK)
        # logger.info("训练集和测试集已缓存。")

        # --- 步骤 4: 训练和评估模型 ---
        logger.info("--- 步骤 4: 训练和评估 Spark MLlib 模型 ---")

        # 定义评估器
        evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")

        # --- 4.1: 线性回归 (MLlib) ---
        logger.info("--- 开始训练 MLlib 线性回归 模型 ---")
        start_lr = time.time()
        lr = LinearRegression(featuresCol='features', labelCol='label')
        lr_model = lr.fit(train_sdf)
        end_lr = time.time()
        logger.info(f"MLlib 线性回归训练耗时：{end_lr - start_lr:.2f} 秒")

        # 评估
        logger.info("评估 MLlib 线性回归模型...")
        lr_predictions = lr_model.transform(test_sdf)
        lr_rmse = evaluator_rmse.evaluate(lr_predictions)
        lr_mae = evaluator_mae.evaluate(lr_predictions)
        logger.success("--- MLlib 线性回归 评估结果 ---")
        logger.success(f"  均方根误差 (RMSE): {lr_rmse:.4f}")
        logger.success(f"  平均绝对误差 (MAE): {lr_mae:.4f}")
        logger.success("---------------------------------")

        # 保存模型
        lr_model_path = models_dir / "mllib_linear_regression_model"
        lr_model.write().overwrite().save(str(lr_model_path))  # MLlib 模型保存为目录
        logger.info(f"MLlib 线性回归模型已保存到：{lr_model_path}")

        # --- 4.2: GBT 回归 (MLlib) ---
        logger.info("--- 开始训练 MLlib GBT 回归 模型 ---")
        start_gbt = time.time()
        gbt = GBTRegressor(featuresCol='features', labelCol='label', seed=42)  # 可添加 maxIter, maxDepth 等参数
        gbt_model = gbt.fit(train_sdf)
        end_gbt = time.time()
        logger.info(f"MLlib GBT 回归训练耗时：{end_gbt - start_gbt:.2f} 秒")

        # 评估
        logger.info("评估 MLlib GBT 回归模型...")
        gbt_predictions = gbt_model.transform(test_sdf)
        gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
        gbt_mae = evaluator_mae.evaluate(gbt_predictions)
        logger.success("--- MLlib GBT 回归 评估结果 ---")
        logger.success(f"  均方根误差 (RMSE): {gbt_rmse:.4f}")
        logger.success(f"  平均绝对误差 (MAE): {gbt_mae:.4f}")
        logger.success("-----------------------------")

        # 保存模型
        gbt_model_path = models_dir / "mllib_gbt_regression_model"
        gbt_model.write().overwrite().save(str(gbt_model_path))
        logger.info(f"MLlib GBT 回归模型已保存到：{gbt_model_path}")

        # --- (可选) 步骤 5: 特征重要性分析 (GBT) ---
        if hasattr(gbt_model, 'featureImportances'):
            logger.info("--- MLlib GBT 特征重要性 ---")
            # 注意：MLlib 的 featureImportances 是一个 SparseVector 或 DenseVector
            # 需要与 numeric_feature_cols 对应起来
            importances = gbt_model.featureImportances.toArray()  # 转为 numpy array
            if len(importances) == len(numeric_feature_cols):
                feature_importance_list = sorted(zip(numeric_feature_cols, importances), key=lambda x: x[1],
                                                 reverse=True)
                logger.info("Top 10 Features:")
                for feature, importance in feature_importance_list[:10]:
                    logger.info(f"  {feature}: {importance:.4f}")
                # 可以将完整的重要性保存到文件
                # import pandas as pd # 临时导入 pandas 用于保存 csv
                # importance_df = pd.DataFrame(feature_importance_list, columns=['feature', 'importance'])
                # importance_path = plots_dir / "mllib_gbt_feature_importance.csv"
                # importance_df.to_csv(importance_path, index=False)
                # logger.info(f"特征重要性已保存到：{importance_path}")
            else:
                logger.warning("特征重要性向量长度与特征列数量不匹配，无法显示名称。")
                logger.warning(f"重要性向量：{importances}")

        # 可选：解除缓存
        # train_sdf.unpersist()
        # test_sdf.unpersist()
        # logger.info("训练集和测试集已解除缓存。")

        logger.info("=====================================================")
        logger.info("===      Spark MLlib 模型训练脚本 执行完毕      ===")
        logger.info("=====================================================")

    except Exception as e:
        logger.critical(f"Spark MLlib 模型训练过程中发生严重错误：{e}")
        logger.exception("Traceback:")
    finally:
        if spark:
            try:
                stop_spark_session(spark)
            except Exception as stop_e:
                logger.error(f"停止 SparkSession 时发生错误：{stop_e}")
        else:
            logger.info("SparkSession 未成功初始化或已停止。")

        end_run_time = time.time()
        logger.info(
            f"--- Spark MLlib 模型训练脚本总执行时间：{end_run_time - start_run_time:.2f} 秒 ---")


if __name__ == "__main__":
    run_model_training_spark()
