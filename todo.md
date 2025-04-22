# Electricity Demand 数据分析项目 TODO

## 数据探索与预处理 (EDA & Preprocessing)

-   **环境设置与库安装** ✅ (已完成)
-   **数据加载与初步概览 (Spark)** ✅ (已完成)
    -   加载 `demand.parquet`, `metadata.parquet`, `weather.parquet` ✅
    -   显示 Schema 和基本信息 ✅
    -   计算并记录各数据集大小 ✅
    -   处理 `timestamp` 列 (转换为 TimestampType) ✅ (`1_run_eda.py` - `convert_timestamp_columns`)
-   **数据质量检查 (Spark)** ✅ (已完成)
    -   检查缺失值 ✅ (`1_run_eda.py` - `check_missing_values_spark`)
    -   检查重复值 ✅ (`1_run_eda.py` - `check_duplicates_spark`)
-   **描述性统计分析 (Spark/抽样)** ✅ (已完成)
    -   Demand ('y') 分布分析 (均值、中位数、分位数、绘制直方图/箱线图 - 抽样) ✅ (`1_run_eda.py` - `analyze_demand_distribution_spark_sampled`)
    -   Metadata 特征分布 (Building Class, Location, Freq 等 - 计数、绘制条形图) ✅ (`1_run_eda.py` - `analyze_metadata_distribution_spark`)
    -   Weather 数值特征分布 (Temperature, Humidity, Precipitation 等 - 描述性统计、绘制直方图/箱线图 - 抽样或全量) ✅ (`1_run_eda.py` - `analyze_weather_distribution_spark`)
    -   检查天气数据中的负值 ✅ (`1_run_eda.py` - `check_negative_weather_values_spark`)
-   **关系分析 (Spark/抽样)** ✅ (已完成)
    -   Demand vs. Metadata (e.g., 按 Building Class 分组分析 'y') ✅ (`1_run_eda.py` - `analyze_demand_vs_metadata_spark_sampled`)
    -   Demand vs. Weather (计算相关性，绘制散点图 - 需要合并数据，抽样) ✅ (`1_run_eda.py` - `analyze_demand_vs_weather_spark_sampled`)
-   **时间戳频率分析 (Spark/抽样)** ✅ (已完成)
    -   分析 Demand 数据时间间隔分布 ✅ (`1_run_eda.py` - `analyze_timestamp_frequency_spark`)
    -   分析 Weather 数据时间间隔分布 ✅ (`1_run_eda.py` - `analyze_timestamp_frequency_spark`)
    -   比较两者频率差异 ✅ (已在 EDA 总结中说明)
-   **数据预处理** ✅ (已完成)
    -   将 Demand 数据重采样至小时频率 ✅ (`2_run_preprocessing.py` - `run_demand_resampling_spark`)
    -   验证重采样结果 ✅ (`2_run_preprocessing.py` - `validate_resampling_spark` in `run_demand_resampling_spark`)
    -   **修正数据合并逻辑 (高优先级)**:
        -   [x] **调查天气数据缺失原因**: 已确认是 Demand 时间戳未对齐到小时导致 Join 失败。(完成于 `analyze_weather_completeness.py`)
        -   [x] **修改 `2_run_preprocessing.py`**: 在与 Weather 数据 Join 之前，将 Demand 数据的时间戳**向下取整到小时** (使用 `date_trunc`)。(完成)
        -   [x] **重新运行 `2_run_preprocessing.py`**: 生成修正后的 `merged_data.parquet`，并验证天气缺失率已大幅降低。(完成 - **成功**)
    -   合并 Demand, Metadata, Weather 数据 ✅ (`2_run_preprocessing.py` - `run_merge_data_spark`)
    -   保存合并后的数据 ✅ (`2_run_preprocessing.py` - `run_merge_data_spark`)

## 特征工程 (Feature Engineering) ⏳ (进行中)

-   [x] **修复内存溢出 (OOM) 问题**:
    -   [x] 尝试注释掉 `handle_missing_values_spark` 中间步骤的 `.count()` 操作。
    -   [x] **优化 Spark 配置**: 修改 `create_spark_session` 以更充分利用 CPU/内存，并允许环境变量覆盖。
-   [ ] **重新运行 `3_run_feature_engineering.py` (全量数据)** (待办 - **下一步执行**)
    -   [ ] **监控 Spark UI 和日志**，确认 OOM 是否解决，检查资源利用率和执行情况。
-   [ ] **特征工程 (续)**:
    -   [ ] **分类特征编码**: 对 `building_class` 等进行编码。 (待办)
    -   [ ] **滞后特征 (Lag Features)** (待办 - 可选)
    -   [ ] **周期性特征编码** (待办 - 可选, 如 sine/cosine 变换)
    -   [ ] **交互特征** (待办 - 可选)
    -   [ ] **产出**: 最终的特征工程后的 Parquet 文件 (`data/features.parquet`)。
-   [ ] **(可选) 特征选择**

## 模型训练与评估 (Modeling & Evaluation)

-   选择模型 (e.g., ARIMA, Prophet, LightGBM, LSTM)
-   划分训练集/验证集/测试集
-   训练模型
-   评估模型性能 (e.g., MAE, RMSE, MAPE)
-   模型调优

## 结果可视化与报告 (Visualization & Reporting)

-   可视化预测结果与实际值对比
-   生成分析报告总结发现

---
**当前状态**: 已优化 Spark Session 创建逻辑以更好地利用资源并尝试解决 OOM 问题。
**下一步**:
1.  **重新运行 `3_run_feature_engineering.py` (全量数据)** 并密切监控 Spark UI 和日志。
2.  如果成功，继续特征工程：分类特征编码。
3.  如果仍然失败，分析 Spark UI 和日志（特别是 OOM 的 Heap Dump 文件，如果生成了的话），可能需要进一步调整配置或优化代码逻辑。

## 当前任务

*   **模型训练 (进行中):**
    -   **问题:** 当前使用 `toPandas()` 将大数据集加载到 Driver 内存，导致潜在的内存溢出。
    -   **方案:** 重构 `4_run_model_training.py`，使用 Spark MLlib 进行分布式模型训练，以处理大规模数据集。
    -   **待确认:** 用户确认是否采用 Spark MLlib 方案。

## 下一步任务 (待定)

*   **Spark MLlib 重构:**
    *   移除 `toPandas()`。
    *   实现 `VectorAssembler` 特征准备。
    *   实现基于 Spark DataFrame 的时间分割。
    *   替换 Scikit-learn/LightGBM 模型为 Spark MLlib 模型 (`LinearRegression`, `GBTRegressor`)。
    *   替换评估逻辑为 Spark MLlib `RegressionEvaluator`。
    *   替换模型保存/加载逻辑为 Spark MLlib 格式。
*   **模型评估与分析:** 分析 MLlib 模型的性能和特征重要性 (如果 GBT 支持)。
*   **(可选) 超参数调优:** 使用 Spark MLlib 的 `CrossValidator` 或 `TrainValidationSplit` 进行超参数搜索。

## 已完成任务

*   ~~环境设置和库安装~~
*   ~~数据加载与初步探索 (0_setup_and_load.py)~~
*   ~~数据概览与质量检查 (1_explore_data_quality.py)~~
*   ~~数据分析与可视化 (2_analyze_data.py)~~
*   ~~特征工程 (3_feature_engineering.py)~~
*   ~~模型训练脚本基础框架 (4_run_model_training.py - 初始 Pandas 版本)~~

## 当前任务进度和下一步计划

- [x] 1. 下载并初步了解数据集 (`data/demand.parquet`, `data/metadata.parquet`, `data/weather.parquet`).
      - 已阅读数据集 README 和概览信息，了解数据结构、数据量、缺失值、重复值、时间范围等基本情况。
- [x] 2. 数据加载和基本探索 (使用 Spark)。
      - 已使用 Spark 加载原始数据。
      - 已进行基本的数据概览，例如查看 Schema, 数据量等。
      - 已对 `demand` 数据进行抽样描述性统计和分布分析（已完成，见概览信息）。
      - 已对 `metadata` 数据进行探索分析（已完成，见概览信息）。
      - 已对 `weather` 数据进行探索分析（已完成，见概览信息）。
      - 已对数据间的关系进行初步分析（已完成，见概览信息）。
      - 已分析时间戳频率匹配性（已完成，见概览信息）。
- [x] 3. 数据清洗和预处理。
      - 已处理 weather 数据中的重复行。
      - 已考虑 demand 数据中的缺失值 (`y` 列) 和非正值。
      - 已考虑 metadata 中的缺失值。
- [x] 4. 特征工程。
      - 已合并 `demand` 和 `metadata` 数据。
      - 已考虑合并天气数据并处理时间频率不匹配问题。
      - 已创建时间相关特征（如小时、星期、月份等）。
      - 已生成 `data/features.parquet` 特征数据集。
- [x] 5. Spark MLlib 模型训练。
    - **错误修复**: 在加载 `data/features.parquet` 时遇到 `[CANNOT_READ_FILE_FOOTER]` 错误，原因是目录中包含非 parquet 文件 `coscli.log`。**下一步：请手动或通过脚本移除 `data/features.parquet/coscli.log` 文件。**
    - 重新尝试加载特征数据并进行模型训练。
    - 选择合适的 Spark MLlib 模型（例如 Linear Regression, Gradient Boosted Trees 等）。
    - 划分训练集和测试集。
    - 训练模型。
    - 评估模型性能。
    - 保存训练好的模型。

## 未来任务 (根据进度更新)

- [ ] 6. 模型评估和调优。
- [ ] 7. 使用模型进行预测。
- [ ] 8. 结果可视化和报告。
- [ ] 9. 考虑更复杂的模型或方法。