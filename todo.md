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