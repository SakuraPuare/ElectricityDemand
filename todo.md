# Electricity Demand 大数据分析

https://huggingface.co/datasets/EDS-lab/electricity-demand

## 最终目标与成果

1.  **构建准确的电力需求预测模型**: 利用历史数据和天气信息，预测未来电力消耗。
2.  **提供可解释的分析**: 理解影响电力需求的关键因素（时间、天气、建筑类型等）。
3.  **生成详细分析报告**: 总结数据探索发现、模型构建过程、性能评估和关键结论。

---

# Electricity Demand 大数据分析任务列表

## 1. 数据获取 (Data Acquisition)

- [ ] 从 Hugging Face Hub 下载 `electricity-demand` 数据集 (`demand.parquet`, `metadata.parquet`/`.csv`, `weather.parquet`)。
- [ ] 确认文件完整性。

## 2. 数据加载与探索性数据分析 (Data Loading & EDA)

- [ ] 选择大数据处理框架 (e.g., Spark, Dask, Pandas)。
- [ ] 加载 Parquet 文件。
- [ ] 查看数据基本信息 (shape, dtypes)。
- [ ] 检查缺失值 (missing values)。
- [ ] 检查重复值 (duplicate values)。
- [ ] 分析 `demand.parquet`:
  - [ ] `y` (用电量) 分布。
  - [ ] 不同 `unique_id` 的用电模式。
  - [ ] `timestamp` 范围和频率。
- [ ] 分析 `metadata.parquet`:
  - [ ] `building_class`, `location`, `freq` 等分布。
- [ ] 分析 `weather.parquet`:
  - [ ] 天气指标分布和范围。
  - [ ] 时间范围和频率与需求数据匹配性。
- [ ] 可视化关键特征和关系。

## 3. 数据预处理与特征工程 (Data Preprocessing & Feature Engineering)

- [ ] 合并 `demand`, `metadata`, `weather` 数据。
  - [ ] 处理 `unique_id`, `location_id` 连接。
  - [ ] 处理 `timestamp` 对齐和时区。
- [ ] 处理缺失值 (imputation)。
- [ ] 处理异常值 (outlier detection/treatment)。
- [ ] 创建时间特征 (year, month, day, hour, weekday, holiday etc.)。
- [ ] 创建滞后特征 (lag features for `y`)。
- [ ] 创建滑动窗口特征 (rolling window statistics for `y` and weather features)。
- [ ] 选择/工程化天气特征 (normalization/standardization, interaction features)。
- [ ] 编码分类特征 (`building_class` etc.)。

## 4. 模型选择与训练 (Model Selection & Training)

- [ ] 按时间划分训练集、验证集、测试集。
- [ ] 选择时间序列预测模型 (Statistical, ML, DL)。
- [ ] 在训练集上训练模型。
- [ ] 使用验证集进行超参数调优。

## 5. 模型评估 (Model Evaluation)

- [ ] 选择评估指标 (MAE, RMSE, MAPE, SMAPE etc.)。
- [ ] 在测试集上评估最终模型。
- [ ] 进行误差分析。

## 6. 结果解释与报告 (Result Interpretation & Reporting)

- [ ] 可视化预测结果 vs 实际值。
- [ ] 分析特征重要性 (if applicable)。
- [ ] 撰写分析报告。
