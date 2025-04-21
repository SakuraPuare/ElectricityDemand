# Electricity Demand EDA Project

## 任务列表

- [x] 初始化项目结构
- [x] 设置日志 (`log_utils.py`)
- [x] 实现数据加载 (`load_data.py` - Spark 版)
- [x] 实现 Demand 分析 (`analyze_demand.py` - Spark 版)
    - [x] `analyze_demand_y_distribution`
    - [x] `analyze_demand_timeseries_sample`
- [x] 实现 Metadata 分析 (`analyze_metadata.py` - Pandas 版，由 Spark DF 转换)
    - [x] `analyze_metadata_categorical` (已更新包含 freq, dataset, timezone)
    - [x] `plot_metadata_categorical` (已更新包含 freq, dataset, timezone)
    - [x] `analyze_metadata_numerical` (已更新包含 cluster_size)
    - [x] `analyze_missing_locations`
- [x] 实现 Weather 分析 (`analyze_weather.py` - Spark 版)
    - [x] `analyze_weather_numerical`
    - [x] `analyze_weather_categorical` (已更新包含 weather_code)
    - [x] `analyze_weather_timeseries_sample`
    - [x] `analyze_weather_correlation` (新增天气特征相关性分析)
- [x] 实现关系分析 (`analyze_relationships.py` - 混合 Spark/Pandas)
    - [x] `analyze_demand_vs_metadata`
    - [x] `analyze_demand_vs_location`
    - [x] `analyze_demand_vs_weather`
- [x] 实现时间特征分析 (`analyze_time.py` - Spark 版)
    - [x] `analyze_datetime_features_spark`
    - [x] `analyze_timestamp_consistency`
- [x] 实现数据质量检查函数 (`data_quality.py` - Spark 版)
    - [x] `check_missing_values_spark`
    - [x] `check_duplicates_spark`
- [ ] **当前任务**: 运行完整的 EDA 脚本 (`1_run_eda.py`) 并检查新增分析的结果。
- [ ] (可选) 在 `Demand` 时间序列分析中加入 ACF/PACF 图绘制 (基于抽样)。
- [ ] (可选) 在 `Demand` 时间序列分析中加入更精细的季节性可视化。
- [ ] (可选) 探索 `Demand` 与 `weather_code` 的关系。
- [ ] 根据 EDA 结果进行数据预处理。
- [ ] 特征工程。
- [ ] 模型训练与评估。

## 下一步

1.  运行 `python src/electricitydemand/1_run_eda.py` 脚本。
2.  检查 `logs/` 目录下的最新日志文件，确认新增分析 (Metadata 的 `freq`, `dataset`, `timezone`, `cluster_size`；Weather 的 `weather_code` 分布和相关性矩阵) 已成功执行并记录了结果。
3.  检查 `plots/` 目录下是否生成了对应的图表 (`metadata_dist_freq.png`, `metadata_hist_cluster_size.png`, `weather_dist_weather_code.png`, `weather_correlation_matrix.png` 等)。
4.  审阅日志和图表，理解这些新特征的分布和关系。

**旧 TODO (已整合或完成):**

*   ~~在 `Metadata` 分析中加入对 `freq`, `cluster_size`, `dataset`, `timezone` 的分布分析和可视化。~~
*   ~~在 `Weather` 分析中加入对 `weather_code` 的分布分析和可视化。~~
*   ~~在 `Weather` 分析中加入天气特征之间的相关性分析 (绘制相关性热力图)。~~
*   ~~更新 `1_run_eda.py` 脚本，调用新增的分析函数。~~
*   ~~检查 `1_run_eda.py` 中是否还有遗漏的分析步骤。~~
*   ~~初始化 Spark EDA 项目结构 (`1_run_eda.py`, `eda/` 模块)。~~
*   ~~实现数据加载 (`load_data.py`)。~~
*   ~~实现数据质量检查 (`data_quality.py`) (Missing Values, Duplicates)。~~
*   ~~实现 Demand 单变量分析 (`analyze_demand.py`) (y 分布, 时间序列抽样)。~~
*   ~~实现 Metadata 单变量分析 (`analyze_metadata.py`) (Categorical, Numerical, Missing Locations)。~~
*   ~~实现 Weather 单变量分析 (`analyze_weather.py`) (Numerical, Categorical, 时间序列抽样)。~~
*   ~~实现关系分析 (`analyze_relationships.py`) (Demand vs Metadata, Demand vs Weather)。~~
*   ~~实现时间特征分析 (`analyze_time.py`) (Timestamp Consistency, Datetime Feature Extraction)。~~

## 当前任务

*   **修复 EDA 脚本中的缺失值检查错误**：
    *   修改 `src/electricitydemand/eda/data_quality.py` 中的 `check_missing_values_spark` 函数，使其能正确处理非数值类型的列（如 Timestamp 和 String）。✅

## 下一步任务

*   **重新运行完整的 EDA 流程**:
    *   执行 `src/electricitydemand/1_run_eda.py` 脚本。
    *   检查日志 (`logs/`) 确认所有 EDA 步骤（数据加载、质量检查、分布分析、关系分析、时间频率分析）都已成功完成，并且没有新的错误。
    *   检查生成的图表 (`charts/`) 是否符合预期。

## 已完成任务

*   **初始化项目结构**: 设置基本的文件夹（`src`, `data`, `logs`, `charts`, `notebooks`）。
*   **配置环境**: 安装必要的库（pyspark, pandas, matplotlib, seaborn, loguru）。
*   **设置日志**: 配置 `loguru` 通过 `src/electricitydemand/utils/log_utils.py`。
*   **下载数据集**: 从 Hugging Face 下载 `demand.parquet`, `metadata.parquet`, `weather.parquet` 到 `data/` 目录。
*   **编写初步的 EDA 脚本 (`1_run_eda.py`)**:
    *   实现 SparkSession 初始化。
    *   实现数据加载函数。
    *   实现数据质量检查（缺失值、重复值）函数 (在 `src/electricitydemand/eda/data_quality.py` 中)。
    *   实现数据分布分析函数 (在 `src/electricitydemand/eda/distribution_analysis.py` 中)。
    *   实现关系分析函数 (在 `src/electricitydemand/eda/relationship_analysis.py` 中)。
    *   实现时间频率分析函数 (在 `src/electricitydemand/eda/time_frequency_analysis.py` 中)。
*   **运行初步的 EDA**: 首次运行 `1_run_eda.py` 并根据日志进行调试。

# 数据分析任务清单

## 1. EDA (Exploratory Data Analysis) - 初步探索性数据分析 (已完成)

-   [x] **环境设置**:
    -   [x] 配置 Python 环境和 Spark。
    -   [x] 设置日志记录 (`loguru`)。
-   [x] **数据加载**:
    -   [x] 使用 Spark 加载 `demand.parquet`, `metadata.parquet`, `weather.parquet`。
-   [x] **数据概览与质量检查**:
    -   [x] 查看各数据集的 Schema、行数。
    *   [x] 检查缺失值比例。
    *   [x] 检查重复值。
    *   [x] 查看时间范围。
-   [x] **单变量分析**:
    *   [x] **Demand**: 分析 `y` (需求量) 的分布（描述性统计、绘制直方图/箱线图 - 使用抽样数据）。检查非正值。
    *   [x] **Metadata**: 分析关键分类特征的分布 (`building_class`, `location`, `freq`, `timezone`, `dataset`)。
    *   [x] **Weather**: 分析关键数值特征的分布 (`temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m` 等)。检查负值（如降水）。
-   [x] **双变量/多变量分析 (基于抽样)**:
    *   [x] **Demand vs. Metadata**: 不同 `building_class` 的 `y` 分布差异 (箱线图)。
    *   [x] **Demand vs. Weather**: 计算 `y` 与关键天气特征 (`temperature_2m`, `apparent_temperature`, `relative_humidity_2m`) 的相关性，并可视化 (散点图)。(需要先按 `unique_id` 抽样，然后合并 `metadata` 获取 `location_id`，再合并 `weather`)。
-   [x] **时间戳频率分析**:
    *   [x] 分析 `demand` 数据中 `timestamp` 的常见间隔 (基于抽样)。
    *   [x] 分析 `weather` 数据中 `timestamp` 的常见间隔 (基于抽样)。
    *   [x] 对比两者频率，识别潜在的对齐问题。
-   [x] **Parquet 转换**:
    -   [x] 将 `demand.parquet` 中的 `timestamp` 从 `string` 转换为 `timestamp` 类型，保存为 `demand_converted.parquet`。(此步骤已完成)

## 2. Preprocessing - 数据预处理 (进行中)

-   [x] **Demand 数据重采样**:
    -   [x] 将 `demand_converted.parquet` 中的数据按 `unique_id` 分组，重采样到 **小时** 频率 (例如，将 15/30 分钟数据聚合为小时数据)。处理缺失的小时。
    -   [x] 使用 Spark 完成此操作。
    -   [x] 保存为 `demand_hourly.parquet`。
-   [ ] **合并数据**:
    -   [ ] 加载 `demand_hourly.parquet`, `metadata.parquet`, `weather.parquet`。
    -   [ ] 将 `demand_hourly` 与 `metadata` 按 `unique_id` 合并。
    -   [ ] 将上一步结果与 `weather` 按 `location_id` 和 `timestamp` 合并。
    -   [ ] 保存合并后的数据集 (例如 `merged_data.parquet`)。
-   [ ] **特征工程 (初步)**:
    -   [ ] 从 `timestamp` 提取时间特征 (年、月、日、星期几、小时等)。
    -   [ ] 处理缺失值 (合并后可能产生新的缺失值，选择合适的填充策略)。
-   [ ] **数据拆分**:
    -   [ ] 按时间拆分训练集、验证集、测试集。

## 3. Feature Engineering - 特征工程 (待办)

-   [ ] 创建滞后特征 (Lag features) for `y`。
-   [ ] 创建滚动窗口统计特征 (Rolling window statistics) for `y` 和天气特征。
-   [ ] 对分类特征进行编码 (如 `building_class`, `location` 等)。
-   [ ] 其他可能的特征 (例如节假日信息、交互特征等)。

## 4. Modeling - 模型训练 (待办)

-   [ ] 选择基线模型 (e.g., SARIMA, Prophet)。
-   [ ] 尝试机器学习模型 (e.g., LightGBM, XGBoost)。
-   [ ] (可选) 尝试深度学习模型 (e.g., LSTM, TCN)。
-   [ ] 模型训练与调优。

## 5. Evaluation - 模型评估 (待办)

-   [ ] 在测试集上评估模型性能 (e.g., MAE, RMSE, MAPE)。
-   [ ] 分析预测误差。
-   [ ] 可视化预测结果。

## 6. Deployment/Reporting - 部署/报告 (待办)

-   [ ] 总结分析结果和模型性能。
-   [ ] (可选) 将模型部署为服务。