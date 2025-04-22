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
    -   [ ] **周期性特征编码** (待办 - 可选，如 sine/cosine 变换)
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

-   [x] 评审并确认 LaTeX 报告 (`reports/main.tex`) 的结构。
-   **下一步：** [进行中] 细化和审查 LaTeX 报告 (`reports/main.tex`) 的内容，确保文字描述与分析结果和图表一致。

## 已完成任务

-   ✅ **项目初始化与环境设置**
    -   [x] 初始化项目结构 (`cookiecutter`)
    -   [x] 设置 Python 版本 (`.python-version`)
    -   [x] 初始化虚拟环境 (`.venv`)
    -   [x] 安装依赖 (`requirements.lock`, `requirements-dev.lock`)
    -   [x] 配置 IDE (`.idea`, `.vscode`, `.cursor`)
    -   [x] 初始化 Git 仓库 (`.gitignore`, `LICENSE`, `README.md`)
    -   [x] 添加 `todo.md`
-   ✅ **数据加载与初步探索 (EDA)**
    -   [x] 实现数据加载功能 (`1_run_eda.py`, `load_data`)
    -   [x] 实现 EDA 主函数 (`1_run_eda.py`, `run_eda`)
    -   [x] 添加日志配置 (`log_utils.py`)
    -   [x] 执行初步 EDA (`1_run_eda.py`)
        -   [x] 计算基本信息 (count, schema)
        -   [x] 检查缺失值
        -   [x] 检查重复值
        -   [x] 计算时间范围
        -   [x] **Demand 数据分析**:
            -   [x] 描述性统计 (抽样)
            -   [x] 非正值检查
            -   [x] 分布可视化 (直方图/KDE - 原始 & 对数)
            -   [x] 时间序列样本可视化 (随机抽取几个 `unique_id`)
        -   [x] **Metadata 数据分析**:
            -   [x] 分类特征分布 (building\_class, location, freq, timezone, dataset)
            -   [x] 数值特征分布 (latitude, longitude, cluster\_size) - 统计与可视化
            -   [x] 地理位置可视化 (散点图)
        -   [x] **Weather 数据分析**:
            -   [x] 描述性统计 (完整数据)
            -   [x] 关键数值特征分布可视化 (temperature, humidity, precipitation, wind\_speed 等)
            -   [x] 负值检查 (precipitation, rain, snowfall)
            -   [x] 分类特征分布 (weather\_code, is\_day)
        -   [x] **关系分析 (抽样)**:
            -   [x] Demand vs Metadata (e.g., building\_class) - 箱线图 (原始 & 对数)
            -   [x] Demand vs Weather (抽样合并) - 相关性计算 & 散点图
            -   [x] Weather 特征间相关性 - 相关矩阵热力图
        -   [x] **时间特征分析**:
            -   [x] 需求/天气时间戳频率分析 (基于抽样)
            -   [x] **周期性分析 (基于 Spark 全量聚合)**
                -   [x] 按小时聚合平均需求
                -   [x] 按星期几聚合平均需求
                -   [x] 按月份聚合平均需求
-   ✅ **数据预处理与合并 (Spark)**
    -   [x] 实现数据加载 (需求、元数据、天气) (`2_run_preprocessing.py`)
    -   [x] 实现需求数据频率转换 (假设已完成/或在此步骤中加入) -> 统一到小时
    -   [x] 实现需求与元数据合并 (`unique_id`)
    -   [x] 实现天气数据时间戳处理与去重
    -   [x] 实现合并后的需求 - 元数据与天气数据合并 (`location_id`, `timestamp@hour`)
    -   [x] 添加合并过程诊断 (检查 `location_id` 匹配情况，`null` 值比例)
    -   [x] 保存合并后的数据 (`merged_data.parquet`)
    -   [x] 编写 Spark 执行脚本 (`2_run_preprocessing.py`) 并记录执行日志
-   ✅ **特征工程 (Spark)**
    -   [x] 加载合并后的数据 (`merged_data.parquet`) (`3_run_feature_engineering.py`)
    -   [x] 实现时间特征提取 (`add_time_features_spark`)
    -   [x] 实现滚动窗口特征提取 (`add_rolling_features_spark`)
    -   [x] 实现缺失值处理 (`handle_missing_values_spark`)
    -   [x] 优化 Spark 性能 (重分区，持久化)
    -   [x] 按年/月分区保存特征数据 (`features.parquet`)
    -   [x] 编写 Spark 执行脚本 (`3_run_feature_engineering.py`) 并记录执行日志
-   ✅ **报告撰写 (LaTeX)**
    -   [x] 创建 LaTeX 文档结构 (`reports/main.tex`)
    -   [x] 插入分析过程中生成的图表
    -   [x] 添加图表标题和引用标签
    -   [x] 撰写报告摘要、引言、各分析章节初稿
    -   [x] 评审并确认报告结构

## 待办任务

-   [ ] 模型训练与评估 (例如使用 Spark MLlib)
    -   [ ] 数据准备 (划分训练/验证/测试集)
    -   [ ] 特征处理 (编码、标准化等)
    -   [ ] 选择和训练模型 (e.g., Linear Regression, GBTRegressor)
    -   [ ] 评估模型性能
    -   [ ] 超参数调优
-   [ ] 完善报告 (模型部分)
-   [ ] 代码清理与文档完善