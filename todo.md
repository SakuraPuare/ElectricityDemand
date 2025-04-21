# Electricity Demand EDA 任务列表

## **项目迁移至 PySpark**

-   [x] **环境准备**:
    -   [x] 安装 `pyspark`
    -   [x] 确保 Java 环境可用
-   [x] **数据加载与基础检查**:
    -   [x] 在脚本 (`load_data.py`, `1_run_eda.py` 等) 中创建 `SparkSession`。
    -   [x] **添加 Spark 配置**:
        -   [x] `.config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")`
        -   [x] `.config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")` (以及对应的 Write 配置)
        -   [x] `.config("spark.sql.parquet.int64AsTimestampNanos", "true")` <--- **解决 Parquet 时间戳读取问题 (SPARK-44988)**
    -   [x] 修改 `load_data.py` 中的 `load_datasets` 函数，使用 `spark.read.parquet`。
    -   [x] 修改 `load_data.py` 中的 `log_basic_info` 函数，使用 Spark API (`.count()`, `.columns`, `.show()`, `.printSchema()`)。
    -   [x] 修改 `load_data.py` 中的 `check_missing_values` 函数，使用 Spark API (`.agg()`, `F.sum(F.when(...))`)。
    -   [x] 修改 `load_data.py` 中的 `check_duplicates` 函数，使用 Spark API (`.groupBy().count().where(...)`)。
    -   [x] 修改 `load_data.py` 中的 `log_time_ranges` 函数，使用 Spark API (`.agg()`, `F.min()`, `F.max()`)。
    -   [x] 更新 `1_run_eda.py` 以使用 Spark 加载数据。
-   [ ] **EDA 分析函数迁移**:
    -   [ ] **Demand 分析 (`analyze_demand.py`)**:
        -   [ ] `analyze_demand_y_distribution`: 使用 Spark `.describe()` 或近似分位数计算。抽样 (`.sample()`) -> Pandas -> 绘图 (`plot_numerical_distribution`)。
        -   [ ] `plot_demand_y_distribution`: 调整以接收 Pandas Series (来自 Spark 抽样)。
        -   [ ] `analyze_demand_timeseries_sample`: 使用 Spark `.filter()` 获取样本 ID -> Pandas -> 绘图/频率分析。
    -   [ ] **Metadata 分析 (`analyze_metadata.py`)**:
        -   [ ] (已部分完成) 当前策略：如果内存允许，将 Metadata collect 到 Pandas (`.toPandas()`)，然后复用现有函数。检查内存占用。
        -   [ ] 备选方案 (如果内存不足)：重写函数，使用 Spark API (`.groupBy().count()`, `.describe()`) 进行分析，抽样 -> Pandas -> 绘图。
    -   [ ] **Weather 分析 (`analyze_weather.py`)**:
        -   [ ] `analyze_weather_numerical`: 使用 Spark `.describe()`，抽样 (`.sample()`) -> Pandas -> 绘图。
        -   [ ] `analyze_weather_categorical`: 使用 Spark `.groupBy().count()`, 抽样/排序 -> Pandas -> 绘图。
    -   [ ] **关系分析 (`analyze_relationships.py`)**:
        -   [ ] `merge_demand_metadata_sample`: 重写，使用 Spark Join 操作 (`ddf_demand.join(ddf_metadata, on='unique_id')`)，然后抽样 (`.sample()`)。
        -   [ ] `analyze_demand_vs_metadata`: 使用 Spark Join 获取合并数据，抽样 -> Pandas -> 绘图 (`sns.boxplot`)。
        -   [ ] `analyze_demand_vs_location`: 同上，增加 Top N 逻辑 (可用 Spark `Window` 函数或 `groupBy().count()` 实现)。
        -   [ ] `analyze_demand_vs_weather`: 重写，使用 Spark Join。`merge_asof` 的 Spark 实现较复杂，可能需要：
            *   将时间戳转换为数值或分桶。
            *   使用窗口函数 (`Window.partitionBy('location_id').orderBy('timestamp')`) 结合 `lag` 或 `last` 函数查找最近的天气记录。
            *   计算相关性 (`.corr()`)，绘制热力图/散点图 (抽样 -> Pandas)。
-   [ ] **数据预处理迁移 (`preprocessing.py`)**:
    -   [ ] `resample_demand_to_hourly`: 使用 Spark 实现：
        *   将 `timestamp` 转换为小时 (`date_trunc` 或 `hour` 函数)。
        *   `groupBy('unique_id', 'hourly_timestamp').agg(F.sum('y').alias('y'))`。
    -   [ ] `validate_resampling`: 抽样检查结果。
-   [ ] **工具函数迁移 (`eda_utils.py`)**:
    -   [ ] 检查 `save_plot`, `plot_*_distribution`, `log_value_counts` 是否仍适用 (它们主要操作 Pandas 对象，应与抽样后的数据配合)。
    -   [ ] 移除或重构 `dask_compute_context`。
-   [ ] **更新运行脚本**:
    -   [ ] 确保所有脚本 (`0_download_data.py`, `2_run_preprocessing.py` 等) 使用 SparkSession 并调用迁移后的函数。
    -   [ ] 移除 Dask 相关代码。

## 阶段一：数据加载与概览 (原 Dask/Pandas)

-   [x] **加载数据**:
    -   [x] 加载 Demand 数据 (`load_demand_data`)
    -   [x] 加载 Metadata 数据 (`load_metadata`)
    -   [x] 加载 Weather 数据 (`load_weather_data`)
-   [x] **数据基本信息检查 (在加载时完成)**:
    -   [x] 打印各数据集形状、列名、数据类型
    -   [x] 打印 Metadata 头部信息

## 阶段二：单变量分析 (原 Dask/Pandas)

-   [x] **Demand 数据 ('y' 列)**:
    -   [x] 分析 'y' 列分布 (抽样) (`analyze_demand_y_distribution`)
    -   [x] 绘制 'y' 列分布图 (原始尺度和对数尺度) (`plot_demand_y_distribution`)
    -   [x] 分析时间序列特性 (抽样 unique_id) (`analyze_demand_timeseries_sample`)
        -   [x] 绘制抽样时间序列图
        -   [x] 分析抽样数据时间戳频率

-   [x] **Metadata 数据**:
    -   [x] 分析分类特征分布 (`analyze_metadata_categorical`)
    -   [x] 绘制分类特征分布图 (`plot_metadata_categorical`)
    -   [x] 分析数值特征分布 (`analyze_metadata_numerical`)
    -   [x] 绘制数值特征分布图 (`analyze_metadata_numerical` 中包含)
    -   [x] 分析缺失位置信息 (`analyze_missing_locations`)

-   [x] **Weather 数据**:
    -   [x] 分析数值特征分布 (`analyze_weather_numerical`)
    -   [x] 绘制数值特征分布图 (`analyze_weather_numerical` 中包含)
    -   [x] 分析分类特征分布 (`analyze_weather_categorical`)
    -   [x] 绘制分类特征分布图 (`analyze_weather_categorical` 中包含)

## 阶段三：关系分析 (原 Dask/Pandas)

-   [x] **Demand vs Metadata**:
    -   [x] Demand (y) vs building_class (`analyze_demand_vs_metadata`)
    -   [x] Demand (y) vs location (Top N) (`analyze_demand_vs_location`)

-   [-] **Demand vs Weather (阻塞)**:
    -   [-] 计算相关性 (`analyze_demand_vs_weather`)  <-- **因 `merge_asof` 排序错误未完成**
    -   [-] 绘制相关性热力图 (`analyze_demand_vs_weather`) <-- **未执行**
    -   [-] 绘制关键特征散点图 (`analyze_demand_vs_weather`) <-- **未执行**

## 阶段四：数据预处理与特征工程 (原 Dask/Pandas)

-   [>] **数据对齐 (统一时间频率)**:
    -   [>] **确定对齐策略**: 将 Demand 数据重采样到 1 小时频率，聚合方式为 `sum`。
    -   [ ] **实现 Demand 数据重采样**:
        -   [x] 按 `unique_id` 分组。
        -   [x] 将 `timestamp` 设为索引。
        -   [x] 使用 `resample('1H').sum()` 聚合 `y`。
        -   [x] 重置索引。 (使用 Dask 完成)
    -   [ ] **验证重采样结果**: 检查时间戳和抽样数据。
-   [ ] **数据清洗**:
    -   [ ] 缺失 `y` 值处理方案 (考虑重采样后的情况)。
    -   [ ] 缺失位置信息处理方案。
    -   [ ] 非正 `y` 值处理方案。
-   [ ] **特征工程**:
    -   [ ] 时间特征创建 (年、月、日、星期、小时等)。
    -   [ ] 滞后特征创建 (基于重采样后的 `y`)。
    -   [ ] 天气特征（含 `weather_code`）处理/编码。
    -   [ ] 元数据分类特征编码 (`building_class`, `location`?)。
    -   [ ] 滚动统计特征创建 (例如，过去 N 小时的平均/最大/最小需求)。
-   [ ] **特征缩放**:
    -   [ ] 确定数值特征缩放方法 (如 StandardScaler)。
-   [ ] **目标 `y` 变换**:
    -   [ ] 考虑是否需要对 `y` 进行变换 (如 log 变换) 以改善分布。

## 阶段五：建模与评估 (原 Dask/Pandas)

-   [ ] **制定建模初步计划**:
    -   [ ] 候选模型选择 (例如，LightGBM, XGBoost, 线性模型？)。
    -   [ ] 验证策略 (例如，时间序列交叉验证)。
    -   [ ] 评估指标 (例如，MAE, RMSE, MAPE)。
-   [ ] **模型训练与评估**:
    -   [ ] 实现训练流程。
    -   [ ] 进行模型评估。

---

**当前任务**: 继续迁移 EDA 分析函数到 PySpark。

---

**下一步建议**:

我们已经成功将数据加载和基础检查迁移到了 Spark。现在 `1_run_eda.py` 可以使用 Spark 加载数据，并将 Metadata 数据转换为 Pandas 进行分析。

下一步，我们应该开始迁移处理 Demand 和 Weather 这两个大数据集的 EDA 函数，这些函数不能简单地 `toPandas()`。

**建议优先迁移 `analyze_demand.py` 中的函数**，因为它们是分析核心目标变量 `y` 的关键。你觉得如何？