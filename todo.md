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
    -   [x] **Demand 分析 (`analyze_demand.py`)**:
        -   [x] `analyze_demand_y_distribution`: 使用 Spark `.summary()`/`.approxQuantile()`, `.filter().count()`, `.sample()` -> Pandas。
        -   [x] `plot_demand_y_distribution`: 接收 Pandas Series (来自 Spark 抽样)，无需修改。
        -   [x] `analyze_demand_timeseries_sample`: 使用 Spark `.distinct()`, `.takeSample()`, `.filter()` -> Pandas -> 绘图/频率分析。
    -   [ ] **Metadata 分析 (`analyze_metadata.py`)**:
        -   [x] (已部分完成) 当前策略：如果内存允许，将 Metadata collect 到 Pandas (`.toPandas()`)，然后复用现有函数。检查内存占用。
        -   [ ] 备选方案 (如果内存不足)：重写函数，使用 Spark API (`.groupBy().count()`, `.describe()`) 进行分析，抽样 -> Pandas -> 绘图。
    -   [ ] **Weather 分析 (`analyze_weather.py`)**:
        -   [ ] `analyze_weather_numerical`: 使用 Spark `.describe()` 或 `.summary()` 获取统计信息。抽样 (`.sample()`) -> Pandas -> 调用 `plot_numerical_distribution` 绘图。增加 Spark 方式的负值检查。
        -   [ ] `analyze_weather_categorical`: 使用 Spark 的 `.groupBy().count()` 计算分类特征的频数。抽样或排序 (`orderBy()`) -> Pandas -> 调用 `plot_categorical_distribution` 绘图。
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
    -   [x] 检查 `save_plot`, `plot_*_distribution`, `log_value_counts` 是否仍适用 (它们主要操作 Pandas 对象，与抽样/聚合后的数据配合)。
    -   [x] 移除 `dask_compute_context` 及其相关导入。
-   [x] **更新运行脚本**:
    -   [x] 确保主脚本 (`1_run_eda.py`) 使用 SparkSession 并调用迁移后的函数。
    -   [x] 移除 Dask 相关代码。

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

**当前任务**: 继续迁移 EDA 分析函数到 PySpark，下一步是 **Weather 分析**。

---

**下一步建议**:

我们已经成功迁移了 Demand 数据分析部分。下一步是迁移 **Weather 数据分析函数 (`analyze_weather.py`)**。这与 Demand 的迁移类似：

1.  `analyze_weather_numerical`: 使用 Spark `.describe()` 或 `.summary()` 获取统计信息。抽样 (`.sample()`) -> Pandas -> 调用 `plot_numerical_distribution` 绘图。增加 Spark 方式的负值检查。
2.  `analyze_weather_categorical`: 使用 Spark 的 `.groupBy().count()` 计算分类特征的频数。抽样或排序 (`orderBy()`) -> Pandas -> 调用 `plot_categorical_distribution` 绘图。

你觉得这个计划如何？

## 当前进度

- [x] 将项目基础框架迁移到 Spark (SparkSession 初始化/停止, 基本配置)。
- [x] 修改 `load_data.py` 使用 Spark 加载 Parquet 文件。
- [x] 更新 `1_run_eda.py` 调用 Spark 加载函数，并管理 SparkSession 生命周期。
- [x] 确认 `analyze_demand.py` 已适配 Spark DataFrame 输入 (抽样 -> Pandas)。
- [x] 确认 `analyze_metadata.py` 仍可使用 (通过 Spark DF -> Pandas DF 转换)。
- [x] 迁移 `analyze_weather.py` 以使用 Spark DataFrame 进行计算和抽样。
- [x] 迁移 `analyze_relationships.py` (`vs_metadata`, `vs_location`) 使用 Spark 抽样 -> Pandas 合并/分析。
- [x] 迁移 `analyze_relationships.py` (`vs_weather`) 使用 Spark 过滤 -> Pandas 合并 (`merge_asof`) / 分析 (注意内存)。
- [x] 迁移 `eda_utils.py` (移除 Dask 相关代码)。
- [x] 更新 `1_run_eda.py` 以调用所有已迁移的分析函数。

## 下一步任务

1.  **运行和测试**: 完整运行 `1_run_eda.py` 脚本。
2.  **检查输出**:
    *   检查日志 (`logs/1_run_eda.log`) 是否有错误或警告，特别是关于内存、数据类型、合并步骤的。
    *   检查 `plots/` 目录下生成的图表是否符合预期。
3.  **性能和内存**: 观察 Spark UI (`http://localhost:4040` 或类似地址) 监控任务执行情况和资源使用。如果 `analyze_demand_vs_weather` 步骤因内存不足失败，考虑：
    *   减少 `n_sample_ids` 的数量。
    *   增加 Spark Driver 内存 (`spark.driver.memory`)。
    *   (高级) 探索在 Spark 中直接实现近似 `merge_asof` 的方法（如果必须处理非常大的样本）。
4.  **代码注释和优化**:
    *   根据运行结果，调整日志级别或内容。
    *   清理不再需要的旧代码或注释。
    *   确保遵循代码风格。
5.  **细化分析**: 根据初步 EDA 结果，决定是否需要进行更深入的特定分析。
6.  **迁移预处理**: 开始迁移 `preprocessing.py` 中的函数到 Spark。