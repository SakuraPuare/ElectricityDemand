# Electricity Demand EDA 任务列表

## 阶段一：数据加载与概览

-   [x] **加载数据**:
    -   [x] 加载 Demand 数据 (`load_demand_data`)
    -   [x] 加载 Metadata 数据 (`load_metadata`)
    -   [x] 加载 Weather 数据 (`load_weather_data`)
-   [x] **数据基本信息检查 (在加载时完成)**:
    -   [x] 打印各数据集形状、列名、数据类型
    -   [x] 打印 Metadata 头部信息

## 阶段二：单变量分析

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
    -   [ ] *(待定)* 分析天气数据时间戳频率 (如果需要单独函数)

## 阶段三：关系分析 (基于抽样)

-   [x] **Demand vs Metadata**:
    -   [x] Demand (y) vs building_class (`analyze_demand_vs_metadata`)
    -   [x] Demand (y) vs location (Top N) (`analyze_demand_vs_location`)

-   [-] **Demand vs Weather (阻塞)**:
    -   [-] 计算相关性 (`analyze_demand_vs_weather`)  <-- **因 `merge_asof` 排序错误未完成**
    -   [-] 绘制相关性热力图 (`analyze_demand_vs_weather`) <-- **未执行**
    -   [-] 绘制关键特征散点图 (`analyze_demand_vs_weather`) <-- **未执行**

## 阶段四：数据预处理与特征工程

-   [>] **数据对齐 (统一时间频率)**:
    -   [>] **确定对齐策略**: 将 Demand 数据重采样到 1 小时频率，聚合方式为 `sum`。
    -   [ ] **实现 Demand 数据重采样**:
        -   [ ] 按 `unique_id` 分组。
        -   [ ] 将 `timestamp` 设为索引。
        -   [ ] 使用 `resample('1H').sum()` 聚合 `y`。
        -   [ ] 重置索引。
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

## 阶段五：建模与评估

-   [ ] **制定建模初步计划**:
    -   [ ] 候选模型选择 (例如，LightGBM, XGBoost, 线性模型？)。
    -   [ ] 验证策略 (例如，时间序列交叉验证)。
    -   [ ] 评估指标 (例如，MAE, RMSE, MAPE)。
-   [ ] **模型训练与评估**:
    -   [ ] 实现训练流程。
    -   [ ] 进行模型评估。

## 待办/讨论

-   [ ] **决定是否修复 `merge_asof` 问题**: 暂时搁置，优先进行数据对齐和特征工程。如果后续模型效果不佳或需要更精细分析，再考虑修复。

---

**当前任务**: 实现 Demand 数据的 1 小时重采样。

---

**下一步建议**:

EDA 总结已完成。下一步的关键是**决策后续方向**。请根据上述总结和讨论点，确定我们优先处理哪些任务：

1.  **数据清洗**: 确定如何处理缺失值和异常值。
2.  **数据对齐**: **决定如何统一 Demand 和 Weather 的时间频率** (例如，将 Demand 重采样到小时)。这是进行后续特征工程和建模的前提。
3.  **特征工程**: 开始构思和实现新的特征。
4.  **修复 Bug**: 是否现在就修复 `analyze_demand_vs_weather` 的合并问题？

**建议优先解决第 2 点（数据对齐）**，因为它对后续所有步骤都有影响。你觉得呢？