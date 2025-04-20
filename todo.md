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

## 阶段四：总结与后续计划

-   [>] **总结 EDA 发现**:
    -   [x] 数据量、缺失值、重复值、时间范围已确认。
    -   [x] Demand 'y' 呈高度右偏分布，含非正值，存在多种时间频率。
    -   [x] Metadata 主要类型、地点、频率已分析，缺失位置与特定数据集/类型相关。
    -   [x] Weather 数值、分类特征分布符合预期，频率为 1H。
    -   [x] Demand 与 building\_class, location 关系已初步可视化。
    -   [-] Demand 与 Weather 关系分析因技术问题阻塞。
-   [ ] **确定数据清洗策略**:
    -   [ ] 缺失 `y` 值处理方案。
    -   [ ] 缺失位置信息处理方案。
    -   [ ] 非正 `y` 值处理方案。
-   [ ] **设计特征工程方案**:
    -   [ ] 时间特征创建。
    -   [ ] 滞后特征创建。
    -   [ ] 天气特征（含 `weather_code`）处理。
    -   [ ] 元数据分类特征编码。
    -   [ ] 滚动统计特征创建。
-   [ ] **确定数据对齐与预处理策略**:
    -   [ ] **Demand 与 Weather 时间频率对齐方案 (关键!)**。
    -   [ ] 特征缩放方法。
    -   [ ] 目标 `y` 变换方法。
-   [ ] **制定建模初步计划**:
    -   [ ] 候选模型选择。
    -   [ ] 验证策略。
    -   [ ] 评估指标。
-   [ ] **决定是否修复 `merge_asof` 问题**:
    -   [ ] 投入时间修复 `analyze_demand_vs_weather`？ 或基于现有信息继续？

---

**下一步建议**:

EDA 总结已完成。下一步的关键是**决策后续方向**。请根据上述总结和讨论点，确定我们优先处理哪些任务：

1.  **数据清洗**: 确定如何处理缺失值和异常值。
2.  **数据对齐**: **决定如何统一 Demand 和 Weather 的时间频率** (例如，将 Demand 重采样到小时)。这是进行后续特征工程和建模的前提。
3.  **特征工程**: 开始构思和实现新的特征。
4.  **修复 Bug**: 是否现在就修复 `analyze_demand_vs_weather` 的合并问题？

**建议优先解决第 2 点（数据对齐）**，因为它对后续所有步骤都有影响。你觉得呢？