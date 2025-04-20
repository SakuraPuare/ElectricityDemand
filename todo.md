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

-   [>] **Demand vs Metadata**:
    -   [>] Demand (y) vs building_class (`analyze_demand_vs_metadata`)
    -   [>] Demand (y) vs location (Top N) (`analyze_demand_vs_location`)

-   [>] **Demand vs Weather**:
    -   [>] 计算相关性 (`analyze_demand_vs_weather`)
    -   [>] 绘制相关性热力图 (`analyze_demand_vs_weather`)
    -   [>] 绘制关键特征散点图 (`analyze_demand_vs_weather`)

## 阶段四：总结与后续步骤

-   [ ] 总结 EDA 发现
-   [ ] 识别数据清洗和特征工程方向
-   [ ] 准备建模所需数据

---

**下一步建议**:

当前任务是 **执行关系分析** (`Demand vs Metadata` 和 `Demand vs Weather`)。我们还将重新运行 Metadata 分析以确认之前的修复有效。请运行 `run_eda.py` 脚本。注意：关系分析，特别是 `Demand vs Weather` 可能需要较长时间进行计算，因为它涉及合并和处理较大的 Dask DataFrame。请关注日志输出和 `plots` 目录下的新图表 (如 `demand_vs_*.png`, `correlation_heatmap.png`, `demand_weather_scatter_*.png`)。

完成后请告诉我，我们将评估结果并进入总结阶段。