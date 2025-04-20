# EDA 任务清单

## 数据加载与概览

*   [x] 加载 Demand 数据 (`ddf_demand`)
*   [x] 加载 Metadata 数据 (`pdf_metadata`)
*   [x] 加载 Weather 数据 (`ddf_weather`)
*   [x] 检查数据基本信息 (形状, 列名, 类型)

## 单变量分析

*   [x] **Demand (y)**
    *   [x] 分析 'y' 列分布 (抽样) - *计算统计量*
    *   [x] 可视化 'y' 列分布 (原始尺度和对数尺度)
    *   [x] 检查非正值比例
    *   [>] 分析时间序列特性 (抽样 `unique_id`, 绘制序列图, 分析频率) - *进行中*
*   [ ] **Metadata**
    *   [ ] 分析分类特征分布 (`building_class`, `location`, `freq`, etc.)
    *   [ ] 可视化分类特征分布 (Top N)
    *   [ ] 分析数值特征分布 (`latitude`, `longitude`, `cluster_size`) 并可视化
    *   [ ] 分析位置信息缺失情况
*   [ ] **Weather**
    *   [ ] 分析关键数值特征分布 (`temperature_2m`, `relative_humidity_2m`, etc.) 并可视化
    *   [ ] 检查特定列负值 (`precipitation`, `rain`, `snowfall`)
    *   [ ] 分析分类特征分布 (`weather_code`) 并可视化
    *   [ ] 分析时间戳频率 (抽样)

## 关系分析 (多变量)

*   [x] **Demand vs Metadata**
    *   [x] 分析 `y` 与 `building_class` 的关系 (箱线图)
    *   [x] 分析 `y` 与 `location` (Top N) 的关系 (箱线图)
*   [ ] **Demand vs Weather**
    *   [ ] ~~分析 `y` 与天气特征的关系 (合并, 相关性, 可视化)~~ - **跳过 (存在 `merge_asof` 排序 BUG)**

## 总结与下一步

*   [ ] 总结 EDA 发现的关键点
*   [ ] 确定数据预处理和特征工程的方向
*   [ ] 确定建模策略的初步想法

---

**当前任务:**

*   完成 **Demand (y)** 的时间序列特性分析 (抽样)。

**下一步:**

*   开始分析 **Metadata**。
*   (长期) 回头尝试修复 `analyze_demand_vs_weather` 中的 `merge_asof` bug 或寻找替代合并方法。