# Electricity Demand 数据分析任务列表

## 第一阶段：数据加载与初步探索

1.  **设置项目环境**
    *   [x] 创建项目结构 (src, data, logs, notebooks, etc.)
    *   [x] 初始化 Python 环境 (e.g., venv)
    *   [x] 安装所需库 (dask, pandas, loguru, matplotlib, seaborn)
    *   [x] 配置日志系统 (`src/electricitydemand/utils/log_utils.py`)

2.  **数据加载与质量检查 (`src/electricitydemand/load_data.py`)**
    *   [x] 使用 Dask 加载 `demand.parquet`, `metadata.parquet`, `weather.parquet`
    *   [x] 记录数据集基本信息 (分区数, 列名, 行数, 样本数据, 数据类型)
    *   [x] 检查并记录缺失值及其比例
    *   [~] 检查重复值
        *   [x] Demand: 基于 `unique_id` 和 `timestamp` (分区内检查未发现重复)
        *   [~] Metadata: 基于 `unique_id` (全局检查因 `AttributeError` 暂时跳过精确计数)
        *   [~] Weather: 基于 `location_id` 和 `timestamp` (全局检查因 `AttributeError` 暂时跳过精确计数，已知存在少量重复)
    *   [x] 记录 Demand 和 Weather 的时间戳范围
    *   [x] (已完成) 暂时注释掉 `load_data.py` 中导致错误的全局重复值检查代码和 `main` 函数调用。

## 第二阶段：探索性数据分析 (EDA)

1.  **创建 EDA 脚本 (`src/electricitydemand/eda.py`)**
    *   [x] 设置脚本结构，导入必要库和日志配置。
    *   [x] 添加函数以加载数据集 (`load_demand_data`)。

2.  **Demand ('y') 分析**
    *   [x] **分布分析:**
        *   [x] 计算 `y` 列的详细描述性统计信息 (基于 0.5% 抽样完成)。
        *   [x] 检查非正值 (<= 0) (基于 0.5% 抽样完成)。
        *   [x] 绘制 `y` 列的直方图和箱线图 (原始尺度和对数尺度)，可视化其分布 (基于进一步抽样绘图完成)。
    *   [x] **时间序列特性 (抽样):**
        *   [x] 抽取少量 (`N=5`) `unique_id` 的数据。
        *   [x] 绘制这些样本的时间序列图，观察趋势、季节性、异常值。
        *   [x] 分析不同 `unique_id` 的时间戳频率是否一致 (与 metadata 中的 `freq` 对比)。

3.  **Metadata 分析**
    *   [ ] 分析分类特征的分布 (`building_class`, `location`, `freq`, `timezone`, `dataset`)。使用计数图 (count plot) 或条形图。
    *   [ ] 分析数值特征 (`latitude`, `longitude`, `cluster_size`) 的分布（如果适用）。
    *   [ ] 检查 `location_id`, `latitude`, `longitude`, `location` 缺失值的具体情况（哪些 `unique_id` 缺失）。

4.  **Weather 分析**
    *   [ ] 分析关键数值天气特征的分布 (e.g., `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`)。使用直方图、箱线图。
    *   [ ] 检查数值特征是否存在异常值或不合理的值（如负降水量）。
    *   [ ] 分析 `weather_code` 的分布。
    *   [ ] 分析天气数据的时间戳频率。

5.  **关系分析 (抽样)**
    *   [ ] **Demand vs. Metadata:**
        *   [ ] 按 `building_class` 分组，比较 `y` 的分布 (箱线图)。
        *   [ ] 按 `location` 或 `timezone` 分组，比较 `y` 的分布。
    *   [ ] **Demand vs. Weather:**
        *   [ ] 将 Demand 数据与对应的 Weather 数据合并 (需要基于 `unique_id` -> `location_id` 和 `timestamp` 进行匹配，注意处理时间戳对齐)。
        *   [ ] 计算 `y` 与关键天气特征的相关性系数 (如温度, 湿度, 降水)。
        *   [ ] 绘制 `y` 与关键天气特征的散点图。

6.  **保存可视化结果**
    *   [ ] 将生成的图表保存到项目目录下的 `plots` 或类似文件夹中。

## 第三阶段：数据预处理与特征工程 (待定)

*   处理缺失值 (填充或删除)
*   处理重复值 (Weather)
*   时间戳对齐 (Demand vs Weather)
*   特征创建 (e.g., 时间特征: 小时, 星期几, 月份; 滞后特征; 天气交互特征)
*   数据标准化/归一化

## 第四阶段：模型构建与评估 (待定)

*   选择合适的预测模型 (e.g., ARIMA, Prophet, LightGBM, LSTM)
*   划分训练集、验证集、测试集
*   模型训练与调优
*   模型评估 (选择合适的评估指标, e.g., MAE, RMSE, MAPE)

## 第五阶段：结果解释与报告 (待定)

*   分析模型预测结果
*   解释特征重要性
*   撰写分析报告

---
**项目技术栈:** Dask, Loguru, Pandas, Matplotlib/Seaborn

## 数据分析任务列表

## 进行中 ⏳

*   **探索性数据分析 (EDA) - Metadata 分析**:
    *   加载 `metadata.parquet` 数据。
    *   分析分类特征 (`building_class`, `location`, `freq`, `timezone`, `dataset`) 的分布。

## 下一步 ➡️

*   **探索性数据分析 (EDA)**:
    *   **Metadata 分析**: 可视化分类特征的分布 (计数图/条形图)。
    *   **Metadata 分析**: 分析数值特征 (`latitude`, `longitude`, `cluster_size`) 的分布。
    *   **Metadata 分析**: 检查地理位置相关列的缺失值情况。
    *   **关系分析**: 开始探索 Demand 与 Metadata 之间的关系。

## 已完成 ✅
*   环境设置与数据下载
*   数据加载与基本信息查看
*   数据质量检查 (缺失值, 重复值初步检查, 时间范围)
*   `load_data.py` 重构和错误处理
*   `eda.py` 脚本创建和数据加载
*   Demand ('y') 分布的描述性统计计算 (基于抽样)
*   Demand ('y') 非正值检查 (基于抽样)
*   Demand ('y') 分布的可视化 (直方图/箱线图，原始/对数尺度，基于抽样)
*   Demand ('y') 时间序列特性分析 (基于 N=5 抽样)