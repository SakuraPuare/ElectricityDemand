# Electricity Demand EDA & Forecasting - TODO

## 任务列表

- [ ] **环境设置与数据下载**
    - [x] 创建项目结构
    - [x] 配置 `conda` 环境 (`environment.yml`)
    - [x] 安装依赖
    - [x] 从 Hugging Face Hub 下载数据集 (`data/`)
    - [x] 转换数据格式 (如果需要，例如 CSV -> Parquet)
- [ ] **初步探索性数据分析 (EDA) - 使用 Spark**
    - [x] **加载数据**: 使用 Spark 读取 Parquet 文件 (`demand`, `metadata`, `weather`)。
    - [ ] **单变量分析**:
        - [x] **Demand (`y`)**:
            - [x] 计算描述性统计 (均值、中位数、标准差、分位数等)。
            - [x] 检查非正值 (<= 0)。
            - [x] 绘制 `y` 的分布图 (原始尺度和对数尺度，使用抽样)。
            - [x] 抽样分析时间序列 (随机选几个 `unique_id`)，检查时间频率。
        - [x] **Metadata**:
            - [x] (转 Pandas) 分析分类特征 (`building_class`, `location`, `freq`, `timezone`, `dataset`) 的值分布并绘图。
            - [x] (转 Pandas) 分析数值特征 (`latitude`, `longitude`, `cluster_size`) 的分布并绘图。
            - [x] (转 Pandas) 分析缺失的地理位置信息 (`location_id`, `latitude`, `longitude`, `location`)。
        - [x] **Weather**:
            - [x] 计算关键数值特征的描述性统计 (如 `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`, `cloud_cover`)。
            - [x] 检查负值 (如 `precipitation`, `rain`, `snowfall`)。
            - [x] 绘制数值特征的分布图 (使用抽样)。
            - [x] 分析分类特征 (`weather_code`, `is_day`) 的值分布并绘图。
            - [x] 抽样分析时间序列 (随机选几个 `location_id`)，检查时间频率。
    - [ ] **关系分析**:
        - [ ] **Demand vs Metadata**:
            - [ ] (抽样 Spark -> Pandas) 分析 `y` 与 `building_class` 的关系 (箱线图)。
            - [ ] (抽样 Spark -> Pandas) 分析 `y` 与 `location` 的关系 (箱线图, Top N location)。
        - [ ] **Demand vs Weather**:
            - [ ] (抽样 Spark Join -> Pandas) 合并抽样后的 `demand` 和 `weather` 数据 (基于 `unique_id -> location_id` 和 `timestamp`)。
            - [ ] 计算 `y` 与关键天气特征的相关性。
            - [ ] 绘制 `y` 与关键天气特征的散点图。
    - [ ] **时间特征分析**:
        - [ ] **Demand**: 分析 `timestamp` 列 (分布、缺失、间隔、时间成分如小时/星期几/月份)。
        - [ ] **Weather**: 分析 `timestamp` 列 (同上)。
    - [ ] **数据质量总结**:
        - [ ] 汇总缺失值情况。
        - [ ] 汇总重复值情况。
        - [ ] 记录数据时间范围。
        - [ ] 总结时间戳频率的不匹配问题。
    - [ ] **修复 EDA 脚本错误**
        - [x] 修复 `TypeError: DataFrame.sample() got an unexpected keyword argument 'frac'`
        - [x] 修复 `TypeError: analyze_demand_vs_weather() got multiple values for argument 'plots_dir'`
- [ ] **数据预处理**
    - [ ] 处理缺失值 (Demand `y`, Metadata `location` 等)。
    - [ ] 处理异常值/极端值 (Demand `y`)。
    - [ ] 时间戳对齐：将 Demand 和 Weather 数据统一到相同的时间频率 (例如，将 Demand 聚合到小时)。
    - [ ] 特征工程：
        - [ ] 创建时间相关的特征 (小时、星期几、月份、年份、节假日标志等)。
        - [ ] 创建滞后特征 (Lag features) for Demand `y`。
        - [ ] 创建滚**动窗口特征 (Rolling window features) for Demand `y` (e.g., rolling mean)。
        - [ ] 可能的天气特征交互或转换。
    - [ ] 数据划分 (训练集、验证集、测试集)，注意时间序列数据的划分方式。
    - [ ] 特征缩放 (如果模型需要)。
- [ ] **模型选择与训练**
    - [ ] 选择基线模型 (e.g., Naive forecast, SARIMA)。
    - [ ] 选择机器学习模型 (e.g., LightGBM, XGBoost)。
    - [ ] (可选) 深度学习模型 (e.g., LSTM, Transformer)。
    - [ ] 训练模型。
- [ ] **模型评估与调优**
    - [ ] 定义评估指标 (e.g., MAE, RMSE, MAPE)。
    - [ ] 在验证集上评估模型。
    - [ ] 进行超参数调优 (e.g., Grid Search, Random Search, Bayesian Optimization)。
    - [ ] 特征重要性分析。
- [ ] **模型预测与结果可视化**
    - [ ] 在测试集上进行预测。
    - [ ] 可视化预测结果与真实值的对比。
    - [ ] 分析预测误差。
- [ ] **报告撰写**
    - [ ] 总结 EDA 发现。
    - [ ] 描述数据预处理步骤。
    - [ ] 阐述模型选择、训练和评估过程。
    - [ ] 展示最终结果和结论。


## 当前任务

*   **重新运行 EDA 脚本**：执行 `python src/electricitydemand/1_run_eda.py` 来验证修复是否成功，并完成剩余的 EDA 步骤。

## 下一步任务 (如果 EDA 成功)

*   **数据预处理**:
    *   开始处理 `Demand` 数据中的缺失值 (`y`)。
    *   思考如何处理 `Metadata` 中的缺失地理位置信息 (可能需要移除这些 `unique_id` 的数据，或者根据其他信息估算)。
    *   确定如何处理 `Demand` 中的极端高值。
    *   设计时间戳对齐策略 (例如，将所有 `Demand` 数据聚合到小时级别以匹配 `Weather` 数据)。