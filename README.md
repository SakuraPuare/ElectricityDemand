# 智能电力需求分析系统

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Spark-3.5.5-orange.svg" alt="Spark 3.5.5">
  <img src="https://img.shields.io/badge/pandas-2.2.3-green.svg" alt="pandas 2.2.3">
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-yellow.svg" alt="scikit-learn 1.6.1">
  <img src="https://img.shields.io/badge/LightGBM-4.6.0-brightgreen.svg" alt="LightGBM 4.6.0">
  <img src="https://img.shields.io/badge/Dask-2025.3.0-blue.svg" alt="Dask 2025.3.0">
</div>

## 📊 项目概述

智能电力需求分析系统是一个基于大数据技术的电力消费数据分析平台，利用 Apache Spark 进行高效的数据处理和分析。该系统通过对多个开源智能电表数据集的整合和分析，帮助理解电力消费模式，预测未来电力需求，并为能源规划和管理提供数据支持。

### 主要功能

- **数据探索分析**：对电力需求、元数据和天气数据进行全面的探索性分析
- **数据预处理**：处理缺失值、重复值，进行数据重采样和合并
- **特征工程**：提取时间特征、天气特征，创建滚动窗口特征
- **模型训练与评估**：使用机器学习模型进行电力需求预测
- **可视化与报告**：通过图表直观展示分析结果
- **时间序列分析**：识别电力需求的季节性和趋势变化

## 🛠️ 技术栈

- **核心框架**：Apache Spark (PySpark)
- **数据处理**：Pandas, NumPy, Dask
- **机器学习**：Scikit-learn, LightGBM
- **数据可视化**：Matplotlib, Seaborn, HvPlot, Panel
- **日志管理**：Loguru
- **数据存储**：Parquet 文件格式
- **特征处理**：时间特征提取、滚动窗口分析

## 📁 项目结构

```
ElectricityDemand/
├── data/                       # 数据目录
│   ├── features.parquet/       # 特征工程后的数据 (按年月分区)
│   ├── merged_data.parquet/    # 合并后的数据
│   └── processed/              # 中间处理数据
├── logs/                       # 日志文件
├── models/                     # 模型存储
│   └── mllib_linear_regression_model/ # Spark MLlib线性回归模型
├── plots/                      # 可视化图表
├── reports/                    # 分析报告
│   └── log/                    # 分析日志报告
├── src/electricitydemand/      # 源代码
│   ├── eda/                    # 数据探索分析模块
│   ├── funcs/                  # 核心功能实现
│   │   ├── data_processing.py  # 数据处理功能
│   │   ├── feature_engineering.py # 特征工程功能
│   │   └── model_training.py   # 模型训练功能
│   └── utils/                  # 工具函数
│       ├── log_utils.py        # 日志工具
│       ├── spark_utils.py      # Spark工具
│       └── project_utils.py    # 项目通用工具
├── .gitignore                  # Git忽略文件
├── pyproject.toml              # 项目依赖配置
├── requirements.lock           # 依赖锁定文件
├── requirements-dev.lock       # 开发依赖锁定文件
└── README.md                   # 项目说明
```

## 📊 数据集介绍

本项目使用 [Electricity Demand Dataset](https://huggingface.co/datasets/EDS-lab/electricity-demand)，该数据集整合了多个开源智能电表数据集，包含以下主要文件：

1. **需求数据 (demand.parquet)**：
   - 包含约 2.38 亿条电力消费记录
   - 字段：唯一标识符、时间戳、电力消费量 (kWh)
   - 时间范围：2011-01-01 至 2017-12-31

2. **元数据 (metadata.parquet)**：
   - 包含 7572 个测量点的元数据
   - 字段：唯一标识符、建筑类型、位置、时区等
   - 主要建筑类型：住宅 (Residential) 和商业 (Commercial)

3. **天气数据 (weather.parquet)**：
   - 包含约 60.5 万条天气记录
   - 字段：位置标识符、时间戳、温度、湿度、降水量等
   - 时间范围：2011-01-01 至 2019-01-01

### 数据特点

- **多样化时间频率**：数据采样频率包括 15 分钟、30 分钟和 1 小时
- **地理分布**：主要集中在英国伦敦地区
- **数据质量**：总体质量较高，存在少量缺失值和异常值

## 🚀 安装与使用

### 环境要求

- Python 3.12
- Apache Spark 3.5.5
- Rye（Python 项目管理工具）
- Hugging Face 账号和访问令牌
- 其他依赖见 `pyproject.toml`

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/SakuraPuare/ElectricityDemand.git
   cd ElectricityDemand
   ```

2. 使用 Rye 管理环境
   ```bash
   # 安装 Rye（如果尚未安装）
   curl -sSf https://rye-up.com/get | bash
   
   # 初始化项目环境
   rye sync
   
   # 激活环境
   . .venv/bin/activate  # Linux/Mac
   # 或
   .venv\Scripts\activate  # Windows
   ```

3. 配置 Hugging Face 访问令牌
   ```bash
   # 设置环境变量
   export HF_TOKEN="你的 Hugging Face 访问令牌"  # Linux/Mac
   # 或
   set HF_TOKEN=你的 Hugging Face 访问令牌  # Windows
   
   # 或使用 huggingface-cli 登录
   huggingface-cli login
   ```

### 使用方法

1. 数据下载与转换（需要 Hugging Face 访问令牌）
   ```bash
   python src/electricitydemand/00_download_data.py
   python src/electricitydemand/01_convert_parquet.py
   ```

2. 数据探索分析
   ```bash
   python src/electricitydemand/1_run_eda.py
   ```

3. 数据预处理
   ```bash
   python src/electricitydemand/2_run_preprocessing.py
   ```

4. 特征工程
   ```bash
   python src/electricitydemand/3_run_feature_engineering.py
   ```

5. 模型训练与评估
   ```bash
   python src/electricitydemand/4_run_model_training.py
   ```

## 📈 分析结果

项目已完成对电力需求数据的全面分析，包括：

-   **数据概览与质量检查**: 对数据量、缺失值、重复值和时间范围进行了评估。发现了电力需求（y）和元数据（地理位置）的少量缺失，以及天气数据中的少量重复记录（已处理）。
-   **需求分布分析**：发现电力需求呈高度右偏分布，商业建筑需求显著高于住宅建筑，存在大量小值和少量极端高值。
-   **周期性模式**：识别出电力需求的每日、每周和季节性模式，并通过时间特征提取和周期性分析图表进行了量化展示。
-   **天气影响**：发现电力需求与温度、湿度等天气因素存在显著相关性。相对湿度呈中度负相关 (-0.202)，温度和体感温度呈弱正相关。同时分析了天气变量间的相关性。
-   **元数据影响**: 元数据中的分类信息（如建筑类型、地理位置、数据集来源）与电力需求水平显著相关。
-   **时间频率匹配**: 解决了需求数据（多频率）与天气数据（1H）时间频率不匹配的问题，通过重采样和时间戳对齐成功合并了数据。
-   **特征工程**：成功提取时间特征（年、月、日、星期几、小时等）、天气特征和滚动统计特征（均值、标准差、最大最小值等），并处理了目标变量缺失和滚动窗口初始期特征缺失的问题。最终特征集已按时间（年月）分区存储。

详细的分析结果和可视化图表可在 `reports` 和 `plots` 目录中查看。

## 📊 核心发现

基于探索性分析和数据处理，我们得出以下核心发现：

1.  **数据质量问题**: 尽管整体数据质量较好，但电力需求目标值 (y) 存在约 1.3% 的缺失；元数据中与地理位置相关的列存在约 3.1% 的缺失，且部分地理位置标识符在天气数据中无匹配项，导致约 1.73% 的合并记录天气信息缺失；天气数据存在极少量重复记录（已处理）。这些问题需要在后续建模时考虑。
2.  **电力需求分布**: 需求值高度右偏，中位数远小于均值，存在极端高值和少量非正值，表明其分布复杂且需要适当处理（如对数变换）。
3.  **建筑类型差异**：商业建筑的电力需求中位数和波动范围显著高于住宅建筑，这是预测中一个重要的区分因素。
4.  **时间模式**：电力需求表现出清晰的多重周期性：
    *   **日内**: 典型的高峰（白天）和低谷（夜晚）模式。
    *   **周内**: 工作日与周末的用电模式存在明显差异。
    *   **年度/季节性**: 冬季和夏季的需求通常高于春秋季。
5.  **地理位置集中与差异**: 数据主要集中在英国伦敦地区，不同地理位置的需求模式和水平存在差异。
6.  **天气影响因素**：天气变量与电力需求存在可量化的相关性，特别是相对湿度（负相关）和温度/体感温度（弱正相关）。极端温度对需求有影响。天气变量内部存在较强相关性（如温度相关变量），需关注多重共线性。
7.  **时间频率不匹配的处理**: 需求数据的多频率（15T, 30T, 1H）与天气数据（1H）的不匹配问题已通过需求数据的重采样和基于小时级时间戳的合并策略得到解决。
8.  **关键特征**: 时间特征（如小时、星期几、月份）和历史需求（滚动统计特征）是捕捉电力需求模式的关键特征，对预测具有重要价值。

## 🧠 模型性能

使用多种机器学习模型进行电力需求预测，包括：

- **线性回归**：提供基准性能，RMSE 约为 145 kWh
- **决策树**：提供更好的非线性关系捕捉，RMSE 约为 98 kWh
- **LightGBM**：提供最佳性能，RMSE 约为 68 kWh

特征重要性分析显示，时间特征（小时、星期几、月份）和历史用电量特征对预测贡献最大。

## 🔮 未来工作

1. **深度学习模型**：探索 LSTM、Transformer 等深度学习模型用于时间序列预测
2. **实时预测系统**：开发实时电力需求预测系统
3. **异常检测**：增加电力消费异常检测功能
4. **用户聚类分析**：基于用电模式进行用户分群
5. **能源效率建议**：根据分析结果提供能源使用优化建议

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 👥 贡献指南

欢迎对本项目进行贡献！请通过以下方式参与：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request