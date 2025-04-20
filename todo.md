# 电力需求数据分析项目

## 数据来源

https://huggingface.co/datasets/EDS-lab/electricity-demand

## 任务清单

### 1. 数据分析 (进行中)
- [x] **环境设置与数据下载**
  - [x] 配置项目环境 (`pyproject.toml`)
  - [x] 实现日志记录功能 (`src/electricitydemand/utils/log_utils.py`)
  - [x] 实现数据下载脚本 (`src/electricitydemand/download_data.py`)
  - [x] 运行数据下载脚本
  - [x] 实现数据加载脚本 (`src/electricitydemand/load_data.py`)
  - [x] 运行数据加载脚本
- [ ] **数据概览与质量检查 (进行中)**
  - [x] 查看数据形状 (分区数)、列名、数据类型
  - [x] 计算行数并查看数据样本
  - [x] 检查并统计缺失值
  - [ ] **检查并处理重复值** <--- **当前任务**
  - [ ] 确定各数据集的时间范围
- [ ] **探索性数据分析 (EDA)**
  - [ ] 分析 `demand` 数据 (`y`) 的分布
  - [ ] 分析 `metadata` 数据 (e.g., `building_class`, `location`)
  - [ ] 分析 `weather` 数据 (e.g., `temperature_2m`, `precipitation`)
  - [ ] 分析 `demand` 与 `metadata` 的关系 (e.g., 不同 `building_class` 的需求差异)
  - [ ] 分析 `demand` 与 `weather` 的关系 (e.g., 需求与温度、湿度的相关性)
  - [ ] 分析时间序列特性 (初步)
- [ ] **数据预处理 (初步)**
  - [ ] (待定，根据 EDA 和建模目标确定)

### 2. (后续任务，例如特征工程、模型训练等)

---
**项目技术栈:** Dask, Loguru, Pandas, Matplotlib/Seaborn

## 数据分析任务列表

## 进行中 ⏳

*   **数据质量检查**: 检查各数据集的重复值 (`.duplicated().sum()`).

## 下一步 ➡️

*   **数据质量检查**:
    *   确定各数据集的时间范围 (`.timestamp.min()`, `.timestamp.max()`).
*   **探索性数据分析 (EDA)**: 开始分析各数据集的分布等。

## 已完成 ✅

*   环境设置与数据下载
*   数据加载与基本信息查看 (列名, 分区数)
*   计算行数并查看数据样本
*   检查并统计缺失值