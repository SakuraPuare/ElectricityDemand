# 电力需求数据分析项目

## 数据来源
https://huggingface.co/datasets/EDS-lab/electricity-demand

## 任务清单

### 1. 数据分析 (进行中)
- [ ] **环境设置与数据加载**
  - [ ] 导入所需库 (dask, loguru, pandas, 可视化库)
  - [ ] 使用 dask 加载 `demand.parquet`, `metadata.parquet`, `weather.parquet`
- [ ] **数据概览与质量检查**
  - [ ] 查看数据形状、列名、数据类型
  - [ ] 检查并统计缺失值
  - [ ] 检查并处理重复值
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

*   **数据加载**: 使用 Dask 加载 `demand`, `metadata`, `weather` Parquet 文件。

## 下一步 ➡️

*   **初步数据探索**:
    *   查看每个数据集的基本信息 (列名, 数据类型, 分区数)。
    *   计算 `demand` 数据集的大致行数 (使用 `len()`)。
    *   检查 `metadata` 和 `weather` 数据集的具体行数 (因为它们较小，可以调用 `.compute()` 或直接使用 `len()` 查看已知的行数)。
    *   使用 `.head()` 查看每个数据集的前几行数据，了解具体内容。

## 已完成 ✅

*   *(暂无)*