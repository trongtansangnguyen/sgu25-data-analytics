# 📊 East Asia & Pacific Economic Analysis

## 🎯 Project Overview

Dự án phân tích toàn diện về kinh tế khu vực **Đông Á & Thái Bình Dương** (East Asia & Pacific) sử dụng dữ liệu từ World Bank. Dự án thực hiện đầy đủ pipeline Data Science bao gồm: Data Cleaning, Exploratory Data Analysis (EDA), Statistical Analysis, và Machine Learning.

### 📍 Scope
- **Khu vực**: 25 quốc gia East Asia & Pacific
- **Thời gian**: 2000-2025 (26 năm)
- **Dữ liệu**: 4 chỉ số kinh tế chính (GDP, CPI, PCE, Population)
- **Số quốc gia phân tích ML**: 19 (có đủ dữ liệu)

## 📂 Project Structure

```
final-project/
├── README.md                          # Documentation chính
├── data/                              # Dữ liệu gốc từ World Bank
│   ├── gdp.csv                        # GDP data (toàn cầu)
│   ├── cpi.csv                        # CPI data
│   ├── PCE.csv                        # Personal Consumption Expenditure
│   ├── pop.csv                        # Population data
│   └── east_asia_pacific/             # Dữ liệu đã xử lý
│       ├── gdp_eap_processed.csv      # GDP normalized (0-1)
│       ├── cpi_eap_processed.csv      # CPI normalized
│       ├── pce_eap_processed.csv      # PCE normalized
│       └── population_eap_processed.csv # Population normalized
├── notebooks/
│   └── east_asia_pacific_analysis.ipynb  # Main analysis (60 cells)
└── reports/                           # Visualizations
    ├── 01_distribution_analysis.png   # Histograms & Boxplots
    ├── 02_scatterplot_relationships.png # Variable relationships
    ├── 03_correlation_heatmap.png     # Pearson & Spearman
    ├── 04_elbow_method.png            # Optimal K selection
    ├── 05_dendrogram.png              # Hierarchical clustering
    ├── 06_cluster_scatterplots.png    # Cluster visualization
    └── 07_classification_evaluation.png # Confusion Matrix & ROC
```

## 🌏 Countries Analyzed (25 Total)

| Code | Country | Region |
|------|---------|--------|
| AUS | Australia | Oceania |
| CHN | China | East Asia |
| FJI | Fiji | Oceania |
| IDN | Indonesia | Southeast Asia |
| JPN | Japan | East Asia |
| KIR | Kiribati | Oceania |
| KOR | Korea, Rep. (South Korea) | East Asia |
| LAO | Lao PDR | Southeast Asia |
| MYS | Malaysia | Southeast Asia |
| MHL | Marshall Islands | Oceania |
| FSM | Micronesia, Fed. Sts. | Oceania |
| MNG | Mongolia | East Asia |
| MMR | Myanmar | Southeast Asia |
| NRU | Nauru | Oceania |
| PNG | Papua New Guinea | Oceania |
| PHL | Philippines | Southeast Asia |
| WSM | Samoa | Oceania |
| SLB | Solomon Islands | Oceania |
| TWN | Taiwan, China | East Asia |
| THA | Thailand | Southeast Asia |
| TLS | Timor-Leste | Southeast Asia |
| TON | Tonga | Oceania |
| TUV | Tuvalu | Oceania |
| VUT | Vanuatu | Oceania |
| VNM | Viet Nam | Southeast Asia |

---

## 📊 Dataset Description

Dữ liệu được lấy từ [World Bank Data Sources](https://pip.worldbank.org/datasources), bao gồm các chỉ số kinh tế quan trọng:

### 1. **GDP (Gross Domestic Product)** - `gdp.csv`
- **Định nghĩa**: Tổng giá trị hàng hóa và dịch vụ được sản xuất trong nước
- **Mô tả**: Thể hiện quy mô nền kinh tế, là một trong những chỉ số quan trọng nhất để đo lường phát triển kinh tế
- **Nguồn**: World Bank - Economy & Growth indicator
- **Đơn vị**: USD hiện tại (Current US$)

### 2. **CPI (Consumer Price Index)** - `cpi.csv`
- **Định nghĩa**: Chỉ số giá tiêu dùng - đo lường mức thay đổi giá cả của hàng hóa và dịch vụ tiêu dùng
- **Mô tả**: Phản ánh lạm phát, được sử dụng để theo dõi sức mua của đồng tiền và chi phí sinh hoạt
- **Nguồn**: World Bank - Financial Sector & Economy & Growth indicator
- **Chỉ số liên quan**: Inflation, consumer prices (annual %)
- **Đơn vị**: Chỉ số (%) hoặc tỷ lệ thay đổi hàng năm

### 3. **PCE (Personal Consumption Expenditure)** - `PCE.csv`
- **Định nghĩa**: Chi tiêu tiêu dùng cá nhân - tổng giá trị hàng hóa và dịch vụ mua bởi các hộ gia đình
- **Mô tả**: Là thành phần lớn nhất của GDP (thường chiếm 50-70% GDP), thể hiện sức khỏe của nền kinh tế và niềm tin của người tiêu dùng
- **Nguồn**: World Bank - Household Consumption Data & Private Sector
- **Chỉ số liên quan**: Household Consumption, Personal remittances
- **Đơn vị**: USD hiện tại (Current US$) hoặc % GDP

### 4. **Population** - `pop.csv`
- **Định nghĩa**: Dân số tổng cộng
- **Mô tả**: Dùng để tính các chỉ số bình quân đầu người (per capita), giúp so sánh công bằng hơn giữa các quốc gia có quy mô khác nhau
- **Nguồn**: World Bank - Health, Climate Change, Education topics
- **Chỉ số liên quan**: Total population indicator
- **Đơn vị**: Số người

## Data Relationships

```
GDP ─┐
     ├─→ PCE (% GDP)
     └─→ GDP per capita (÷ Population)

CPI ─→ Inflation Rate
       (phản ánh sức mua và chi phí sinh hoạt)

Population ─→ GDP per capita
              PCE per capita
              (Chỉ số bình quân đầu người)
```

---

## 🔬 Analysis Methodology

### PHẦN I: Data Loading & Filtering
- Load 4 CSV files từ World Bank
- Filter 25 quốc gia East Asia & Pacific
- Filter temporal range: 2000-2025
- **Output**: 4 filtered datasets

### PHẦN II: Data Preprocessing
1. **Quality Analysis**
   - Missing value detection (CPI: 73.33%)
   - Outlier detection (IQR method)
   - Invalid value checks

2. **Data Cleaning**
   - Mean imputation cho missing values
   - Remove invalid entries (negatives)
   
3. **Normalization**
   - Min-Max scaling (0-1 range)
   - Preserve data distribution

4. **Validation**
   - Verify 0% missing values
   - Confirm data range [0, 1]
   - Export cleaned datasets

### PHẦN III: Exploratory Data Analysis (EDA)

#### 1. Descriptive Statistics
- Mean, Median, Std Dev
- Skewness & Kurtosis
- Distribution characterization

#### 2. Visual Analysis
- **Histograms**: Distribution shapes
- **Boxplots**: Outlier visualization
- **Scatterplots**: Variable relationships (6 pairs)
- **Correlation Heatmaps**: Pearson & Spearman

#### 3. Insights Generation
- Top 5 countries by GDP/Population
- Trend analysis (2000-2025)
- Economic volatility patterns

### PHẦN IV: Machine Learning

#### A. Clustering Analysis

**1. K-Means Clustering**
- Elbow Method: Tested K=2 to K=10
- Optimal K: 4 (Silhouette Score: 0.674)
- Result: 4 distinct economic groups

**2. Hierarchical Clustering**
- Method: Agglomerative (Ward linkage)
- Clusters: 4 groups (K=4)
- Silhouette Score: 0.674
- Visualization: Dendrogram

**Cluster Interpretation:**
- **Cluster 0** (15 countries): Developing economies
- **Cluster 1** (1 country): China - large population profile
- **Cluster 2** (2 countries): Australia, Japan - developed
- **Cluster 3** (1 country): South Korea - high-tech economy

#### B. Classification Analysis

**Target Variable**: GDP High/Low (binary, median threshold)

**1. Random Forest Classifier**
- Accuracy: **83.33%**
- ROC-AUC: 0.778
- Feature Importance:
  - GDP: 66.56%
  - Population: 25.71%
  - PCE: 7.72%

**2. Logistic Regression**
- Accuracy: 50% (predicts only one class)
- ROC-AUC: 1.000

**Evaluation Metrics:**
- Confusion Matrix
- ROC Curve
- Precision, Recall, F1-Score

---

## 📈 Key Findings

### 1. Economic Growth Trends (2000-2025)
- **GDP Growth**: +14.14%
- **Population Growth**: +30.79%
- GDP per capita: Slight decline due to faster population growth

### 2. Top Economies (Normalized Scale 0-1)
1. **Taiwan**: 1.000 (highest GDP)
2. **Australia**: 0.860
3. **Japan**: 0.510
4. **South Korea**: 0.350
5. **China**: 0.160

### 3. Population Leaders
1. **China**: 1.000 (1.4B people)
2. **Indonesia**: 0.200 (273M)
3. **Japan**: 0.090 (126M)
4. **Philippines**: 0.080 (110M)
5. **Vietnam**: 0.070 (98M)

### 4. Correlation Insights
- **GDP ↔ PCE**: Strong positive (0.85+)
- **GDP ↔ Population**: Moderate positive
- **CPI ↔ others**: Weak/moderate correlation
- **Spearman > Pearson**: Non-linear relationships exist

### 5. Clustering Insights
- Clear separation between developed/developing economies
- China and South Korea are economic outliers
- Silhouette Score 0.674: Good cluster quality
- Both K-Means & Hierarchical agree on groupings

---
