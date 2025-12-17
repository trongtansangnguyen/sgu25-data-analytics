# Kế Hoạch Nghiên Cứu: Phân Tích Đói Nghèo và Bất Bình Đẳng
## Khu vực Đông Á - Thái Bình Dương (2000-2024)

---

## 1. Mục Tiêu Nghiên Cứu

### Mục tiêu chính
Áp dụng kỹ thuật phân tích dữ liệu và học máy để khám phá các mối quan hệ nhân quả/tương quan trong dữ liệu đói nghèo và bất bình đẳng tại khu vực Đông Á - Thái Bình Dương.

### Mục tiêu chung
Phân tích sự phân bố, xu hướng của đói nghèo và bất bình đẳng trong khu vực EAP, từ đó rút ra insight hỗ trợ việc xây dựng chính sách.

### Mục tiêu cụ thể
1. Thực hiện các bước tiền xử lý dữ liệu Thống kê Đói nghèo và Bất bình đẳng (World Bank)
2. Thực hiện Phân tích Khám phá Dữ liệu (EDA) để hiểu cấu trúc và xu hướng dữ liệu
3. Áp dụng thuật toán Phân cụm (Clustering) để nhóm các quốc gia có đặc điểm đói nghèo tương đồng
4. Áp dụng thuật toán Phân nhóm (Classification) để dự đoán chỉ số liên quan đến đói nghèo
5. Trực quan hóa kết quả một cách hiệu quả

---

## 2. Phạm Vi Nghiên Cứu

- **Khu vực:** Đông Á - Thái Bình Dương (East Asia Pacific)
- **Thời gian:** 2000 - 2024
- **Nguồn dữ liệu:** World Bank's Poverty and Inequality Platform

### Bộ dữ liệu có sẵn
- `gini_index_eap.csv` - Chỉ số Gini (Bất bình đẳng thu nhập)
- `poverty_headcount_eap.csv` - Tỷ lệ nghèo (3 ngưỡng: $2.15, $3.65, $6.85/ngày)
- `gdp.csv` - GDP (Tổng sản phẩm quốc nội)
- `cpi.csv` - CPI (Chỉ số giá tiêu dùng)
- `pop.csv` - Dân số
- `PCE.csv` - Chi tiêu tiêu dùng cá nhân

---

## 3. Câu Hỏi Nghiên Cứu

### Câu hỏi gốc (từ đề tài)

| Loại Phân Tích | Câu Hỏi |
|----------------|---------|
| **Tương quan** | Có mối liên hệ nào giữa mức độ bất bình đẳng (chỉ số Gini) và tỷ lệ nhập học tiểu học ở các quốc gia đang phát triển trong khu vực EAP không? |
| **Phân cụm** | Các quốc gia trong khu vực EAP có thể được phân cụm thành những nhóm nào dựa trên các chỉ số về đói nghèo và tiếp cận dịch vụ cơ bản? |
| **Dự đoán/Phân nhóm** | Các yếu tố kinh tế xã hội (GDP/người, CPI, dân số) ảnh hưởng như thế nào đến tỷ lệ nghèo đa chiều của một quốc gia? |

### Câu hỏi bổ sung (đề xuất)

**4. Phân tích Xu hướng Thời gian**
- Quốc gia nào trong khu vực EAP có tốc độ giảm nghèo nhanh nhất trong giai đoạn 2000-2024? 
- Các yếu tố nào góp phần (GDP, CPI, dân số)?

**5. So sánh Đa ngưỡng Nghèo**
- Có sự khác biệt nào trong ranking các quốc gia khi sử dụng các ngưỡng nghèo khác nhau ($2.15 vs $3.65 vs $6.85/ngày)?
- Quốc gia nào có tỷ lệ nghèo tương đối cao ở ngưỡng trung bình ($3.65) nhưng thấp ở ngưỡng cực nghèo ($2.15)?

**6. Tác động của Khủng hoảng**
- COVID-19 (2020-2021) và các sự kiện kinh tế toàn cầu ảnh hưởng như thế nào đến chỉ số Gini và tỷ lệ nghèo ở khu vực EAP?
- Quốc gia nào phục hồi nhanh nhất sau khủng hoảng?

**7. Paradox Tăng trưởng - Bất bình đẳng**
- Các quốc gia có GDP tăng mạnh có giảm được bất bình đẳng (Gini) không, hay bất bình đẳng vẫn tăng?
- Phân tích các trường hợp điển hình: Trung Quốc, Việt Nam, Indonesia

---

## 4. Phương Pháp Nghiên Cứu

### 4.1 Tiền xử lý Dữ liệu
- **Làm sạch:** Xử lý các giá trị không hợp lệ hoặc không rõ ràng
- **Xử lý Missing Values:** 
  - Gán giá trị trung bình/trung vị
  - Interpolation cho dữ liệu time series
  - Xóa bỏ nếu missing quá nhiều
- **Chuẩn hóa:** Min-Max Scaling hoặc Z-Score Standardization
- **Merge datasets:** Kết hợp dữ liệu từ nhiều nguồn theo Country và Year

### 4.2 Phân tích Khám phá Dữ liệu (EDA)
- Thống kê mô tả (mean, median, std, distribution)
- Phân tích xu hướng theo thời gian (line charts, area charts)
- Phân tích phân bố (histograms, box plots, violin plots)
- Ma trận tương quan (correlation heatmap)
- Phân tích outliers

### 4.3 Kỹ thuật Học Máy

#### Phân cụm (Clustering)
- **Thuật toán:** K-Means Clustering
- **Chọn k tối ưu:** Elbow Method, Silhouette Score
- **Đánh giá:** Silhouette Score, Davies-Bouldin Index
- **Mục tiêu:** Nhóm các quốc gia có đặc điểm đói nghèo tương đồng

#### Phân nhóm (Classification)
- **Thuật toán:** Random Forest Classifier
- **Biến mục tiêu:** Tỷ lệ nghèo cao/thấp (binary classification)
- **Features:** GDP/capita, CPI, Population growth, Gini index
- **Đánh giá:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Mục tiêu:** Dự đoán quốc gia có nguy cơ nghèo đói cao

---

## 5. Timeline và Công Việc

| Giai Đoạn | Công Việc | Thời Gian | Trọng Số |
|-----------|-----------|-----------|----------|
| **Giai đoạn 1: Chuẩn bị** | 1. Review dữ liệu có sẵn<br>2. Xây dựng notebook structure<br>3. Setup environment | 2 ngày | 10% |
| **Giai đoạn 2: Tiền xử lý** | 1. Load và merge datasets<br>2. Xử lý missing values<br>3. Chuẩn hóa dữ liệu<br>4. Feature engineering | 4 ngày | 20% |
| **Giai đoạn 3: EDA** | 1. Thống kê mô tả<br>2. Visualization (time series, distributions)<br>3. Correlation analysis<br>4. Initial insights | 4 ngày | 20% |
| **Giai đoạn 4: ML & Viz** | 1. K-Means Clustering<br>2. Random Forest Classification<br>3. Model evaluation<br>4. Results visualization | 6 ngày | 30% |
| **Giai đoạn 5: Báo cáo** | 1. Viết báo cáo<br>2. Slide thuyết trình<br>3. Code documentation | 4 ngày | 20% |

**Tổng thời gian:** 20 ngày

---

## 6. Deliverables (Đầu ra)

1. **Jupyter Notebook** với code đầy đủ và comments
2. **Báo cáo PDF** trả lời tất cả câu hỏi nghiên cứu
3. **Slide thuyết trình** (PowerPoint/PDF)
4. **Dashboard/App** (optional - nếu có thời gian)

---

## 7. Các Quốc Gia Quan Tâm (EAP)

Dự kiến phân tích các quốc gia chính trong khu vực:
- Trung Quốc
- Indonesia
- Việt Nam
- Philippines
- Thái Lan
- Malaysia
- Campuchia
- Lào
- Myanmar
- Các quốc gia khác (tùy theo dữ liệu có sẵn)

---

## 8. Tools & Libraries

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn
- **Statistical Analysis:** scipy, statsmodels
- **Dashboard (optional):** streamlit, dash

---

## 9. Ghi Chú

- Tập trung vào khu vực Đông Á - Thái Bình Dương
- Khoảng thời gian: 2000-2024 (24 năm)
- Ưu tiên trả lời 7 câu hỏi nghiên cứu
- Đảm bảo visualization rõ ràng và insight có ý nghĩa thực tiễn

---

*Ngày tạo: 17/12/2024*
*Cập nhật lần cuối: 17/12/2024*
