# EAP Dashboard (Streamlit)

Hướng dẫn xem và chạy dashboard cho 4 chỉ số chuẩn hoá (normalized) của khu vực Đông Á & Thái Bình Dương (EAP): GDP, CPI, PCE, Population.

## Yêu cầu môi trường
- Python 3.9+ (khuyến nghị 3.10–3.12)
- Các gói: streamlit, plotly, pandas

Cài đặt nhanh (trong virtualenv/venv của bạn):

```bash
pip install streamlit plotly pandas
```

Nếu đang dùng `conda`:
```bash
conda install -c conda-forge streamlit plotly pandas
```

## Cấu trúc dữ liệu
- CSV normalized đặt tại: `final-project/data/east_asia_pacific/`
  - `gdp_eap_processed.csv`
  - `cpi_eap_processed.csv`
  - `pce_eap_processed.csv`
  - `population_eap_processed.csv`

## Chạy dashboard
Từ thư mục `final-project/`:

```bash
streamlit run app.py
```

Sau khi chạy, mở trình duyệt theo URL Streamlit in ra (thường là `http://localhost:8501`).

## Bộ lọc & tuỳ chọn
- Countries (EAP): chọn 1 hoặc nhiều quốc gia.
- Year range: khoảng năm cần hiển thị.
- Indicators: chọn chỉ số gốc (gdp, cpi, pce, pop).
- Transform:
  - Raw (normalized)
  - Per-capita (gdp/pop, pce/pop)
  - YoY % (tăng trưởng năm trước)
  - CAGR % (tăng trưởng kép từ năm bắt đầu)
- Indicator for sparklines: chỉ số dùng để tạo grid mini-line theo quốc gia.

## Các tab trong dashboard
### Overview
- Mục đích: xem nhanh mức độ của từng series tại năm mới nhất trong vùng lọc.
- Cách đọc: mỗi ô KPI là trung bình giá trị của series (ví dụ `GDP (normalized)` hoặc `gdp_per_capita`) trên các quốc gia đã chọn. Giá trị normalized nằm [0–1]; các biến chuyển đổi (YoY%, CAGR%) là phần trăm.
- Mẹo: thay đổi `Transform` trong sidebar để KPI phản ánh raw / per-capita / YoY / CAGR.

### Time Series
- Mục đích: theo dõi xu hướng theo thời gian cho nhiều quốc gia và series.
- Cách đọc: màu thể hiện quốc gia, kiểu nét (line dash) thể hiện series. Di chuột để xem năm và giá trị chi tiết.
- Mẹo: lọc `Indicators` chỉ còn 1–2 series để đồ thị dễ đọc; dùng `Year range` để phóng vào giai đoạn quan tâm.

### Comparison
- Mục đích: so sánh mức độ giữa các quốc gia trong một năm cụ thể.
- Cách đọc: chọn năm bằng slider, cột nhóm theo `series`. So sánh chiều cao cột để nhận diện thứ hạng.
- Mẹo: kết hợp với `Ranking` để xem top/bottom rõ ràng hơn.

### Distribution (Box / Violin)
- Mục đích: xem phân bố, độ trải và outlier của giá trị theo series trong năm chọn.
- Cách đọc: Boxplot hiển thị median, quartiles, outliers; Violin hiển thị mật độ và box bên trong.
- Mẹo: chuyển `chart type` giữa `box` và `violin` để nhấn mạnh phần bạn quan tâm (phân vị hay mật độ).

### Ranking (Top-N)
- Mục đích: liệt kê top N quốc gia có giá trị cao nhất cho một `series` ở năm chọn.
- Cách đọc: chọn `series`, `Top N` và năm. Biểu đồ thanh ngang sắp xếp theo giá trị.
- Mẹo: dùng cho mỗi chuyển đổi khác nhau (per-capita/YoY/CAGR) để thấy top thay đổi ra sao.

### Small multiples (Faceted lines)
- Mục đích: xem song song xu hướng của nhiều series.
- Cách đọc: mỗi ô (facet) là một series; trong ô, các đường màu là quốc gia. So sánh hình dạng đường giữa các facet.
- Mẹo: giảm số quốc gia để đường không quá chồng chéo.

### Scatter (GDP vs PCE, size=Population)
- Mục đích: xem mối liên hệ giữa GDP và PCE, quy mô theo dân số.
- Cách đọc: vị trí theo trục x/y; kích thước bong bóng theo `Population`; có `animation` theo `year` để xem quỹ đạo.
- Mẹo: bật `Transform=per_capita` rồi so sánh scatter ở tab này (tab scatter dùng dữ liệu raw), để nhận ra khác biệt giữa mức độ tổng và per capita.

### Change (Diverging bar giữa 2 năm)
- Mục đích: đo biến động giá trị giữa hai mốc năm.
- Cách đọc: thanh dương là tăng, âm là giảm; chọn `start` và `end` bằng slider. Mỗi màu là một series.
- Mẹo: dùng cho `YoY%` hoặc `CAGR%` để thấy mức tăng trưởng mạnh/ yếu.

### Radar profile
- Mục đích: tổng quan cấu hình một quốc gia ở năm chọn trên nhiều series.
- Cách đọc: mỗi trục là một series; hình đa giác càng rộng → giá trị normalized càng cao.
- Mẹo: chọn series đã chuyển đổi (per-capita) để so sánh công bằng giữa quốc gia đông dân và ít dân.

### Sparklines (lưới mini-line)
- Mục đích: nhìn nhanh xu hướng của một indicator cho nhiều quốc gia.
- Cách đọc: chọn `Indicator for sparklines` trong sidebar; mỗi ô là 1 quốc gia, đường thể hiện giá trị theo thời gian.
- Mẹo: hữu ích để “quét” và phát hiện mẫu hình bất thường.

### Correlation (tương quan)
- Mục đích: đo lường mối liên hệ tuyến tính giữa các series.
- Cách đọc: hệ số Pearson trong [-1, 1]; màu đỏ/ xanh thể hiện chiều liên hệ; số lớn tuyệt đối → liên hệ mạnh.
- Mẹo: dùng với series per-capita để xem liên hệ sau khi chuẩn hoá theo dân số.

### Availability (độ phủ dữ liệu)
- Mục đích: kiểm tra ô dữ liệu trống theo `country` và `year`.
- Cách đọc: heatmap xám; ô sáng → có dữ liệu; ô tối → trống.
- Mẹo: nếu phân tích báo thiếu dữ liệu ở tab khác, kiểm tra nhanh tại đây.

### Download
- Mục đích: tải subset CSV theo bộ lọc hiện tại.
- Cách đọc: nhấn “Tải CSV đã lọc”, dùng file cho phân tích ngoài Streamlit.

## Lỗi thường gặp
- Thiếu thư viện `plotly.express` hoặc `streamlit`: cài đặt bằng lệnh ở phần “Yêu cầu môi trường”.
- Lỗi đường dẫn dữ liệu: kiểm tra các file trong `final-project/data/east_asia_pacific/` còn nguyên tên.
 - Không thấy biểu đồ hoặc giá trị: kiểm tra đã chọn quốc gia/series và `Year range` có chứa dữ liệu.
 - Giá trị `NaN` với `CAGR%`: nếu giá trị đầu kỳ bằng 0 hoặc thiếu, CAGR sẽ không xác định.

## Gợi ý mở rộng
- Thêm chuyển đổi per-capita cho nhiều chỉ số khác.
- Thêm bộ lọc vùng/nhóm quốc gia.
- Xuất hình ảnh biểu đồ (`st.download_button` cho PNG/SVG).
