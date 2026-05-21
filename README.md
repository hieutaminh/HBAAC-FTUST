# HBAAC-FTUST

Script dự báo số lượng bán hàng theo ngày cho từng SKU, phục vụ cuộc thi **HBAAC 2026**.  
Horizon dự báo: **F1 (06/09/2025) → F56 (31/10/2025)**, chia làm hai giai đoạn:
- **Validation** (F1–F28): dùng để đánh giá public leaderboard.
- **Evaluation** (F29–F56): dùng để đánh giá private leaderboard.

---

## Cấu trúc thư mục cần chuẩn bị

```
HBAAC2026/
├── train.csv              # Dữ liệu lịch sử bán hàng (bắt buộc)
└── forecast.py            # Script dự báo
```


---

## Chạy trên Kaggle Notebook

```python
# 1. Tạo Notebook mới → chọn Dataset chứa train.csv & sample_submission.csv
# Input data thường nằm tại: /kaggle/input/<tên-dataset>/

# 2. Upload forecast.py vào phần "Add-ons > Upload files" hoặc paste code trực tiếp

# 3. Sửa DATA_DIR
DATA_DIR = "/kaggle/input/<tên-dataset>"   # sửa <tên-dataset> cho đúng

# Output sẽ được lưu tại /kaggle/working/
# Sửa output_path trong build_submission:
#   build_submission(panel, "/kaggle/working/submission_v1.csv")

# 4. Chạy toàn bộ cell
```

**Lưu ý Kaggle:**
- Không có quyền ghi vào `/kaggle/input/` → chỉ được ghi vào `/kaggle/working/`.
- Sau khi chạy xong, file CSV trong `/kaggle/working/` sẽ hiển thị ở tab **Output** để tải về.

---

## Chạy trên Local

### Yêu cầu

| Thư viện  | Phiên bản khuyến nghị |
|-----------|----------------------|
| Python    | ≥ 3.8                |
| pandas    | ≥ 1.3                |
| numpy     | ≥ 1.21               |

### Cài đặt

```bash
pip install pandas numpy
```

### Cấu hình

Mở `forecast.py`, sửa biến `DATA_DIR` ở đầu file:

```python
DATA_DIR = "/đường/dẫn/tới/HBAAC2026"   # ← sửa thành đường dẫn thực tế
```

### Chạy

```bash
python forecast.py
```

### Output

File `submission_v1.csv` sẽ được tạo trong thư mục `DATA_DIR`.

---

## Giải thích

### 1. `load_data` — Đọc dữ liệu

Đọc `train.csv`, parse cột `Date` thành kiểu datetime.

---

### 2. `make_daily_panel` — Tạo bảng dữ liệu ngày × SKU

Gom nhóm dữ liệu theo **ngày + mã SKU**, tính tổng `Quantity`.  
Các ngày không có giao dịch được điền `0` (cross-join đầy đủ).  
Số lượng âm (hàng trả) được clip về `0` — ngày trả hàng = ngày không có doanh số.

---

### 3. `sku_features` — Trích xuất đặc trưng mỗi SKU

Tính các chỉ số thống kê cho từng SKU từ lịch sử đến thời điểm `cutoff`:

| Đặc trưng | Ý nghĩa |
|-----------|---------|
| `n_sale_days` | Số ngày thực sự có bán hàng |
| `sale_freq` | Tỷ lệ ngày có bán / tổng số ngày lịch sử |
| `mean_pos_qty` | Số lượng trung bình trong các ngày có bán |
| `recent_blend` | Trung bình có trọng số theo độ gần đây (7/14/28/56/90 ngày) |
| `dow_freq_0..6` | Tần suất bán theo từng thứ trong tuần (0=Thứ 2) |
| `dow_mean_0..6` | Số lượng trung bình bán theo từng thứ |

---

### 4. `predict` — Dự báo

Với mỗi SKU, dự báo từng ngày trong horizon dựa trên:

1. **Lọc SKU thưa thớt**: nếu số ngày có bán ≤ 3 → dự báo = 0.
2. **Ước tính theo ngày trong tuần**: `p_sale(thứ) × qty_trung_bình(thứ)`.
3. **Pha trộn recency**: kết hợp với `recent_blend` (trọng số `alpha = 0.5` cho SKU active, `0.2` cho SKU thưa).
4. **Shrinkage**: SKU có ít lịch sử bán được co lại về 0 để tránh over-estimate.

---

### 5. `backtest` — Đánh giá hồi (kiểm tra độ chính xác)

Giả lập dự báo: dùng dữ liệu trước ngày `cutoff` để dự báo 28 ngày tiếp theo, rồi so sánh với thực tế.

**Output in ra màn hình:**

```
Backtest: train up to YYYY-MM-DD, validate YYYY-MM-DD – YYYY-MM-DD
  Overall MAE : x.xxxx        ← Sai số tuyệt đối trung bình (thấp = tốt)
  Overall RMSE: x.xxxx        ← Sai số bình phương trung bình (phạt nặng outlier)
  Zero-rate in actuals: xx.x% ← % ngày có qty = 0 trong tập validate
  MAE by sparsity tier:
    sparse(<=3)   : x.xxxx    ← SKU rất ít bán (≤3 ngày)
    low(4-30)     : x.xxxx    ← SKU ít bán
    medium(31-100): x.xxxx    ← SKU bán vừa
    active(>100)  : x.xxxx    ← SKU bán thường xuyên
```

> **Đọc kết quả:** MAE tier `active` thường quan trọng nhất vì đóng góp nhiều vào tổng sai số. `zero_pct` cao (> 80%) nghĩa là dữ liệu rất thưa — mô hình cần ưu tiên không over-predict.

---

### 6. `build_submission` — Tạo file nộp bài

Dự báo full 56 ngày (28 validation + 28 evaluation), ghép vào đúng format của `sample_submission.csv`, làm tròn 4 chữ số thập phân, lưu ra `submission_v1.csv`.

**Output in ra màn hình:**

```
Submission saved: .../submission_v1.csv  (NNN rows)
```

Theo sau là 5 dòng preview để kiểm tra format trước khi nộp.

---

## Tóm tắt luồng chạy

```
train.csv
    └─► load_data()
            └─► make_daily_panel()
                    ├─► backtest()  ──► in MAE/RMSE ra màn hình
                    └─► build_submission()  ──► submission_v1.csv
```
