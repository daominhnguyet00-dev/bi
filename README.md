# 📈 Ứng dụng Web Kiểm Định Chiến Lược Đầu Tư Kết Hợp SMA & OBV (Logic OR) trên HOSE

Ứng dụng web tương tác được xây dựng trên nền tảng **Streamlit**, chuyển giao toàn bộ quy trình nghiên cứu định lượng từ Jupyter Notebook sang sản phẩm ứng dụng sẵn sàng triển khai trên **GitHub** và **Streamlit Community Cloud**.

---

## 📌 1. Giới thiệu Dự án

Dự án thực hiện quy trình kiểm định chiến lược giao dịch định lượng hoàn chỉnh trên thị trường chứng khoán Việt Nam (Sở Giao dịch Chứng khoán TP.HCM - HOSE):

1. **Lọc cổ phiếu đa nhân tố (TA Screening) trên tập Train (2020 – 2021):**
   - **Xu hướng (Trend - 30%):** Tỷ lệ số phiên SMA50 > SMA200 và Tương quan giá so với SMA200.
   - **Động lượng (Momentum - 25%):** Tỷ suất sinh lợi 6 tháng (~126 phiên).
   - **Dòng tiền (Volume - 25%):** Giá trị giao dịch trung bình và OBV xác nhận chuẩn hóa (OBV Confirmation).
   - **Rủi ro (Risk - 20%):** Độ biến động năm (Volatility thấp hơn có điểm cao hơn) và Mức sụt giảm tối đa (Max Drawdown gần 0 hơn có điểm cao hơn).
   - Tự động chọn ra **Top 5 cổ phiếu** có TA Score cao nhất (Mặc định: `DIG`, `DPM`, `KBC`, `MSN`, `DCM`).

2. **Chiến lược Tín hiệu Kết hợp SMA + OBV (Logic OR):**
   - **SMA Signal:** Mua khi SMA ngắn hạn cắt lên SMA dài hạn; Bán khi SMA ngắn hạn cắt xuống SMA dài hạn.
   - **OBV Signal:** Mua khi OBV cắt lên đường trung bình OBV-MA; Bán khi OBV cắt xuống OBV-MA.
   - **Logic OR:**
     - **MUA (+1):** Khi SMA Báo Mua **HOẶC** OBV Báo Mua.
     - **BÁN (-1):** Khi SMA Báo Bán **HOẶC** OBV Báo Bán.
     - **Xung đột tín hiệu:** Khi cả Mua và Bán đồng thời xuất hiện trong cùng phiên $\rightarrow$ Giữ tiền mặt (Tín hiệu = 0).

3. **Chống Thiên lệch Nhìn trước (No Look-ahead Bias):**
   - Tín hiệu xác lập tại ngày $t$ chỉ được thực thi khớp lệnh bắt đầu từ phiên giao dịch tiếp theo ($t+1$).
   - Tập dữ liệu ngoài mẫu (Test 2022) hoàn toàn không được sử dụng để lọc cổ phiếu, tối ưu tham số hay tính trọng số danh mục.

4. **Kỹ thuật Khởi tạo Warm-up 349 Phiên:**
   - Để hỗ trợ đường trung bình dài hạn lên tới **SMA350** ngay từ ngày giao dịch đầu tiên của năm 2022, thuật toán lùi đúng 349 phiên giao dịch lịch sử trước `TEST_START` để làm bước đệm tính toán indicator.
   - Lợi nhuận và hiệu quả chỉ được ghi nhận chính xác từ phiên bắt đầu của Test trở đi.

5. **Phân bổ Danh mục & So sánh Benchmark:**
   - Danh mục **Equal Weight (Tỷ trọng đều 1/N)**.
   - Danh mục **MPT (Lý thuyết Danh mục Hiện đại)**: Tối đa hóa Sharpe Ratio trên Train bằng thuật toán SLSQP.
   - So sánh trực tiếp với Benchmark **Mua & Nắm giữ (Buy & Hold)** trong giai đoạn downtrend khốc liệt của năm 2022.

---

## 🗂️ 2. Cấu trúc Thư mục

```text
├── app.py                  # Mã nguồn chính ứng dụng Streamlit
├── requirements.txt        # Danh sách thư viện cần thiết để cài đặt & deploy
├── README.md               # Tài liệu hướng dẫn sử dụng và triển khai
├── HOSE_2020_2023_in.csv   # Dữ liệu giá OHLCV lịch sử sàn HOSE (2020 - 2022)
└── nhom_5.ipynb            # Notebook nghiên cứu gốc
```

---

## 💻 3. Hướng dẫn Cài đặt & Chạy Cục bộ (Localhost)

### Yêu cầu hệ thống
- Python 3.9 đến 3.12.
- Git (tùy chọn để clone repository).

### Các bước thực hiện:

1. **Tải mã nguồn về máy:**
   ```bash
   git clone <URL_REPOSITORY_CỦA_BẠN>
   cd <THƯ_MỤC_DỰ_ÁN>
   ```

2. **Tạo và kích hoạt môi trường ảo (Khuyên dùng):**
   - Trên Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Web:**
   ```bash
   streamlit run app.py
   ```
   Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## 🚀 4. Hướng dẫn Đăng tải lên GitHub & Deploy lên Streamlit Cloud

### Bước 1: Khởi tạo và đẩy mã nguồn lên GitHub

1. Truy cập [GitHub](https://github.com/) và tạo một Repository mới (ví dụ: `hose-sma-obv-backtest`). Để chế độ **Public**.
2. Tại thư mục chứa các file trên máy tính của bạn, mở terminal/powershell và chạy các lệnh:
   ```bash
   git init
   git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
   git commit -m "feat: initial commit for Streamlit app deployment"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-github-cua-ban>/hose-sma-obv-backtest.git
   git push -u origin main
   ```

### Bước 2: Triển khai miễn phí trên Streamlit Community Cloud

1. Đăng nhập vào [Streamlit Community Cloud](https://share.streamlit.io/) bằng tài khoản GitHub.
2. Nhấn vào nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin triển khai:
   - **Repository:** `<tai-khoan-github-cua-ban>/hose-sma-obv-backtest`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Nhấn **"Deploy!"**.
5. Đợi 1–2 phút để hệ thống tự động cài đặt `requirements.txt` và khởi chạy container. Sau khi hoàn tất, bạn sẽ nhận được đường link web trực tuyến (ví dụ: `https://hose-sma-obv.streamlit.app`) để chia sẻ.

---

## 📊 5. Tóm tắt Kết quả Thực nghiệm

| Chỉ tiêu (Test 2022) | Equal Weight Buy & Hold | SMA+OBV (OR) - Equal Weight | SMA+OBV (OR) - MPT |
| :--- | :---: | :---: | :---: |
| **Total Return [%]** | **-50.41%** | **-0.85%** | **+8.08%** |
| **Annual Volatility [%]** | 44.19% | 24.64% | 24.75% |
| **Sharpe Ratio** | -1.38 | **0.09** | **0.44** |
| **Max Drawdown [%]** | **-64.30%** | **-28.08%** | **-21.19%** |

### Nhận xét Định lượng:
- **Khả năng phòng vệ sụt giá (Capital Preservation):** Trong năm 2022 khi thị trường giảm điểm khốc liệt, chiến lược Mua & Nắm giữ chịu thua lỗ nặng (-50.41%), sụt giảm từ đỉnh lên tới -64.30%. Nhờ cơ chế tín hiệu OR, hệ thống đã kịp thời kích hoạt bán và chuyển phần lớn tài sản sang tiền mặt, hạn chế tối đa rủi ro suy giảm tài sản.
- **Tối ưu hóa MPT ngoài mẫu:** Danh mục MPT xác lập từ dữ liệu Train (DIG 26.6%, KBC 22.4%, MSN 23.1%, DCM 27.9%, DPM 0.0%) đạt tỷ suất sinh lợi dương **+8.08%** và Sharpe Ratio vượt trội **0.44** ngay trong năm khủng hoảng.

---

## 🛠️ 6. Công nghệ Sử dụng

- **Giao diện & Ứng dụng:** [Streamlit](https://streamlit.io/)
- **Xử lý Dữ liệu:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Tối ưu hóa Danh mục:** [SciPy Optimize (SLSQP)](https://scipy.org/)
- **Tối ưu Tham số Kỹ thuật:** [Hyperopt (TPE)](https://hyperopt.github.io/hyperopt/)
- **Phân tích Kỹ thuật:** [TA Library](https://technical-analysis-library-in-python.readthedocs.io/)
- **Trực quan hóa Dữ liệu:** [Plotly Graph Objects](https://plotly.com/python/)
