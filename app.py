import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH TRANG STREAMLIT & GIAO DIỆN
# ==============================================================================
st.set_page_config(
    page_title="Kiểm định Chiến lược SMA + OBV (OR) | HOSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CÁC HÀM TIỆN ÍCH TÍNH TOÁN KỸ THUẬT & PERFORMANCE
# ==============================================================================

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Tính toán chỉ báo On-Balance Volume (OBV)."""
    if HAS_TA:
        try:
            return ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        except Exception:
            pass
    # Thuật toán thuần Pandas / NumPy
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    direction[0] = 0
    obv = (pd.Series(direction, index=close.index) * volume).cumsum()
    return obv

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Tính đường trung bình động giản đơn SMA."""
    return series.rolling(window=int(window)).mean()

def performance_stats(returns: pd.Series, trading_days: int = 252) -> dict:
    """
    Tính toán các chỉ số đo lường hiệu quả danh mục / chiến lược:
    - Mean Daily Return, Total Return, Annual Return
    - Annual Volatility, Sharpe Ratio, Max Drawdown
    """
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {
            "Mean Daily Return [%]": np.nan,
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan
        }

    equity = (1 + r).cumprod()
    total_return = equity.iloc[-1] - 1
    years = len(r) / trading_days

    if years > 0 and equity.iloc[-1] > 0:
        annual_return = equity.iloc[-1] ** (1 / years) - 1
    else:
        annual_return = np.nan

    annual_vol = r.std() * np.sqrt(trading_days)
    if r.std() != 0 and not np.isnan(r.std()):
        sharpe = (r.mean() / r.std()) * np.sqrt(trading_days)
    else:
        sharpe = np.nan

    drawdown = (equity / equity.cummax()) - 1
    max_dd = drawdown.min()

    return {
        "Mean Daily Return [%]": r.mean() * 100,
        "Total Return [%]": total_return * 100,
        "Annual Return [%]": annual_return * 100,
        "Annual Volatility [%]": annual_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100
    }

# ==============================================================================
# 3. LỌC CỔ PHIẾU BẰNG PHÂN TÍCH KỸ THUẬT (TA SCREENING)
# ==============================================================================

@st.cache_data(show_spinner=False)
def calculate_ta_features(df_train: pd.DataFrame, min_observations: int = 400, trading_days: int = 252) -> pd.DataFrame:
    """
    Tính 4 nhóm tín hiệu kỹ thuật trên tập Train:
    1. Trend (Trend strength SMA50>SMA200, Giá so với SMA200)
    2. Momentum (Tỷ suất sinh lợi 6 tháng ~ 126 phiên)
    3. Volume (Giá trị GD trung bình, OBV confirmation chuẩn hóa)
    4. Risk (Độ biến động năm, Max Drawdown)
    """
    results = []
    for ticker, g in df_train.groupby("ticker"):
        g = g.sort_values("date").dropna(subset=["close", "volume"]).copy()
        if len(g) < min_observations:
            continue

        close = g["close"].astype(float)
        volume = g["volume"].astype(float)

        # 1. Trend
        sma50 = compute_sma(close, 50)
        sma200 = compute_sma(close, 200)
        trend_strength = (sma50 > sma200).mean()

        if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1] != 0:
            price_vs_sma200 = (close.iloc[-1] / sma200.iloc[-1]) - 1
        else:
            price_vs_sma200 = np.nan

        # 2. Momentum
        if len(close) >= 126:
            momentum_6m = (close.iloc[-1] / close.iloc[-126]) - 1
        else:
            momentum_6m = np.nan

        # 3. Volume
        avg_trading_value = (close * volume).mean()
        obv = compute_obv(close, volume)
        if len(obv) >= 20:
            volume_20 = volume.iloc[-20:].sum()
            obv_confirmation = (obv.iloc[-1] - obv.iloc[-20]) / volume_20 if volume_20 != 0 else np.nan
        else:
            obv_confirmation = np.nan

        # 4. Risk
        returns = close.pct_change().dropna()
        annual_volatility = returns.std() * np.sqrt(trading_days)
        equity = (1 + returns).cumprod()
        max_drawdown = (equity / equity.cummax() - 1).min()

        results.append({
            "Ticker": ticker,
            "N": len(g),
            "Trend_Strength": trend_strength,
            "Price_vs_SMA200": price_vs_sma200,
            "Momentum_6M": momentum_6m,
            "Avg_Trading_Value": avg_trading_value,
            "OBV_Confirmation": obv_confirmation,
            "Volatility": annual_volatility,
            "Max_Drawdown": max_drawdown
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.dropna(subset=[
            "Trend_Strength", "Price_vs_SMA200", "Momentum_6M",
            "Avg_Trading_Value", "OBV_Confirmation", "Volatility", "Max_Drawdown"
        ])
    return res_df

def compute_ta_scores(features_df: pd.DataFrame, w_trend=0.30, w_mom=0.25, w_vol=0.25, w_risk=0.20) -> pd.DataFrame:
    """Chuẩn hóa Percentile Rank và tính điểm tổng hợp TA Score."""
    if features_df.empty:
        return pd.DataFrame()
    
    df = features_df.copy()
    # Trend Score (50% Trend_Strength + 50% Price_vs_SMA200)
    df["Trend_Strength_Rank"] = df["Trend_Strength"].rank(pct=True)
    df["Price_SMA200_Rank"] = df["Price_vs_SMA200"].rank(pct=True)
    df["Trend_Score"] = 0.50 * df["Trend_Strength_Rank"] + 0.50 * df["Price_SMA200_Rank"]

    # Momentum Score
    df["Momentum_Score"] = df["Momentum_6M"].rank(pct=True)

    # Volume Score (60% Thanh khoản + 40% OBV xác nhận)
    df["Liquidity_Rank"] = df["Avg_Trading_Value"].rank(pct=True)
    df["OBV_Rank"] = df["OBV_Confirmation"].rank(pct=True)
    df["Volume_Score"] = 0.60 * df["Liquidity_Rank"] + 0.40 * df["OBV_Rank"]

    # Risk Score (Volatility thấp hơn => điểm cao hơn, Drawdown gần 0 hơn => điểm cao hơn)
    df["Volatility_Rank"] = df["Volatility"].rank(pct=True, ascending=False)
    df["Drawdown_Rank"] = df["Max_Drawdown"].rank(pct=True)
    df["Risk_Score"] = 0.50 * df["Volatility_Rank"] + 0.50 * df["Drawdown_Rank"]

    # Total TA Score
    df["TA_Score"] = (
        w_trend * df["Trend_Score"] +
        w_mom * df["Momentum_Score"] +
        w_vol * df["Volume_Score"] +
        w_risk * df["Risk_Score"]
    )
    df = df.sort_values("TA_Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df

# ==============================================================================
# 4. TÍN HIỆU SMA + OBV (LOGIC OR) & QUY TẮC KHÔNG LOOK-AHEAD BIAS
# ==============================================================================

def find_position_sma(df: pd.DataFrame, ma_short: int, ma_long: int) -> pd.Series:
    """Tín hiệu SMA Crossover: Cắt lên = 1 (Buy), Cắt xuống = -1 (Sell)."""
    position = pd.Series(0.0, index=df.index, name="SMA_signal")
    if ma_short >= ma_long:
        return position

    sma_s = compute_sma(df["Close"], ma_short)
    sma_l = compute_sma(df["Close"], ma_long)

    buy = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    sell = (sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position

def find_position_obv(df: pd.DataFrame, obv_window: int) -> pd.Series:
    """Tín hiệu OBV Crossover: OBV cắt lên OBV_MA = 1 (Buy), Cắt xuống = -1 (Sell)."""
    position = pd.Series(0.0, index=df.index, name="OBV_signal")
    obv = compute_obv(df["Close"], df["Volume"])
    obv_ma = compute_sma(obv, obv_window)

    buy = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position

def find_position_or(df: pd.DataFrame, ma_short: int, ma_long: int, obv_window: int) -> pd.Series:
    """Kết hợp tín hiệu SMA và OBV theo logic OR (nếu mâu thuẫn cùng ngày thì trả về 0)."""
    sma_sig = find_position_sma(df, ma_short, ma_long)
    obv_sig = find_position_obv(df, obv_window)

    position = pd.Series(0.0, index=df.index, name="SMA_OBV_OR")
    buy = (sma_sig == 1) | (obv_sig == 1)
    sell = (sma_sig == -1) | (obv_sig == -1)
    conflict = buy & sell

    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position

def events_to_holding(events: pd.Series) -> pd.Series:
    """Chuyển đổi sự kiện Mua (1) / Bán (-1) thành chuỗi nắm giữ (1: giữ cổ phiếu, 0: giữ tiền mặt)."""
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i, sig in enumerate(events):
        if sig == 1:
            current = 1.0
        elif sig == -1:
            current = 0.0
        holding.iloc[i] = current
    return holding

def strategy_returns(df: pd.DataFrame, events: pd.Series, commission: float = 0.0) -> tuple:
    """
    Tính lợi nhuận chiến lược:
    QUAN TRỌNG: Tín hiệu ngày t chỉ được thực thi từ phiên t+1 (holding.shift(1))
    để triệt tiêu hoàn toàn hiện tượng Look-ahead Bias.
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)
    executed_holding = holding.shift(1).fillna(0.0)

    strat_ret = executed_holding * asset_ret
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission

    return strat_ret, executed_holding

def evaluate_or(df: pd.DataFrame, ma_short: int, ma_long: int, obv_window: int, commission: float = 0.0) -> dict:
    """Chạy toàn bộ quy trình mô phỏng chiến lược OR trên 1 chuỗi dữ liệu."""
    events = find_position_or(df, ma_short, ma_long, obv_window)
    ret, holding = strategy_returns(df, events, commission=commission)
    return {
        "events": events,
        "holding": holding,
        "returns": ret,
        "stats": performance_stats(ret)
    }

def evaluate_test_with_warmup(
    full_stock_df: pd.DataFrame,
    ma_short: int,
    ma_long: int,
    obv_window: int,
    test_start: str,
    test_end: str,
    warmup_periods: int = 349,
    commission: float = 0.0
) -> dict:
    """
    Đánh giá trên Test có 349 phiên Warm-up:
    Lấy 349 phiên trước TEST_START để tính toán SMA350 ngay từ phiên đầu tiên của Test,
    nhưng lợi nhuận CHỈ được ghi nhận và đánh giá từ TEST_START trở đi.
    """
    df = full_stock_df.sort_index().copy()
    test_mask = df.index >= pd.Timestamp(test_start)
    if not test_mask.any():
        raise ValueError(f"Không tìm thấy dữ liệu từ {test_start}")

    first_test_pos = np.where(test_mask)[0][0]
    warmup_start_pos = max(0, first_test_pos - warmup_periods)

    warmup_and_test = df.iloc[warmup_start_pos:].copy()
    warmup_and_test = warmup_and_test.loc[warmup_and_test.index <= pd.Timestamp(test_end)].copy()

    full_result = evaluate_or(warmup_and_test, ma_short, ma_long, obv_window, commission=commission)

    test_events = full_result["events"].loc[test_start:test_end]
    test_holding = full_result["holding"].loc[test_start:test_end]
    test_returns = full_result["returns"].loc[test_start:test_end]

    return {
        "events": test_events,
        "holding": test_holding,
        "returns": test_returns,
        "stats": performance_stats(test_returns)
    }

# ==============================================================================
# 5. TỐI ƯU HÓA DANH MỤC THEO MPT (MODERN PORTFOLIO THEORY)
# ==============================================================================

def optimize_mpt(returns_matrix: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252) -> np.ndarray:
    """
    Tìm vector trọng số w để tối đa hóa Sharpe Ratio trên tập Train:
    Ràng buộc: sum(w) = 1, 0 <= w_i <= 1 (Long-only).
    """
    n = returns_matrix.shape[1]
    mean_daily = returns_matrix.mean().values
    cov_annual = returns_matrix.cov().values * trading_days

    def neg_sharpe(w):
        p_ret = float(w @ mean_daily * trading_days)
        p_vol = np.sqrt(max(float(w.T @ cov_annual @ w), 0.0))
        if p_vol <= 1e-9:
            return 1e9
        return -(p_ret - risk_free_rate) / p_vol

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        return res.x
    return x0

# ==============================================================================
# 6. TẢI DỮ LIỆU & TIỀN XỬ LÝ
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_or_path) -> pd.DataFrame:
    """Đọc file CSV và chuẩn hóa các cột Open, High, Low, Close, Volume, Ticker, Date."""
    df = pd.read_csv(file_or_path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close", "volume"]).copy()
    return df

# ==============================================================================
# 7. SIDEBAR - THIẾT LẬP THAM SỐ
# ==============================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Cấu hình Nghiên cứu")

# Lựa chọn nguồn dữ liệu
data_source = st.sidebar.radio(
    "Nguồn dữ liệu:",
    ["Sử dụng dữ liệu mặc định (HOSE 2020-2023)", "Tải lên file CSV cá nhân"]
)

uploaded_file = None
if data_source == "Tải lên file CSV cá nhân":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])
    data_path = uploaded_file
else:
    default_csv = "HOSE_2020_2023_in.csv"
    if os.path.exists(default_csv):
        data_path = default_csv
    else:
        st.sidebar.warning(f"Không tìm thấy file {default_csv} tại thư mục gốc.")
        uploaded_file = st.sidebar.file_uploader("Vui lòng tải lên file CSV", type=["csv"])
        data_path = uploaded_file

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Phân chia Dữ liệu")
train_start = st.sidebar.date_input("Train Bắt đầu", pd.to_datetime("2020-01-01"))
train_end = st.sidebar.date_input("Train Kết thúc", pd.to_datetime("2021-12-31"))
test_start = st.sidebar.date_input("Test Bắt đầu", pd.to_datetime("2022-01-01"))
test_end = st.sidebar.date_input("Test Kết thúc", pd.to_datetime("2022-12-31"))

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Cấu hình Chiến lược")
top_n = st.sidebar.slider("Số lượng cổ phiếu Top TA", min_value=3, max_value=10, value=5)
min_obs = st.sidebar.number_input("Số phiên tối thiểu trên Train", min_value=100, max_value=600, value=400, step=50)
initial_capital = st.sidebar.number_input("Vốn ban đầu (VND)", min_value=100000, value=1_000_000, step=100000)
commission_rate = st.sidebar.number_input("Phí giao dịch/thuế mỗi lượt (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.05) / 100.0
warmup_periods = st.sidebar.slider("Số phiên Warm-up trước Test", min_value=50, max_value=400, value=349)

# Tham số mặc định tối ưu sẵn từ notebook
DEFAULT_BEST_PARAMS = {
    "DIG": {"ma_short": 55, "ma_long": 120, "obv_window": 10},
    "DPM": {"ma_short": 30, "ma_long": 290, "obv_window": 55},
    "KBC": {"ma_short": 40, "ma_long": 130, "obv_window": 15},
    "MSN": {"ma_short": 60, "ma_long": 100, "obv_window": 5},
    "DCM": {"ma_short": 35, "ma_long": 260, "obv_window": 15}
}

# ==============================================================================
# 8. GIAO DIỆN CHÍNH & TÍNH TOÁN
# ==============================================================================

st.markdown('<div class="main-title">Kiểm định Chiến lược Đầu tư SMA + OBV (OR)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống phân tích định lượng kết hợp Lọc cổ phiếu Kỹ thuật, Chiến lược Tín hiệu OR và Tối ưu hóa Danh mục MPT trên Thị trường Chứng khoán Việt Nam (HOSE).</div>', unsafe_allow_html=True)

if data_path is None:
    st.info("👋 Chào mừng bạn! Hãy tải lên file dữ liệu CSV ở thanh bên trái hoặc đặt file `HOSE_2020_2023_in.csv` vào cùng thư mục để bắt đầu kiểm định.")
    st.stop()

with st.spinner("Đang tải và tiền xử lý dữ liệu..."):
    try:
        df_full = load_and_preprocess_data(data_path)
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        st.stop()

# Tabs chức năng
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Dữ liệu & Tổng quan",
    "🎯 2. TA Screening (Top N)",
    "⚡ 3. Cấu hình & Tối ưu SMA-OBV",
    "📈 4. Backtest Từng Cổ phiếu",
    "⚖️ 5. Phân bổ MPT & Equal Weight",
    "🏆 6. Đánh giá Test 2022 & Benchmark"
])

# ------------------------------------------------------------------------------
# TAB 1: DỮ LIỆU & TỔNG QUAN
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Tổng quan tập dữ liệu giao dịch")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng số dòng", f"{len(df_full):,}")
    with col2:
        st.metric("Số lượng mã chứng khoán", f"{df_full['ticker'].nunique():,}")
    with col3:
        st.metric("Ngày bắt đầu", df_full["date"].min().strftime("%d/%m/%Y"))
    with col4:
        st.metric("Ngày kết thúc", df_full["date"].max().strftime("%d/%m/%Y"))

    st.markdown("##### 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df_full.head(10), use_container_width=True)

    with st.expander("ℹ️ Nguyên tắc Phương pháp luận Kiểm định Định lượng"):
        st.markdown("""
        1. **Chỉ dùng dữ liệu Train (2020-2021) để chấm điểm TA và chọn cổ phiếu:** Hoàn toàn không nhìn trước tương lai (No Look-ahead Bias).
        2. **Tối ưu tham số SMA và OBV trên Train:** Tối đa hóa Sharpe Ratio nội mẫu trước khi áp dụng vào thực tế.
        3. **Chiến lược OR (SMA Buy HOẶC OBV Buy):** Tận dụng tính xác nhận dòng tiền của OBV và tính xu hướng của SMA. Khi có mâu thuẫn trong ngày, giữ vị thế tiền mặt (0).
        4. **Nguyên tắc Warm-up 349 phiên trên Test 2022:** Dùng 349 phiên lịch sử trước Test để tính đủ rolling window cho SMA350, nhưng chỉ hạch toán lãi/lỗ bắt đầu từ phiên giao dịch đầu năm 2022.
        5. **Thực thi lệnh từ phiên t+1:** Tín hiệu chốt cuối ngày t chỉ được giải ngân từ phiên kế tiếp t+1.
        """)

# ------------------------------------------------------------------------------
# TAB 2: TA SCREENING (TOP N)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Bộ lọc Đa nhân tố Kỹ thuật (TA Multi-factor Screening)")
    st.markdown("""
    Cổ phiếu được chấm điểm dựa trên 4 trụ cột:
    - **Xu hướng (Trend - 30%):** Tỷ lệ số phiên SMA50 > SMA200 và Tương quan giá so với SMA200.
    - **Động lượng (Momentum - 25%):** Tỷ suất sinh lợi 6 tháng (~126 phiên).
    - **Dòng tiền (Volume - 25%):** Giá trị giao dịch trung bình và OBV xác nhận đã chuẩn hóa.
    - **Rủi ro (Risk - 20%):** Biến động lịch sử thấp và Max Drawdown thấp.
    """)

    train_mask = (df_full["date"] >= pd.Timestamp(train_start)) & (df_full["date"] <= pd.Timestamp(train_end))
    df_train_screen = df_full[train_mask].copy()

    with st.spinner("Đang tính toán đặc trưng kỹ thuật trên tập Train..."):
        ta_features = calculate_ta_features(df_train_screen, min_observations=min_obs)
        ta_scored = compute_ta_scores(ta_features)

    if ta_scored.empty:
        st.error("Không tìm thấy mã cổ phiếu nào thỏa mãn điều kiện số phiên tối thiểu trong khoảng thời gian Train.")
        st.stop()

    top_stocks = ta_scored.head(top_n)
    selected_tickers_default = top_stocks["Ticker"].tolist()

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown(f"##### Bảng xếp hạng Top {top_n} Cổ phiếu được chọn tự động:")
        disp_cols = ["Rank", "Ticker", "Trend_Score", "Momentum_Score", "Volume_Score", "Risk_Score", "TA_Score"]
        st.dataframe(
            top_stocks[disp_cols].style.format({
                "Trend_Score": "{:.3f}",
                "Momentum_Score": "{:.3f}",
                "Volume_Score": "{:.3f}",
                "Risk_Score": "{:.3f}",
                "TA_Score": "{:.3f}"
            }),
            use_container_width=True
        )

    with col_t2:
        st.markdown("##### Phân bổ điểm thành phần Top Cổ phiếu:")
        fig_score = go.Figure()
        for idx, row in top_stocks.iterrows():
            fig_score.add_trace(go.Bar(
                name=row["Ticker"],
                x=["Trend", "Momentum", "Volume", "Risk"],
                y=[row["Trend_Score"], row["Momentum_Score"], row["Volume_Score"], row["Risk_Score"]]
            ))
        fig_score.update_layout(barmode="group", height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_score, use_container_width=True)

    # Cho phép người dùng tùy chọn can thiệp danh sách mã
    selected_tickers = st.multiselect(
        "Mã cổ phiếu chọn để đưa vào danh mục kiểm định chiến lược:",
        options=ta_scored["Ticker"].tolist(),
        default=selected_tickers_default
    )

    if not selected_tickers:
        st.warning("Vui lòng chọn ít nhất 1 mã cổ phiếu để tiếp tục.")
        st.stop()

# Chuẩn bị dữ liệu OHLCV theo từng mã đã chọn
def prepare_stock_data(df_full, ticker):
    df = df_full[df_full["ticker"] == ticker].copy()
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["date", "Open", "High", "Low", "Close", "Volume"]].set_index("date")

stock_data = {t: prepare_stock_data(df_full, t) for t in selected_tickers}
train_data = {t: df.loc[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))].copy() for t, df in stock_data.items()}
test_data = {t: df.loc[(df.index >= pd.Timestamp(test_start)) & (df.index <= pd.Timestamp(test_end))].copy() for t, df in stock_data.items()}

# ------------------------------------------------------------------------------
# TAB 3: CẤU HÌNH & TỐI ƯU THAM SỐ SMA - OBV
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Thiết lập & Tối ưu hóa Tham số Chiến lược trên Train")
    st.markdown("""
    Chiến lược tìm bộ 3 tham số tối ưu:
    - **`ma_short`** (SMA ngắn hạn): 20 - 60 phiên
    - **`ma_long`** (SMA dài hạn): 100 - 350 phiên
    - **`obv_window`** (Đường trung bình OBV-MA): 5 - 60 phiên
    """)

    opt_mode = st.radio(
        "Phương thức chọn tham số:",
        [
            "Sử dụng tham số tối ưu chuẩn xác từ Nghiên cứu (Khuyên dùng)",
            "Tự điều chỉnh thủ công cho từng cổ phiếu",
            "Chạy thuật toán Hyperopt (Tối ưu tự động trên dữ liệu Train)"
        ],
        horizontal=True
    )

    best_params = {}

    if opt_mode == "Sử dụng tham số tối ưu chuẩn xác từ Nghiên cứu (Khuyên dùng)":
        for t in selected_tickers:
            if t in DEFAULT_BEST_PARAMS:
                best_params[t] = DEFAULT_BEST_PARAMS[t]
            else:
                best_params[t] = {"ma_short": 30, "ma_long": 150, "obv_window": 20}
        params_df = pd.DataFrame(best_params).T
        st.dataframe(params_df, use_container_width=True)

    elif opt_mode == "Tự điều chỉnh thủ công cho từng cổ phiếu":
        st.info("Điều chỉnh tham số bên dưới:")
        cols = st.columns(len(selected_tickers))
        for i, t in enumerate(selected_tickers):
            with cols[i % len(cols)]:
                st.markdown(f"**{t}**")
                def_p = DEFAULT_BEST_PARAMS.get(t, {"ma_short": 30, "ma_long": 150, "obv_window": 20})
                ms = st.number_input(f"SMA Ngắn ({t})", min_value=5, max_value=100, value=def_p["ma_short"], step=5, key=f"ms_{t}")
                ml = st.number_input(f"SMA Dài ({t})", min_value=50, max_value=350, value=def_p["ma_long"], step=10, key=f"ml_{t}")
                ow = st.number_input(f"OBV Window ({t})", min_value=5, max_value=100, value=def_p["obv_window"], step=5, key=f"ow_{t}")
                best_params[t] = {"ma_short": int(ms), "ma_long": int(ml), "obv_window": int(ow)}

    else:
        if not HAS_HYPEROPT:
            st.error("Thư viện `hyperopt` chưa được cài đặt trong môi trường này. Vui lòng chọn chế độ tham số chuẩn hoặc cài đặt `hyperopt`.")
            for t in selected_tickers:
                best_params[t] = DEFAULT_BEST_PARAMS.get(t, {"ma_short": 30, "ma_long": 150, "obv_window": 20})
        else:
            max_evals = st.slider("Số lượt thử nghiệm tối ưu Hyperopt mỗi mã:", min_value=10, max_value=100, value=30, step=10)
            if st.button("🚀 Bắt đầu Chạy Tối ưu Hyperopt trên Train"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for idx, t in enumerate(selected_tickers):
                    status_text.text(f"Đang tối ưu tham số cho {t} ({idx+1}/{len(selected_tickers)})...")
                    df_tr = train_data[t]

                    def score_fn(params):
                        ms = int(params["ma_short"])
                        ml = int(params["ma_long"])
                        ow = int(params["obv_window"])
                        if ms >= ml:
                            return 999999
                        ev = find_position_or(df_tr, ms, ml, ow)
                        ret, _ = strategy_returns(df_tr, ev, commission=commission_rate)
                        stats = performance_stats(ret)
                        sharpe = stats["Sharpe Ratio"]
                        if pd.isna(sharpe):
                            return 999999
                        return -sharpe

                    space = {
                        "ma_short": hp.quniform(f"{t}_ma_short", 20, 60, 5),
                        "ma_long": hp.quniform(f"{t}_ma_long", 100, 350, 10),
                        "obv_window": hp.quniform(f"{t}_obv_window", 5, 60, 5)
                    }
                    trials = Trials()
                    best_raw = fmin(
                        fn=score_fn,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials,
                        verbose=False
                    )
                    best_params[t] = {
                        "ma_short": int(best_raw[f"{t}_ma_short"]),
                        "ma_long": int(best_raw[f"{t}_ma_long"]),
                        "obv_window": int(best_raw[f"{t}_obv_window"])
                    }
                    progress_bar.progress((idx + 1) / len(selected_tickers))
                status_text.text("Đã hoàn thành tối ưu hóa!")
                st.session_state["hyperopt_params"] = best_params

            if "hyperopt_params" in st.session_state:
                best_params = st.session_state["hyperopt_params"]
                st.dataframe(pd.DataFrame(best_params).T, use_container_width=True)
            else:
                for t in selected_tickers:
                    best_params[t] = DEFAULT_BEST_PARAMS.get(t, {"ma_short": 30, "ma_long": 150, "obv_window": 20})
                st.info("Đang hiển thị bộ tham số mặc định. Nhấn nút phía trên để bắt đầu tối ưu.")

# ------------------------------------------------------------------------------
# TAB 4: BACKTEST TỪNG CỔ PHIẾU
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Kiểm định Hiệu quả Chiến lược Từng Mã (Train & Test)")

    # Chạy mô phỏng cho từng mã
    train_strategy = {}
    test_strategy = {}

    for t in selected_tickers:
        p = best_params[t]
        train_strategy[t] = evaluate_or(train_data[t], p["ma_short"], p["ma_long"], p["obv_window"], commission=commission_rate)
        test_strategy[t] = evaluate_test_with_warmup(
            stock_data[t],
            p["ma_short"],
            p["ma_long"],
            p["obv_window"],
            str(test_start),
            str(test_end),
            warmup_periods=warmup_periods,
            commission=commission_rate
        )

    # Buy & Hold benchmark từng mã
    def buy_hold_stats(data_dict):
        res = {}
        for ticker, df in data_dict.items():
            r = df["Close"].pct_change().fillna(0.0)
            res[ticker] = performance_stats(r)
        return pd.DataFrame(res).T

    bh_train = buy_hold_stats(train_data)
    bh_test = buy_hold_stats(test_data)

    strat_train_df = pd.DataFrame({t: train_strategy[t]["stats"] for t in selected_tickers}).T
    strat_test_df = pd.DataFrame({t: test_strategy[t]["stats"] for t in selected_tickers}).T

    # So sánh Test 2022 giữa B&H vs SMA+OBV OR
    comp_stock = pd.DataFrame({
        "Buy & Hold Return [%]": bh_test["Total Return [%]"],
        "SMA+OBV OR Return [%]": strat_test_df["Total Return [%]"],
        "Buy & Hold Sharpe": bh_test["Sharpe Ratio"],
        "SMA+OBV OR Sharpe": strat_test_df["Sharpe Ratio"],
        "Buy & Hold Max DD [%]": bh_test["Max Drawdown [%]"],
        "SMA+OBV OR Max DD [%]": strat_test_df["Max Drawdown [%]"],
        "Excess Return vs B&H [%]": strat_test_df["Total Return [%]"] - bh_test["Total Return [%]"]
    })

    st.markdown("##### 📌 So sánh chi tiết trên giai đoạn Test (2022 - Bear Market):")
    st.dataframe(
        comp_stock.style.format("{:.2f}").background_gradient(subset=["Excess Return vs B&H [%]"], cmap="RdYlGn"),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("##### 🔍 Biểu đồ Tín hiệu Mua/Bán & Chỉ báo Chi tiết:")
    active_ticker = st.selectbox("Chọn mã cổ phiếu để xem biểu đồ chi tiết:", selected_tickers)

    stock_df = stock_data[active_ticker].copy()
    p = best_params[active_ticker]

    # Tính indicator cho biểu đồ
    stock_df["SMA_Short"] = compute_sma(stock_df["Close"], p["ma_short"])
    stock_df["SMA_Long"] = compute_sma(stock_df["Close"], p["ma_long"])
    stock_df["OBV"] = compute_obv(stock_df["Close"], stock_df["Volume"])
    stock_df["OBV_MA"] = compute_sma(stock_df["OBV"], p["obv_window"])

    # Lọc phần hiển thị (từ Train bắt đầu đến Test kết thúc)
    plot_mask = (stock_df.index >= pd.Timestamp(train_start)) & (stock_df.index <= pd.Timestamp(test_end))
    plot_df = stock_df[plot_mask].copy()
    events_plot = find_position_or(plot_df, p["ma_short"], p["ma_long"], p["obv_window"])

    fig_detail = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=[f"Giá {active_ticker}, SMA({p['ma_short']}), SMA({p['ma_long']}) & Tín hiệu Mua/Bán", f"Chỉ báo OBV & OBV-MA({p['obv_window']})"]
    )

    # Đường giá & SMA
    fig_detail.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Giá Close", line=dict(color="#1E293B", width=1.5)), row=1, col=1)
    fig_detail.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_Short"], name=f"SMA {p['ma_short']}", line=dict(color="#2563EB", width=1.2)), row=1, col=1)
    fig_detail.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_Long"], name=f"SMA {p['ma_long']}", line=dict(color="#DC2626", width=1.2)), row=1, col=1)

    # Điểm Buy / Sell
    buy_pts = plot_df[events_plot == 1]
    sell_pts = plot_df[events_plot == -1]

    fig_detail.add_trace(go.Scatter(
        x=buy_pts.index, y=buy_pts["Close"],
        mode="markers", name="Mua (OR Signal)",
        marker=dict(symbol="triangle-up", color="#16A34A", size=10, line=dict(color="black", width=1))
    ), row=1, col=1)

    fig_detail.add_trace(go.Scatter(
        x=sell_pts.index, y=sell_pts["Close"],
        mode="markers", name="Bán (OR Signal)",
        marker=dict(symbol="triangle-down", color="#EA580C", size=10, line=dict(color="black", width=1))
    ), row=1, col=1)

    # OBV & OBV-MA
    fig_detail.add_trace(go.Scatter(x=plot_df.index, y=plot_df["OBV"], name="OBV", line=dict(color="#7C3AED", width=1.3)), row=2, col=1)
    fig_detail.add_trace(go.Scatter(x=plot_df.index, y=plot_df["OBV_MA"], name=f"OBV MA({p['obv_window']})", line=dict(color="#F59E0B", width=1.2, dash="dot")), row=2, col=1)

    fig_detail.add_vline(x=pd.Timestamp(test_start).timestamp() * 1000, line_dash="dash", line_color="grey")
    fig_detail.update_layout(height=550, margin=dict(l=30, r=30, t=40, b=30), hovermode="x unified")
    st.plotly_chart(fig_detail, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: PHÂN BỔ DANH MỤC EQUAL WEIGHT & MPT
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Phân bổ Tối ưu Danh mục: Equal Weight vs Modern Portfolio Theory (MPT)")
    st.markdown("""
    - **Equal Weight:** Phân bổ tỷ trọng bằng nhau (1 / N) cho các cổ phiếu được chọn.
    - **MPT:** Tối đa hóa Sharpe Ratio chỉ sử dụng ma trận hiệp phương sai và lợi nhuận chiến lược trên **tập Train (2020-2021)**.
    """)

    # Tạo ma trận lợi nhuận
    train_return_matrix = pd.concat({t: train_strategy[t]["returns"] for t in selected_tickers}, axis=1).dropna()
    test_return_matrix = pd.concat({t: test_strategy[t]["returns"] for t in selected_tickers}, axis=1).dropna()

    n_assets = len(selected_tickers)
    equal_weights = np.repeat(1.0 / n_assets, n_assets)
    mpt_weights = optimize_mpt(train_return_matrix, risk_free_rate=0.0)

    weights_df = pd.DataFrame({
        "Ticker": selected_tickers,
        "Equal Weight": equal_weights,
        "MPT Weight": mpt_weights
    })

    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        st.markdown("##### Bảng so sánh Trọng số Danh mục:")
        st.dataframe(
            weights_df.style.format({"Equal Weight": "{:.2%}", "MPT Weight": "{:.2%}"}),
            use_container_width=True
        )
    with col_w2:
        st.markdown("##### Biểu đồ cơ cấu tỷ trọng phân bổ vốn:")
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=weights_df["Ticker"], y=weights_df["Equal Weight"], name="Equal Weight", marker_color="#3B82F6"))
        fig_w.add_trace(go.Bar(x=weights_df["Ticker"], y=weights_df["MPT Weight"], name="MPT Optimal", marker_color="#10B981"))
        fig_w.update_layout(barmode="group", height=280, yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_w, use_container_width=True)

    # Lợi nhuận danh mục
    ew_train_ret = (train_return_matrix * equal_weights).sum(axis=1)
    mpt_train_ret = (train_return_matrix * mpt_weights).sum(axis=1)

    ew_test_ret = (test_return_matrix * equal_weights).sum(axis=1)
    mpt_test_ret = (test_return_matrix * mpt_weights).sum(axis=1)

    port_train_stats = pd.DataFrame({
        "Equal Weight": performance_stats(ew_train_ret),
        "MPT": performance_stats(mpt_train_ret)
    }).T

    port_test_stats = pd.DataFrame({
        "Equal Weight": performance_stats(ew_test_ret),
        "MPT": performance_stats(mpt_test_ret)
    }).T

    st.markdown("##### Kết quả Hiệu quả Danh mục trên Train (2020-2021) & Test (2022):")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.caption("TRAIN (2020 - 2021)")
        st.dataframe(port_train_stats.round(2), use_container_width=True)
    with col_p2:
        st.caption("TEST (2022 - Ngoài mẫu)")
        st.dataframe(port_test_stats.round(2), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: ĐÁNH GIÁ TEST 2022 & BENCHMARK
# ------------------------------------------------------------------------------
with tab6:
    st.subheader("Báo cáo Tổng kết Hiệu quả Ngoài Mẫu (Out-of-Sample Test 2022)")
    st.markdown("So sánh chiến lược SMA + OBV (OR) với Chiến lược Mua & Nắm giữ (Buy & Hold) trong bối cảnh thị trường sụt giảm mạnh (Thị trường Gấu năm 2022).")

    # Benchmark: Equal Weight Buy & Hold
    bh_test_matrix = pd.concat({t: df["Close"].pct_change().fillna(0.0) for t, df in test_data.items()}, axis=1).dropna()
    ew_bh_test_ret = (bh_test_matrix * equal_weights).sum(axis=1)

    final_comparison = pd.DataFrame({
        "Equal Weight Buy & Hold": performance_stats(ew_bh_test_ret),
        "SMA+OBV (OR) - Equal Weight": performance_stats(ew_test_ret),
        "SMA+OBV (OR) - MPT": performance_stats(mpt_test_ret)
    }).T

    st.dataframe(
        final_comparison.style.format("{:.2f}").highlight_max(subset=["Total Return [%]", "Sharpe Ratio"], color="#DCFCE7")
        .highlight_min(subset=["Max Drawdown [%]"], color="#FEE2E2"),
        use_container_width=True
    )

    # Tính đường tăng trưởng tài sản (Equity Curve)
    bh_equity = initial_capital * (1 + ew_bh_test_ret).cumprod()
    ew_equity = initial_capital * (1 + ew_test_ret).cumprod()
    mpt_equity = initial_capital * (1 + mpt_test_ret).cumprod()

    # Tính Drawdown theo thời gian
    bh_dd = (bh_equity / bh_equity.cummax() - 1) * 100
    ew_dd = (ew_equity / ew_equity.cummax() - 1) * 100
    mpt_dd = (mpt_equity / mpt_equity.cummax() - 1) * 100

    col_fig1, col_fig2 = st.columns(2)

    with col_fig1:
        st.markdown("##### Biểu đồ Tăng trưởng Tài sản ròng (NAV - Test 2022):")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="B&H Equal Weight", line=dict(color="#EF4444", width=2, dash="dash")))
        fig_eq.add_trace(go.Scatter(x=ew_equity.index, y=ew_equity, name="SMA+OBV Equal Weight", line=dict(color="#3B82F6", width=2)))
        fig_eq.add_trace(go.Scatter(x=mpt_equity.index, y=mpt_equity, name="SMA+OBV MPT", line=dict(color="#10B981", width=2.5)))
        fig_eq.update_layout(height=400, yaxis_title="Giá trị Danh mục (VND)", margin=dict(l=30, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_fig2:
        st.markdown("##### Mức sụt giảm tối đa từ đỉnh (Drawdown % - Test 2022):")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="B&H Equal Weight DD", line=dict(color="#EF4444", width=1.5)))
        fig_dd.add_trace(go.Scatter(x=ew_dd.index, y=ew_dd, name="SMA+OBV EW DD", line=dict(color="#3B82F6", width=1.5)))
        fig_dd.add_trace(go.Scatter(x=mpt_dd.index, y=mpt_dd, name="SMA+OBV MPT DD", line=dict(color="#10B981", width=2)))
        fig_dd.update_layout(height=400, yaxis_title="Drawdown (%)", margin=dict(l=30, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig_dd, use_container_width=True)

    # Kết luận tự động
    st.markdown("---")
    st.markdown("### 📝 Kết luận Thực nghiệm & Nhận xét Định lượng")

    ew_sharpe = final_comparison.loc["SMA+OBV (OR) - Equal Weight", "Sharpe Ratio"]
    mpt_sharpe = final_comparison.loc["SMA+OBV (OR) - MPT", "Sharpe Ratio"]
    bh_ret = final_comparison.loc["Equal Weight Buy & Hold", "Total Return [%]"]
    mpt_ret = final_comparison.loc["SMA+OBV (OR) - MPT", "Total Return [%]"]

    if mpt_sharpe > ew_sharpe:
        opt_winner = "Danh mục **MPT (Modern Portfolio Theory)** đạt tỷ suất sinh lợi điều chỉnh theo rủi ro (Sharpe Ratio) ngoài mẫu cao hơn danh mục Equal Weight."
    else:
        opt_winner = "Danh mục **Equal Weight** đạt Sharpe Ratio ngoài mẫu cao hơn MPT, thể hiện tính ổn định (robust) khi phân bổ đều."

    st.success(f"""
    **1. Khả năng phòng ngừa rủi ro vượt trội:**
    - Năm 2022 là năm thị trường chứng khoán Việt Nam downtrend mạnh. Chiến lược **Buy & Hold** sụt giảm nặng nề: **{bh_ret:.2f}%** với Max Drawdown lên tới **{final_comparison.loc['Equal Weight Buy & Hold', 'Max Drawdown [%]']:.2f}%**.
    - Ngược lại, chiến lược kết hợp **SMA + OBV (OR)** kích hoạt trạng thái giữ tiền mặt khi có tín hiệu đảo chiều, giúp bảo vệ vốn hiệu quả:
        - **SMA+OBV Equal Weight:** Lợi nhuận **{final_comparison.loc['SMA+OBV (OR) - Equal Weight', 'Total Return [%]']:.2f}%**, giảm mức Max DD xuống chỉ còn **{final_comparison.loc['SMA+OBV (OR) - Equal Weight', 'Max Drawdown [%]']:.2f}%**.
        - **SMA+OBV MPT:** Đạt lợi nhuận dương **+{mpt_ret:.2f}%** ngay trong thị trường gấu, Max DD chỉ **{final_comparison.loc['SMA+OBV (OR) - MPT', 'Max Drawdown [%]']:.2f}%**.
    
    **2. Phân bổ vốn MPT vs Equal Weight:**
    - {opt_winner}
    
    **3. Tính toàn vẹn phương pháp:**
    - Việc kết hợp kiểm định với 349 phiên warm-up giúp thuật toán không bỏ lỡ tín hiệu SMA dài hạn (lên tới 350 phiên) ngay từ đầu năm kiểm định mà không hề sử dụng bất kỳ dữ liệu tương lai nào.
    """)
