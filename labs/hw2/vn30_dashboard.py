import yahoo_fin.stock_info as si
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# Danh sách Mã Cổ phiếu VN30 (Dùng hậu tố .VN cho yfinance/yahoo_fin)
# Lưu ý: Danh sách này là mẫu và cần được cập nhật theo thời gian thực
# ==============================================================================
VN30_TICKERS = [
    'VCB.VN', 'MSN.VN', 'HPG.VN', 'FPT.VN', 'GAS.VN', 'VIC.VN', 
    'VHM.VN', 'VNM.VN', 'BID.VN', 'CTG.VN', 'ACB.VN', 'MBB.VN', 
    'SSI.VN', 'VPB.VN', 'SAB.VN', 'TCH.VN', 'PDR.VN', 'PLX.VN'
]

# ==============================================================================
# Tab 1 Summary
# ==============================================================================

def tab1(ticker):
    
    st.title("Tóm tắt Thị trường")
    st.header(f"Mã: {ticker.replace('.VN', '')}")
    
    # Lấy bảng tóm tắt báo giá (Quote Table)
    @st.cache_data
    def get_summary_info(ticker):
        try:
            # yf.Ticker().info thường cung cấp nhiều dữ liệu hơn cho thị trường quốc tế
            stock = yf.Ticker(ticker)
            info = stock.info
            
            summary_data = {
                "Giá hiện tại (Current Price)": info.get('currentPrice'),
                "Giá trị vốn hóa (Market Cap)": info.get('marketCap'),
                "P/E Ratio": info.get('trailingPE'),
                "P/B Ratio": info.get('priceToBook'),
                "Tỷ suất cổ tức (Yield)": info.get('dividendYield'),
                "Giá cao nhất 52 tuần": info.get('fiftyTwoWeekHigh'),
                "Giá thấp nhất 52 tuần": info.get('fiftyTwoWeekLow'),
                "Beta": info.get('beta'),
                "Vol. trung bình 10 ngày": info.get('averageDailyVolume10Day'),
            }
            
            # Chuyển đổi thành DataFrame để hiển thị
            df = pd.DataFrame(list(summary_data.items()), columns=['Thuộc tính', 'Giá trị'])
            df.set_index('Thuộc tính', inplace=True)
            
            return df
        except Exception as e:
            st.error(f"Không thể tải dữ liệu tóm tắt cho {ticker}. Lỗi: {e}")
            return pd.DataFrame()
        
    if ticker != '-':
        summary_df = get_summary_info(ticker)
        if not summary_df.empty:
            st.dataframe(summary_df)
            
    # Biểu đồ giá lịch sử
    @st.cache_data 
    def get_stock_data(ticker):
        # Lấy dữ liệu giá tối đa có thể
        stockdata = yf.download(ticker, period = 'max')
        return stockdata
        
    if ticker != '-':
        chartdata = get_stock_data(ticker) 
        if not chartdata.empty:
            fig = px.area(chartdata, x=chartdata.index, y=chartdata['Close'].squeeze(), title=f'Biến động Giá Đóng cửa {ticker.replace(".VN", "")}')
            
            # Thêm Range Selector
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1T", step="month", stepmode="backward"),
                        dict(count=3, label="3T", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1N", step="year", stepmode="backward"),
                        dict(count=5, label="5N", step="year", stepmode="backward"),
                        dict(label = "MAX", step="all")
                    ])
                )
            )
            st.plotly_chart(fig)
            
# ==============================================================================
# Tab 2 Chart (Biểu đồ Kỹ thuật)
# ==============================================================================

def tab2(ticker):
    st.title("Biểu đồ Phân tích Kỹ thuật")
    st.header(f"Mã: {ticker.replace('.VN', '')}")
    
    st.write("Đặt Thời lượng là '-' để chọn Phạm vi Ngày")
    
    c1, c2, c3, c4, c5 = st.columns((1,1,1,1,1))
    
    with c1:
        start_date = st.date_input("Ngày Bắt đầu", datetime.today().date() - timedelta(days=90))
    with c2:
        end_date = st.date_input("Ngày Kết thúc", datetime.today().date())        
    with c3:
        duration = st.selectbox("Chọn Thời lượng", ['-', '1Mo', '3Mo', '6Mo', 'YTD','1Y', '3Y','5Y', 'MAX'])          
    with c4: 
        inter = st.selectbox("Chọn Khoảng thời gian", ['1d', '1wk', '1mo'])
    with c5:
        plot = st.selectbox("Chọn Loại Biểu đồ", ['Line', 'Candle'])
        
    # Hàm lấy dữ liệu với SMA 50
    @st.cache_data             
    def get_chart_data(ticker, start_date, end_date, duration, inter):
        # 1. Lấy dữ liệu lịch sử tối đa để tính SMA
        SMA_data = yf.download(ticker, period = 'max')
        SMA_data['SMA_50'] = SMA_data['Close'].rolling(50).mean()
        SMA_data = SMA_data.reset_index()[['Date', 'SMA_50']]
        
        # 2. Lấy dữ liệu chính theo lựa chọn của người dùng
        if duration != '-':        
            chartdata = yf.download(ticker, period = duration, interval = inter)
        else:
            chartdata = yf.download(ticker, start=start_date, end=end_date, interval = inter)
        
        chartdata = chartdata.reset_index()
        # 3. Hợp nhất với cột SMA 50
        chartdata = chartdata.merge(SMA_data, on='Date', how='left')
        return chartdata
    
    if ticker != '-':
        chartdata = get_chart_data(ticker, start_date, end_date, duration, inter) 
        if not chartdata.empty:
            
            fig = make_subplots(specs=[[{"secondary_y": True}]]) # Hai trục Y
            
            # Thêm Biểu đồ Giá
            if plot == 'Line':
                fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['Close'], mode='lines', 
                                         name = 'Giá Đóng cửa'), secondary_y = False)
            else:
                fig.add_trace(go.Candlestick(x = chartdata['Date'], open = chartdata['Open'], 
                                             high = chartdata['High'], low = chartdata['Low'], 
                                             close = chartdata['Close'], name = 'Nến'), secondary_y = False)
              
            # Thêm Đường SMA 50
            fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['SMA_50'], mode='lines', 
                                     name = '50-day SMA'), secondary_y = False)
            
            # Thêm Biểu đồ Volume
            fig.add_trace(go.Bar(x = chartdata['Date'], y = chartdata['Volume'], 
                                 name = 'Khối lượng'), secondary_y = True)

            # Cấu hình trục Y Volume (để bar volume nhỏ hơn)
            fig.update_yaxes(range=[0, chartdata['Volume'].max()*3], showticklabels=False, secondary_y=True)
            fig.update_layout(title_text=f"Biểu đồ Giá và Volume cho {ticker.replace('.VN', '')}")
            fig.update_xaxes(rangeslider_visible=False) # Không cần range slider vì đã có range selector
        
            st.plotly_chart(fig, use_container_width=True)
           
# ==============================================================================
# Tab 3 Financials (Tài chính)
# ==============================================================================

def tab3(ticker):
      st.title("Phân tích Báo cáo Tài chính")
      st.header(f"Mã: {ticker.replace('.VN', '')}")
      
      statement = st.selectbox("Chọn Báo cáo", ['Income Statement (Báo cáo Thu nhập)', 'Balance Sheet (Bảng Cân đối)', 'Cash Flow (Lưu chuyển Tiền tệ)'])
      period = st.selectbox("Kỳ báo cáo", ['Yearly (Năm)', 'Quarterly (Quý)'])
      
      # Các hàm lấy dữ liệu tài chính (dùng yfinance)
      @st.cache_data
      def get_financial_data(ticker, statement, period):
          stock = yf.Ticker(ticker)
          yearly = (period == 'Yearly (Năm)')
          
          if statement == 'Income Statement (Báo cáo Thu nhập)':
              data = stock.financials if yearly else stock.quarterly_financials
          elif statement == 'Balance Sheet (Bảng Cân đối)':
              data = stock.balance_sheet if yearly else stock.quarterly_balance_sheet
          else: # Cash Flow
              data = stock.cashflow if yearly else stock.quarterly_cashflow
              
          if data is None or data.empty:
              return pd.DataFrame({"Lưu ý": ["Không có dữ liệu hoặc lỗi truy xuất."]})

          # Định dạng lại DataFrame (transpose và reset index)
          data = data.T # Transpose để cột là kỳ báo cáo
          data.index = data.index.strftime('%Y-%m-%d')
          data.index.name = 'Ngày Báo cáo'
          return data.T

          
      if ticker != '-':
          data = get_financial_data(ticker, statement, period)
          st.dataframe(data, use_container_width=True)

# ==============================================================================
# Tab 4 Monte Carlo Simulation
# ==============================================================================

def tab4(ticker):
     st.title("Mô phỏng Monte Carlo & VaR")
     st.header(f"Mã: {ticker.replace('.VN', '')}")
     
     # Dropdown cho chọn số lượng mô phỏng và chân trời thời gian
     simulations = st.selectbox("Số lượng Mô phỏng (N)", [50, 100, 200, 500, 1000], index=0)
     time_horizon = st.selectbox("Chân trời Thời gian (T) - Ngày", [30, 60, 90], index=0)
     
     # Hàm Monte Carlo
     @st.cache_data(show_spinner="Đang chạy mô phỏng Monte Carlo...")
     def montecarlo(ticker, time_horizon, simulations):
     
         end_date = datetime.now().date()
         start_date = end_date - timedelta(days=90) # Lấy 90 ngày dữ liệu để tính biến động
     
         stock_price = yf.download(ticker, start=start_date, end=end_date)
         close_price = stock_price['Close']
         
         # Kiểm tra dữ liệu
         if close_price.empty or len(close_price) < 2:
             st.warning("Không đủ dữ liệu lịch sử để chạy mô phỏng.")
             return None, None
     
         daily_return = close_price.pct_change().dropna()
         daily_volatility = np.std(daily_return) # Độ lệch chuẩn của lợi suất hàng ngày
     
         simulation_df = pd.DataFrame()
         last_price = close_price.iloc[-1]
     
         for i in range(simulations):        
                next_price = []
                temp_last_price = last_price
                
                for x in range(time_horizon):
                      # Lợi suất tương lai ngẫu nhiên (dựa trên phân phối chuẩn)
                      future_return = np.random.normal(0, daily_volatility) 
                      future_price = temp_last_price * (1 + future_return)
                      next_price.append(future_price)
                      temp_last_price = future_price
    
                simulation_df[i] = next_price
                
         return simulation_df, last_price
          
     if ticker != '-':
         mc_df, current_price = montecarlo(ticker, time_horizon, simulations)
         
         if mc_df is not None:
             # Plot Biểu đồ Mô phỏng
             fig, ax = plt.subplots(figsize=(12, 8))
             ax.plot(mc_df)
             plt.title(f'Mô phỏng Monte Carlo cho {ticker.replace(".VN", "")} trong {time_horizon} ngày')
             plt.xlabel('Ngày')
             plt.ylabel('Giá (VND)')
             
             plt.axhline(y= current_price.iloc[-1], color ='red', linestyle='--')
             plt.legend([f'Giá hiện tại: {np.round(current_price, 2)} VND'], loc='upper left')
             st.pyplot(fig)
             
             # Tính toán Value at Risk (VaR)
             st.subheader('Giá trị Rủi ro (Value at Risk - VaR)')
             ending_price = mc_df.iloc[-1, :].values
             
             # Phân vị thứ 5 (5th Percentile)
             ending_price = np.asarray(ending_price).astype(float).flatten()
             future_price_95ci = np.percentile(ending_price, 5)

             
             # Plot Phân phối giá kết thúc
             fig1, ax1 = plt.subplots(figsize=(12, 8))
             ax1.hist(ending_price, bins=50)
             plt.axvline(future_price_95ci, color='red', linestyle='--', linewidth=2)
             plt.legend([f'Phân vị thứ 5 (95% VaR): {np.round(future_price_95ci, 2)} VND'])
             plt.title('Phân phối Giá Kết thúc Mô phỏng')
             plt.xlabel('Giá (VND)')
             plt.ylabel('Tần suất')
             st.pyplot(fig1)
             
             # Kết quả VaR
             VaR = current_price - future_price_95ci
             st.markdown(f"**VaR tại mức tin cậy 95% là: {np.round(VaR, 2)} VND**")
             st.info("Ý nghĩa: Có 95% khả năng khoản lỗ lớn nhất của cổ phiếu sẽ không vượt quá VaR trong khoảng thời gian T.")
         
# ==============================================================================
# Tab 5 Portfolio's Trend (Xu hướng Danh mục)
# ==============================================================================

def tab5():
      st.title("So sánh Xu hướng Danh mục VN30")
      
      # Lọc bỏ hậu tố .VN cho hiển thị
      display_tickers = [t.replace('.VN', '') for t in VN30_TICKERS] 
      
      selected_display_tickers = st.multiselect(
          "Chọn mã cổ phiếu trong danh mục", 
          options = display_tickers, 
          default = ['FPT', 'HPG', 'VCB']
      )
      
      # Thêm lại hậu tố .VN cho việc tải dữ liệu
      selected_tickers = [t + '.VN' for t in selected_display_tickers]
      
      time_period = st.selectbox("Chọn Thời gian So sánh", ['1Y', '3Y', '5Y'])

      @st.cache_data
      def get_portfolio_data(tickers, period):
          df = pd.DataFrame()
          for ticker in tickers:
              data = yf.download(ticker, period = period)
              if not data.empty:
                  # Tính Normalized Price (Giá chuẩn hóa về 100) để so sánh hiệu suất dễ hơn
                  df[ticker.replace('.VN', '')] = data['Close'] / data['Close'].iloc[0] * 100
          return df
          
      if selected_tickers:
          df_portfolio = get_portfolio_data(selected_tickers, time_period)
          
          if not df_portfolio.empty:
              fig = px.line(df_portfolio, 
                            title=f'So sánh Hiệu suất Cổ phiếu trong Danh mục ({time_period} - Chuẩn hóa về 100)')
              fig.update_yaxes(title="Giá trị Chuẩn hóa (Bắt đầu = 100)")
              st.plotly_chart(fig, use_container_width=True) 
      else:
          st.warning("Vui lòng chọn ít nhất một mã cổ phiếu để xem xu hướng.")
        
    
# ==============================================================================
# Main body
# ==============================================================================

def run_vn30_dashboard():
    
    st.set_page_config(layout="wide", page_title="VN30 Financial Dashboard")
    
    # Danh sách Ticker trong Sidebar
    ticker_list = ['-'] + VN30_TICKERS
    
    # Thêm Selection Box (sidebar)
    global selected_ticker
    selected_ticker = st.sidebar.selectbox("Chọn Mã Cổ phiếu VN30", ticker_list)
    
    # Add Radio Box (sidebar)
    select_tab = st.sidebar.radio("Chọn Tab Phân tích", 
                                  ['Tóm tắt', 'Biểu đồ Kỹ thuật', 'Báo cáo Tài chính', 
                                   'Monte Carlo Simulation', "Xu hướng Danh mục"])
    
    # Show the selected tab
    if selected_ticker == '-':
        st.info("Vui lòng chọn một mã cổ phiếu VN30 từ thanh bên (sidebar) để bắt đầu phân tích.")
    elif select_tab == 'Tóm tắt':
        tab1(selected_ticker)
    elif select_tab == 'Biểu đồ Kỹ thuật':
        tab2(selected_ticker)
    elif select_tab == 'Báo cáo Tài chính':
        tab3(selected_ticker)
    elif select_tab == 'Monte Carlo Simulation':
        tab4(selected_ticker)
    elif select_tab == "Xu hướng Danh mục":
        tab5() # Tab này không cần ticker vì nó cho phép chọn nhiều mã
       
    
if __name__ == "__main__":
    run_vn30_dashboard()