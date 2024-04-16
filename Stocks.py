import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# create start and end dates
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# create the streamlit page
st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# load stock data from yahoo finance
@st.cache 
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading Data...done!")

# analyze data
st.subheader("Previous data")
st.write(data.tail())

# plot the data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting using facebook prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date':'ds', 'Close':'y'})

m = Prophet()
m.fit(df_train)
future_df = m.make_future_dataframe(periods = period)
forecast = m.predict(future_df)

st.subheader('Forecast data')
st.write(forecast.tail())

# plotting forecast data and components
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)