# Import library
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# Load data
file_name ='brentcrudoil.csv'
df= pd.read_csv(file_name)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Eksplorasi dan analisis dataset
st.title("Eksplorasi dan Analisis Dataset Harga Minyak")
st.header("Informasi Umum tentang Dataset")
st.write("data head:")
st.write(df.head())
st.write(f"Jumlah Baris dan Kolom: {df.shape}")
st.write("Tipe Data Kolom:")
st.write(df.dtypes)
st.write("Statistik Deskriptif:")
st.write(df.describe())

# Menampilkan grafik menggunakan matplotlib di dalam aplikasi Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Date'], df['Close'], label='Harga Penutupan', color='blue')
ax.set_title('Data Historis Harga Penutupan')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Penutupan')
ax.legend()

st.pyplot(fig)

# Analisis tren menggunakan moving average
window_size = 10
rolling_mean = df['Close'].rolling(window=window_size).mean()
df['Rolling Mean'] = rolling_mean

st.subheader("Harga Penutupan dengan Rolling Mean")
st.line_chart(df.set_index('Date')[['Close', 'Rolling Mean']])

# Menentukan header untuk aplikasi Streamlit
st.header("Analisis Time Series dengan ARIMA")

# Pisahkan data menjadi training dan testing set
train_size_arima = int(len(df) * 0.8)
train_arima, test_arima = df['High'][:train_size_arima], df['High'][train_size_arima:]

# Menggunakan auto_arima untuk menentukan order ARIMA secara otomatis
autoarima_model = auto_arima(train_arima, seasonal=False, suppress_warnings=True)
order_arima = autoarima_model.get_params()['order']

# Membuat model ARIMA
arima_model = ARIMA(train_arima, order=order_arima)
arima_fit = arima_model.fit()

# Prediksi menggunakan model ARIMA
arima_forecast = arima_fit.predict(start=len(train_arima), end=len(train_arima) + len(test_arima) - 1, typ='levels')

# Plot hasil ARIMA
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Date'], df['High'], label="Original Data")
ax.plot(test_arima.index, arima_forecast, label="ARIMA Forecast", color='green')
ax.set_title("Time Series Analysis with ARIMA")
ax.set_xlabel("year")
ax.set_ylabel("High")
ax.legend()
st.pyplot(fig)

# Membuat PairGrid dengan hue berdasarkan kolom 'Low'
st.header("Visualisasi 'Category'")
g = sns.PairGrid(df, hue='Low')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
st.pyplot(plt.gcf())

# Menentukan header untuk aplikasi Streamlit
st.header("Training Data dan Modeling dengan Linear Regression")

# Mendefinisikan fitur (x) dan target (y)
x = df[["Date"]]
y = df['Close']

# Melatih model Linear Regression
linear_reg_model = LinearRegression().fit(x, y)

# Menghitung tanggal prediksi otomatis (misal: 10 hari setelah data terakhir)
last_date = df['Date'].max()
prediction_date = last_date + pd.DateOffset(days=10)

# Mengonversi tanggal prediksi menjadi nilai numerik
prediction_numeric = (prediction_date - df['Date'].min()).days

# Melakukan prediksi menggunakan model
prediction_result = linear_reg_model.predict([[prediction_numeric]])

# Menampilkan hasil prediksi
st.write(f"Prediksi 'Close' pada tanggal {prediction_date}: {prediction_result[0]}")

st.header("Visualisasi Moving Averages")

# Menambahkan kolom 'MA' untuk Moving Averages
window_size = 10
df['MA'] = df['Close'].rolling(window=window_size).mean()

# Plot hasil Moving Averages
fig_ma, ax_ma = plt.subplots(figsize=(10, 6))
ax_ma.plot(df['Date'], df['Close'], label='Harga Penutupan')
ax_ma.plot(df['Date'], df['MA'], label=f'Moving Averages ({window_size} Hari)')
ax_ma.set_title('Moving Averages')
ax_ma.set_xlabel('Tanggal')
ax_ma.set_ylabel('Harga Penutupan')
ax_ma.legend()
st.pyplot(fig_ma)

st.header("Visualisasi Exponential Smoothing")

# Menambahkan kolom 'Exp_Smooth' untuk Exponential Smoothing
alpha = 0.2  # Anda dapat menyesuaikan nilai alpha sesuai kebutuhan
model = ExponentialSmoothing(df['High'], trend='add', seasonal='add', seasonal_periods=10)
df['Exp_Smooth'] = model.fit(smoothing_level=alpha).fittedvalues

# Plot hasil Exponential Smoothing
fig_exp_smooth, ax_exp_smooth = plt.subplots(figsize=(10, 6))
ax_exp_smooth.plot(df['Date'], df['chg(high)'], label='Harga tertinggi')
ax_exp_smooth.plot(df['Date'], df['Exp_Smooth'], label='Exponential Smoothing')
ax_exp_smooth.set_title('Exponential Smoothing')
ax_exp_smooth.set_xlabel('Tanggal')
ax_exp_smooth.set_ylabel('Harga Tertinggi')
ax_exp_smooth.legend()
st.pyplot(fig_exp_smooth)