from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from scipy.stats import zscore

matplotlib.use('Agg')

app = Flask(__name__)

model = load_model("model.keras")

def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock=stock, no_of_days=no_of_days))
    return render_template("index.html")

@app.route("/predict")
def predict():
    stock = request.args.get("stock", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(stock, start, end)
    if stock_data.empty:
        return render_template("result.html", error="Invalid stock ticker or no data available.")
    
    stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()

    fig4 = plt.figure(figsize=(15, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.plot(stock_data['SMA50'], label='SMA 50', linestyle='--')
    plt.plot(stock_data['SMA200'], label='SMA 200', linestyle='--')
    plt.title("Price Trend with Moving Averages")
    plt.legend()
    trend_plot = plot_to_html(fig4)

    splitting_len = int(len(stock_data) * 0.9)
    x_test = stock_data[['Close']][splitting_len:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    predictions = model.predict(x_data)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    plotting_data = pd.DataFrame({
        'Original Test Data': inv_y_test.flatten(),
        'Predicted Test Data': inv_predictions.flatten()
    }, index=x_test.index[100:])

    fig1 = plt.figure(figsize=(15, 6))
    plt.plot(stock_data['Close'], 'b', label='Close Price')
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    original_plot = plot_to_html(fig1)

    fig2 = plt.figure(figsize=(15, 6))
    plt.plot(plotting_data['Original Test Data'], label="Original Test Data")
    plt.plot(plotting_data['Predicted Test Data'], label="Predicted Test Data", linestyle="--")
    plt.legend()
    plt.title("Original vs Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    predicted_plot = plot_to_html(fig2)

    last_100 = stock_data[['Close']].tail(100)
    last_100_scaled = scaler.transform(last_100)

    future_predictions = []
    last_100_scaled = last_100_scaled.reshape(1, -1, 1)
    for _ in range(no_of_days):
        next_day = model.predict(last_100_scaled)
        future_predictions.append(scaler.inverse_transform(next_day))
        last_100_scaled = np.append(last_100_scaled[:, 1:, :], next_day.reshape(1, 1, -1), axis=1)

    future_predictions = np.array(future_predictions).flatten()

    fig3 = plt.figure(figsize=(15, 6))
    plt.plot(range(1, no_of_days + 1), future_predictions, marker='o', label="Predicted Future Prices", color="purple")
    plt.title("Future Close Price Predictions")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.grid(alpha=0.3)
    plt.legend()
    future_plot = plot_to_html(fig3)

    
    stock_data['Volatility'] = stock_data['Close'].rolling(window=10).std()
    fig5 = plt.figure(figsize=(15, 5))
    plt.plot(stock_data['Volatility'], color='orange', label='10-Day Rolling Volatility')
    plt.title("Volatility Measurement")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    volatility_plot = plot_to_html(fig5)


    z_scores = zscore(stock_data['Close'].dropna())
    anomalies = stock_data[(np.abs(z_scores) > 3)]

    fig6 = plt.figure(figsize=(15, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.scatter(anomalies.index, anomalies['Close'], color='red', label='Anomalies')
    plt.title("Anomaly Detection")
    plt.legend()
    anomaly_plot = plot_to_html(fig6)


    return render_template(
        "result.html",
        stock=stock,
        original_plot=original_plot,
        predicted_plot=predicted_plot,
        future_plot=future_plot,
        trend_plot=trend_plot,
        volatility_plot=volatility_plot,
        anomaly_plot=anomaly_plot,
        enumerate =enumerate,
        future_predictions=future_predictions
    )

if __name__ == "__main__":
    app.run(debug=True)
