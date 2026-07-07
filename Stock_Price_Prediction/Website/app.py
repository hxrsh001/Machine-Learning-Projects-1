import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import LSTM
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model with compatibility handling
custom_objects = {
    'LSTM': lambda **kwargs: LSTM(**{k: v for k, v in kwargs.items() if k != 'time_major'})
}

model_path = os.path.join(BASE_DIR, "stock_dl_model.h5")
model = load_model(model_path, custom_objects=custom_objects)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')

        if not stock:
            stock = 'GOOG'

        # Download stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2025, 1, 1)

        df = yf.download(stock, start=start, end=end)

        if df.empty:
            return render_template(
                'index.html',
                error="Invalid Stock Symbol or No Data Found."
            )

        # Descriptive statistics
        data_desc = df.describe()

        # Moving averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Train/Test split
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Prediction
        y_predicted = model.predict(x_test, verbose=0)

        scale_factor = 1 / scaler.scale_[0]

        y_predicted *= scale_factor
        y_test *= scale_factor

        ###############################
        # Plot 1
        ###############################
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(df.Close, label="Closing Price")
        ax1.plot(ema20, label="EMA 20")
        ax1.plot(ema50, label="EMA 50")

        ax1.set_title("Closing Price vs EMA (20 & 50)")
        ax1.legend()

        ema_chart_path = os.path.join(BASE_DIR, "static", "ema_20_50.png")
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        ###############################
        # Plot 2
        ###############################
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        ax2.plot(df.Close, label="Closing Price")
        ax2.plot(ema100, label="EMA 100")
        ax2.plot(ema200, label="EMA 200")

        ax2.set_title("Closing Price vs EMA (100 & 200)")
        ax2.legend()

        ema_chart_path_100_200 = os.path.join(
            BASE_DIR,
            "static",
            "ema_100_200.png"
        )

        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        ###############################
        # Plot 3
        ###############################
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        ax3.plot(y_test, label="Original Price")
        ax3.plot(y_predicted, label="Predicted Price")

        ax3.set_title("Prediction vs Original")
        ax3.legend()

        prediction_chart_path = os.path.join(
            BASE_DIR,
            "static",
            "stock_prediction.png"
        )

        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        ###############################
        # Save Dataset
        ###############################
        csv_file_path = os.path.join(
            BASE_DIR,
            "static",
            f"{stock}_dataset.csv"
        )

        df.to_csv(csv_file_path)

        return render_template(
            'index.html',
            plot_path_ema_20_50='static/ema_20_50.png',
            plot_path_ema_100_200='static/ema_100_200.png',
            plot_path_prediction='static/stock_prediction.png',
            data_desc=data_desc.to_html(classes='table table-bordered'),
            dataset_link=f"{stock}_dataset.csv"
        )

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(BASE_DIR, "static", filename),
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)