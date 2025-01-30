import matplotlib.pyplot as plt
import pandas as pd
from predict_stock_price import make_predictions, prepare_test_data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def plot_results(predictions):
    data = pd.read_csv('AAPL_data.csv')
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Actual Data')
    plt.plot(predictions, label='Predicted Data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x_test, y_test = prepare_test_data()
    model = load_model('stock_price_lstm_model.h5')
    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = make_predictions(model, x_test, scaler)
    plot_results(predictions)
