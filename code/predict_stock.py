import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess_data

def make_predictions(model, x_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def prepare_test_data():
    _, _, scaled_data, training_data_len = preprocess_data('AAPL_data.csv')
    test_data = scaled_data[training_data_len-60:, :]
    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        y_test.append(test_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test

if __name__ == '__main__':
    x_test, y_test = prepare_test_data()
    model = load_model('stock_price_lstm_model.h5')
    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = make_predictions(model, x_test, scaler)
    print(predictions)
