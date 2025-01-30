from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from data_preprocessing import preprocess_data

def build_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    model.save('stock_price_lstm_model.h5')
    print("Model trained and saved as stock_price_lstm_model.h5")
    return model

if __name__ == '__main__':
    x_train, y_train, _, _ = preprocess_data('AAPL_data.csv')
    build_model(x_train, y_train)
