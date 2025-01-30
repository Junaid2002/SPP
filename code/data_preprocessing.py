import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filename):
    data = pd.read_csv(filename)
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaled_data, training_data_len

if __name__ == '__main__':
    x_train, y_train, scaled_data, training_data_len = preprocess_data('AAPL_data.csv')
