import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def data_reformat(df):
    df = df.filter(
        items=['time', 'energy', 'holiday', 'visibility', 'windBearing', 'temperature', 'dewPoint', 'pressure',
               'apparentTemperature', 'windSpeed', 'humidity', 'precipType', 'icon'])
    df_new = df.set_index('time')
    df_new.index.freq = '30T'
    dti = pd.DatetimeIndex(df_new.index, freq='30min')
    data_length = dti.freq
    df.index = dti
    df.drop(columns='time', inplace=True)
    df.select_dtypes(exclude=['number', 'bool']).columns
    df['precipType'].unique()
    df['precipType'] = (df['precipType'] == 'rain').apply(int)
    df['icon'].unique()
    df['icon'] = pd.factorize(df['icon'])[0]
    return df, data_length


def preprocess(df):
    selected_features = ['temperature', 'dewPoint', 'pressure', 'apparentTemperature', 'windSpeed', 'humidity', 'icon',
                         'energy']
    reformated_df, data_length = data_reformat(df)
    columns_to_drop = reformated_df.columns.drop(selected_features)
    reformated_df.drop(columns=columns_to_drop, inplace=True)
    return reformated_df, data_length


def split_timestamps(df, n_trials):
    n_records = len(df)
    part_size = int(n_records / n_trials)
    indices = [i for i in range(part_size, n_records - 1, part_size)]
    timestamps = [df.index[i] for i in indices]
    return timestamps, n_records, part_size


def fetch_trial_data(df, test_start, freq, max_train_size, max_test_size):
    train_stop = test_start - freq
    return (df[:train_stop].tail(max_train_size), df[test_start:].head(max_test_size))


def mape(actual, prediction):
    actual, prediction = np.array(actual), np.array(prediction)
    return np.mean(np.abs((prediction - actual) / actual)) * 100


def rmse(actual, prediction):
    actual, prediction = np.array(actual), np.array(prediction)
    mse = mean_squared_error(actual, prediction)
    return np.sqrt(mse)


def cv(prediction):
    prediction = np.array(prediction)
    return np.std(prediction, ddof=1) / np.mean(prediction) * 100


def split_univariate_X_y(dataset, lags):
    X, y = list(), list()
    for i in range(lags, dataset.shape[0]):
        X.append(dataset[i - lags:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)


def split_multivariate_X_y(dataset, n_steps):
    X, y = list(), list()
    for i in range(len(dataset)):
        end_ix = i + n_steps
        if end_ix > len(dataset)-1:
            break
        X.append(dataset[i:end_ix, :])
        y.append(dataset[end_ix, -1])
    return np.array(X), np.array(y)


def reshape_X_y_to_hstack(X, y):
    X_seq = np.array(X)
    y_seq = np.array(y)
    X_seq = X_seq.reshape((len(X_seq), X_seq.shape[1]))
    y_seq = y_seq.reshape((len(y_seq), 1))
    return np.hstack((X_seq, y_seq))
