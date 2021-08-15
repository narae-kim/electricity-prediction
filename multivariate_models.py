import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import variation
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import common_util as cu

PLOT_DIR = 'multivariate'
LAGS = 48
epochs = 1000

n_features = 8
n_seq = 2
n_steps = int(LAGS / n_seq)

df = pd.read_csv('data/processedMultivariateData_0.csv')

sc = StandardScaler()
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

num = 0
for house in ['3422', '3668', '3737', '3851']:
    num += 1
    print("\n*** House", num, "-", house)

    df_house = df[df['LCLid'] == 'MAC00' + house]
    df_house, data_length = cu.preprocess(df_house)
    timestamps, n_records, part_size = cu.split_timestamps(df_house, 10)

    df_rest, df_test = cu.fetch_trial_data(df_house, timestamps[8], data_length, n_records, part_size)
    df_train, df_validate = cu.fetch_trial_data(df_rest, timestamps[7], data_length, n_records, part_size)
    print("The number of records in the training set:", len(df_train), "from", df_train.index[0], "to", df_train.index[-1])
    print("The number of records in the validation set:", len(df_validate), "from ", df_validate.index[0], "to", df_validate.index[-1])
    print("The number of records in the test set:", len(df_test), "from", df_test.index[0], "to", df_test.index[-1], "\n")

    X_train_df = df_train.iloc[:, 1:]
    X_validate_df = df_validate.iloc[:, 1:]
    X_test_df = df_test.iloc[:, 1:]
    y_train_df = df_train.iloc[:, 0:1]
    y_validate_df = df_validate.iloc[:, 0:1]
    y_test_df = df_test.iloc[:, 0:1]

    # training / validation
    train_dataset = cu.reshape_X_y_to_hstack(df_train.iloc[:, 1:], y_train_df)
    training_set_scaled = sc.fit_transform(train_dataset)
    X_train, y_train = cu.split_multivariate_X_y(training_set_scaled, LAGS)
    X_train_deep = X_train
    X_train_ann = np.reshape(X_train, (-1, X_train.shape[1] * X_train.shape[2]))
    X_train_cnnlstm = X_train_deep.reshape((X_train_deep.shape[0], n_seq, n_steps, n_features))

    validate_dataset = cu.reshape_X_y_to_hstack(df_validate.iloc[:, 1:], y_validate_df)
    validate_set_scaled = sc.transform(validate_dataset)
    X_validate, y_validate = cu.split_multivariate_X_y(validate_set_scaled, LAGS)
    X_validate_deep = X_validate
    X_validate_cnnlstm = X_validate_deep.reshape((X_validate_deep.shape[0], n_seq, n_steps, n_features))
    X_validate_ann = np.reshape(X_validate, (-1, X_validate.shape[1] * X_validate.shape[2]))

    X_train_svm = np.concatenate((X_train, X_validate))
    X_train_svm = np.reshape(X_train_svm, (X_train_svm.shape[0], X_train_svm.shape[1] * X_train_svm.shape[2]))
    y_train_svm = np.concatenate((y_train, y_validate))

    # test
    test_dataset = cu.reshape_X_y_to_hstack(df_test.iloc[:, 1:], y_test_df)
    test_set_scaled = sc.transform(test_dataset)
    X_test, y_test = cu.split_multivariate_X_y(test_set_scaled, LAGS)
    X_test_deep = X_test
    X_test_svm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    X_test_ann = np.reshape(X_test, (-1, X_test.shape[1] * X_test.shape[2]))
    X_test_cnnlstm = X_test_deep.reshape((X_test_deep.shape[0], n_seq, n_steps, n_features))

    actual_y = np.array(y_test_df[LAGS:])

    # SVM
    SVR_model = SVR().fit(X_train_svm, y_train_svm)

    predicted_y_svm = SVR_model.predict(X_test_svm)

    predicted_y_svm = np.append(np.repeat(np.nan, 48), predicted_y_svm)
    test_set_scaled[:, -1] = predicted_y_svm.reshape(-1)
    test_set_scaled_svm = sc.inverse_transform(test_set_scaled)
    predicted_y_svm = test_set_scaled_svm[:, -1].reshape(-1, 1)
    predicted_y_svm = predicted_y_svm[48:]
    print(f'RMSE SVM: {cu.rmse(actual_y, predicted_y_svm)}')
    print(f'MAPE SVM: {cu.mape(actual_y, predicted_y_svm)} (%)')
    print(f'CV SVM: {variation(predicted_y_svm)[0] * 100} (%)')
    # print(f'CV SVM: {cu.cv(predicted_y_svm)} (%)')
    predicted_y_svm = np.append(np.repeat(np.nan, LAGS), predicted_y_svm)
    predicted_y_svm = pd.DataFrame(predicted_y_svm, columns=['energy'])
    predicted_y_svm.index = y_test_df.index

    # ANN
    ANN_model = Sequential()
    ANN_model.add(Dense(32, activation='relu', input_dim=X_train_ann.shape[1]))
    ANN_model.add(Dense(units=32, activation='relu'))
    ANN_model.add(Dense(units=32, activation='relu'))
    ANN_model.add(Dense(units=1))
    ANN_model.compile(optimizer='adam', loss='mse')

    ANN_model.fit(X_train_ann, y_train, validation_data=(X_validate_ann, y_validate), epochs=epochs,
                  callbacks=[callback_early_stopping])

    predicted_y_ann = ANN_model.predict(X_test_ann, verbose=0)
    predicted_y_ann = np.append(np.repeat(np.nan, 48), predicted_y_ann)
    test_set_scaled[:, -1] = predicted_y_ann.reshape(-1)
    test_set_scaled_ann = sc.inverse_transform(test_set_scaled)
    predicted_y_ann = test_set_scaled_ann[:, -1].reshape(-1, 1)
    predicted_y_ann = predicted_y_ann[48:]
    print(f'RMSE ANN: {cu.rmse(actual_y, predicted_y_ann)}')
    print(f'MAPE ANN: {cu.mape(actual_y, predicted_y_ann)} (%)')
    print(f'CV ANN: {variation(predicted_y_ann)[0] * 100} (%)')
    # print(f'CV ANN: {cu.cv(predicted_y_ann)} (%)')
    predicted_y_ann = np.append(np.repeat(np.nan, LAGS), predicted_y_ann)
    predicted_y_ann = pd.DataFrame(predicted_y_ann, columns=['energy'])
    predicted_y_ann.index = y_test_df.index

    # CNN
    CNN_model = Sequential()
    CNN_model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                         input_shape=(X_train_deep.shape[1], X_train_deep.shape[2])))
    CNN_model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(50, activation='relu'))
    CNN_model.add(Dense(1))
    CNN_model.compile(optimizer='adam', loss='mse')

    CNN_model.fit(X_train_deep, y_train, validation_data=(X_validate_deep, y_validate), epochs=epochs,
                  callbacks=[callback_early_stopping])

    predicted_y_cnn = CNN_model.predict(X_test_deep, verbose=0)
    predicted_y_cnn = np.append(np.repeat(np.nan, 48), predicted_y_cnn)
    test_set_scaled[:, -1] = predicted_y_cnn.reshape(-1)
    test_set_scaled_cnn = sc.inverse_transform(test_set_scaled)
    predicted_y_cnn = test_set_scaled_cnn[:, -1].reshape(-1, 1)
    predicted_y_cnn = predicted_y_cnn[48:]
    print(f'RMSE CNN: {cu.rmse(actual_y, predicted_y_cnn)}')
    print(f'MAPE CNN: {cu.mape(actual_y, predicted_y_cnn)} (%)')
    print(f'CV CNN: {variation(predicted_y_cnn)[0] * 100} (%)')
    # print(f'CV CNN: {cu.cv(predicted_y_cnn)} (%)')
    predicted_y_cnn = np.append(np.repeat(np.nan, LAGS), predicted_y_cnn)
    predicted_y_cnn = pd.DataFrame(predicted_y_cnn, columns=['energy'])
    predicted_y_cnn.index = y_test_df.index

    # LSTM
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_deep.shape[1], X_train_deep.shape[2])))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=50, return_sequences=True))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=50, return_sequences=True))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(LSTM(units=50))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(units=1))
    LSTM_model.compile(optimizer='adam', loss='mse')

    LSTM_model.fit(X_train_deep, y_train, validation_data=(X_validate_deep, y_validate), epochs=epochs,
                   callbacks=[callback_early_stopping])

    predicted_y_lstm = LSTM_model.predict(X_test_deep)
    predicted_y_lstm = np.append(np.repeat(np.nan, 48), predicted_y_lstm)
    test_set_scaled[:, -1] = predicted_y_lstm.reshape(-1)
    test_set_scaled_lstm = sc.inverse_transform(test_set_scaled)
    predicted_y_lstm = test_set_scaled_lstm[:, -1].reshape(-1, 1)
    predicted_y_lstm = predicted_y_lstm[48:]
    print(f'RMSE LSTM: {cu.rmse(actual_y, predicted_y_lstm)}')
    print(f'MAPE LSTM: {cu.mape(actual_y, predicted_y_lstm)} (%)')
    print(f'CV LSTM: {variation(predicted_y_lstm)[0] * 100} (%)')
    # print(f'CV LSTM: {cu.cv(predicted_y_lstm)} (%)')
    predicted_y_lstm = np.append(np.repeat(np.nan, LAGS), predicted_y_lstm)
    predicted_y_lstm = pd.DataFrame(predicted_y_lstm, columns=['energy'])
    predicted_y_lstm.index = y_test_df.index

    # CNN-LSTM
    CNNLSTM_model = Sequential()
    CNNLSTM_model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(
    X_train_cnnlstm.shape[1], X_train_cnnlstm.shape[2], X_train_cnnlstm.shape[3]))))
    CNNLSTM_model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
    CNNLSTM_model.add(TimeDistributed(Dropout(0.2)))
    CNNLSTM_model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    CNNLSTM_model.add(TimeDistributed(Flatten()))
    CNNLSTM_model.add(LSTM(50, activation='relu', return_sequences=True))
    CNNLSTM_model.add(LSTM(50, activation='relu'))
    CNNLSTM_model.add(Dropout(0.2))
    CNNLSTM_model.add(Dense(1))
    CNNLSTM_model.compile(optimizer='adam', loss='mse')

    CNNLSTM_model.fit(X_train_cnnlstm, y_train, validation_data=(X_validate_cnnlstm, y_validate), epochs=epochs,
                      callbacks=[callback_early_stopping])

    predicted_y_cnnlstm = CNNLSTM_model.predict(X_test_cnnlstm)
    predicted_y_cnnlstm = np.append(np.repeat(np.nan, 48), predicted_y_cnnlstm)
    test_set_scaled[:, -1] = predicted_y_cnnlstm.reshape(-1)
    test_set_scaled_cnnlstm = sc.inverse_transform(test_set_scaled)
    predicted_y_cnnlstm = test_set_scaled_cnnlstm[:, -1].reshape(-1, 1)
    predicted_y_cnnlstm = predicted_y_cnnlstm[48:]
    print(f'RMSE CNN-LSTM: {cu.rmse(actual_y, predicted_y_cnnlstm)}')
    print(f'MAPE CNN-LSTM: {cu.mape(actual_y, predicted_y_cnnlstm)} (%)')
    print(f'CV CNN-LSTM: {variation(predicted_y_cnnlstm)[0] * 100} (%)')
    # print(f'CV CNN-LSTM: {cu.cv(predicted_y_cnnlstm)} (%)')
    predicted_y_cnnlstm = np.append(np.repeat(np.nan, LAGS), predicted_y_cnnlstm)
    predicted_y_cnnlstm = pd.DataFrame(predicted_y_cnnlstm, columns=['energy'])
    predicted_y_cnnlstm.index = y_test_df.index

    # plot
    fig = plt.figure(figsize=[8, 3])
    ax = fig.add_subplot(111)
    plt.plot(y_test_df.index, y_test_df, color='k', linewidth=1.5)
    plt.plot(predicted_y_svm.index, predicted_y_svm, color='g', linewidth=1)
    plt.ylabel('Electricity Usage (kWh)', fontsize=11)
    plt.xlim([y_test_df.index[LAGS], y_test_df.index[-1]])
    plt.legend([f'Actual - house {num}', 'SVM'], loc='best', fontsize=10)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    plt.savefig(f'plots/{PLOT_DIR}/multivariate_timeseries_svm_{house}.png')

    fig = plt.figure(figsize=[8, 3])
    ax = fig.add_subplot(111)
    plt.plot(y_test_df.index, y_test_df, color='k', linewidth=1.5)
    plt.plot(predicted_y_ann.index, predicted_y_ann, color='y', linewidth=1)
    plt.ylabel('Electricity Usage (kWh)', fontsize=11)
    plt.xlim([y_test_df.index[LAGS], y_test_df.index[-1]])
    plt.legend([f'Actual - house {num}', 'ANN'], loc='best', fontsize=10)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    plt.savefig(f'plots/{PLOT_DIR}/multivariate_timeseries_ann_{house}.png')

    fig = plt.figure(figsize=[8, 3])
    ax = fig.add_subplot(111)
    plt.plot(y_test_df.index, y_test_df, color='k', linewidth=1.5)
    plt.plot(predicted_y_cnn.index, predicted_y_cnn, color='b', linewidth=1)
    plt.ylabel('Electricity Usage (kWh)', fontsize=11)
    plt.xlim([y_test_df.index[LAGS], y_test_df.index[-1]])
    plt.legend([f'Actual - house {num}', 'CNN'], loc='best', fontsize=10)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    plt.savefig(f'plots/{PLOT_DIR}/multivariate_timeseries_cnn_{house}.png')

    fig = plt.figure(figsize=[8, 3])
    ax = fig.add_subplot(111)
    plt.plot(y_test_df.index, y_test_df, color='k', linewidth=1.5)
    plt.plot(predicted_y_lstm.index, predicted_y_lstm, color='r', linewidth=1)
    plt.ylabel('Electricity Usage (kWh)', fontsize=11)
    plt.xlim([y_test_df.index[LAGS], y_test_df.index[-1]])
    plt.legend([f'Actual - house {num}', 'LSTM'], loc='best', fontsize=10)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    plt.savefig(f'plots/{PLOT_DIR}/multivariate_timeseries_lstm_{house}.png')

    fig = plt.figure(figsize=[8, 3])
    ax = fig.add_subplot(111)
    plt.plot(y_test_df.index, y_test_df, color='k', linewidth=1.5)
    plt.plot(predicted_y_cnnlstm.index, predicted_y_cnnlstm, color='m', linewidth=1)
    plt.ylabel('Electricity Usage (kWh)', fontsize=11)
    plt.xlim([y_test_df.index[LAGS], y_test_df.index[-1]])
    plt.legend([f'Actual - house {num}', 'CNN-LSTM'], loc='best', fontsize=10)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    plt.savefig(f'plots/{PLOT_DIR}/multivariate_timeseries_cnnlstm_{house}.png')

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    plt.plot(y_test_df.index, y_test_df, color='k', linewidth=1.5)
    plt.plot(predicted_y_svm.index, predicted_y_svm, color='g', marker='.', linewidth=0.7)
    plt.plot(predicted_y_ann.index, predicted_y_ann, color='y', marker='^', linewidth=0.7)
    plt.plot(predicted_y_cnn.index, predicted_y_cnn, color='b', marker='+', linewidth=0.7)
    plt.plot(predicted_y_lstm.index, predicted_y_lstm, color='r', marker='x', linewidth=0.7)
    plt.plot(predicted_y_cnnlstm.index, predicted_y_cnnlstm, color='m', marker='*', linewidth=0.7)
    plt.ylabel('Electricity Usage (kWh)', fontsize=11)
    plt.xlim([y_test_df.index[LAGS], y_test_df.index[-1]])
    plt.legend([f'Actual - house {num}', 'SVM', 'ANN', 'CNN', 'LSTM', 'CNN-LSTM'], loc='best', fontsize=10)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    plt.savefig(f'plots/{PLOT_DIR}/multivariate_timeseries_models_{house}.png')
    plt.close('all')