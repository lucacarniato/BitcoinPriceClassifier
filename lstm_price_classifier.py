# keras accuracy and loss functions
# https://stackoverflow.com/questions/45632549/keras-accuracy-for-my-model-always-0-when-training
import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller
import optuna

MODEL_PATH = "models/model"
FUTURE_PERIOD_PREDICT = 1
TEST_PERCENT = 0.10
VALIDATION_PERCENT = 0.10
PRICE_HEADERS = ['Open', 'High', 'Low', 'Close', 'VolumeFrom']
PRICE_DATA = "training_datas/coinbase-1h-btc-eur-api.csv"
SENTIMENT_DATA = "training_datas/sentiment.csv"
SENTIMENT_DATA_TOTAL = None  # "training_datas/sentiment_total.csv"
SENTIMENT_HEADERS = []
RUN_WITH_BEST_PARAMETERS = False


SEQUENCE_LEN = 24*10 #119
NUM_UNITS = 128
VERBOSE = False
MARGIN_PERCENT = 0.0055
PROBABILITY_THRESHOLD = 0.51  # 0.5


########################################################################################################################
# optimization
########################################################################################################################
def optuna_optimization():
    n_trials = 50
    n_jobs = 1
    optuna_study = optuna.create_study(study_name='classification_optimization', storage='sqlite:///params.sqlite',
                                       load_if_exists=True)
    optuna_study.optimize(optimize_model, n_trials=n_trials, n_jobs=n_jobs)

def model_parameter_distributions(trial):
    return {
        'dropout_01': trial.suggest_loguniform('dropout_01', 0.05, 0.5),
        'dropout_02': trial.suggest_loguniform('dropout_02', 0.05, 0.5),
        'sequence_len': int(trial.suggest_int('sequence_len', 6, 24 * 10))
    }


def run_base_model(**parameters):
    parameters['sequence_len'] = SEQUENCE_LEN
    parameters['dropout_01'] = 0.2
    parameters['dropout_02'] = 0.1
    build_model(**parameters)


def optimize_model(trial):
    parameters = model_parameter_distributions(trial)
    return build_model(**parameters)


########################################################################################################################
# Data preparation
########################################################################################################################

def get_historical_data():
    import cbpro
    import time
    from datetime import datetime
    from calendar import timegm

    public_client = cbpro.PublicClient()
    start_date_his = '2019-07-14T21:00:00.0'  # 2019-07-14T21:00:00Z
    end_date_his = '2019-10-10T12:00:00.0'

    granularity = 3600
    num_data_call = 300
    end_date_his_epoch = timegm(datetime.strptime(end_date_his, '%Y-%m-%dT%H:%M:%S.%f').timetuple())

    all_data = []
    start_epoch = timegm(datetime.strptime(start_date_his, '%Y-%m-%dT%H:%M:%S.%f').timetuple())
    end_epoch = start_epoch + granularity * (num_data_call - 1)

    last_call = False
    while (1):

        start_struct_time = time.gmtime(start_epoch)
        end_struct_time = time.gmtime(end_epoch)

        start = time.strftime('%Y-%m-%dT%H:%M:%SZ', start_struct_time)
        end = time.strftime('%Y-%m-%dT%H:%M:%SZ', end_struct_time)

        while (True):
            try:
                print('retriving from coinbase from ', start, ' to ', end)
                rates = public_client.get_product_historic_rates('BTC-EUR', granularity=3600, start=start, end=end)
                time.sleep(0.1)
                rates = rates[::-1]
                all_data = all_data + rates
                break
            except:
                print('exception, retrying...')
                time.sleep(2.0)

        if last_call:
            break

        start_epoch = end_epoch + granularity
        if (start_epoch + granularity * (num_data_call - 1)) >= end_date_his_epoch:
            end_epoch = end_date_his_epoch
            last_call = True
        else:
            end_epoch = start_epoch + granularity * (num_data_call - 1)

    headers = ['Date', 'Low', 'High', 'Open', 'Close', 'VolumeFrom']
    df = pd.DataFrame(all_data[::-1], columns=headers)

    df['Date'] = df['Date'].apply(lambda d: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(d)))

    df.set_index("Date", inplace=True)
    df.to_csv(PRICE_DATA)


def build_test_validation_df(price_data=PRICE_DATA, sentiment_data=None):
    main_df = pd.read_csv(price_data)
    main_df = main_df.sort_values(by=['Date'])
    main_df.set_index("Date", inplace=True)
    main_df = main_df[PRICE_HEADERS]
    main_df.fillna(method="ffill", inplace=True)
    main_df.dropna(inplace=True)

    # drop 0 volume
    main_df = main_df[main_df.VolumeFrom != 0]
    # main_df = main_df.tail(10000)

    # Append sentiment data
    sentiment_data = SENTIMENT_DATA_TOTAL
    if sentiment_data:
        sentiment_df = pd.read_csv(sentiment_data)
        SENTIMENT_HEADERS = list(sentiment_df.columns.values)
        SENTIMENT_HEADERS.remove('Date')
        sentiment_df = sentiment_df.sort_values(by=['Date'])
        sentiment_df.set_index("Date", inplace=True)
        sentiment_df = sentiment_df[SENTIMENT_HEADERS]
        sentiment_df.fillna(method="ffill", inplace=True)
        sentiment_df.dropna(inplace=True)

        main_df = main_df[~main_df.isin(sentiment_df)].dropna()

        # import matplotlib.pyplot as plt
        # plt.plot(sentiment_df['SentimentA'],sentiment_df['SentimentB'])

        result = pd.concat([main_df, sentiment_df], axis=1, sort=False)
        main_df = result
        main_df.dropna(inplace=True)

        print(main_df.head())
        # main_df.to_csv('training_datas/tests/result.csv')

    main_df['future'] = main_df['Close'].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify_sb, main_df['Close'], main_df['future']))
    main_df.dropna(inplace=True)

    dates = sorted(main_df.index.values)
    dev_split = dates[-int((TEST_PERCENT + VALIDATION_PERCENT) * len(dates))]
    val_split = dates[-int(VALIDATION_PERCENT * len(dates))]

    dev_df = main_df[(main_df.index < dev_split)]
    test_df = main_df[(dev_split <= main_df.index) & (main_df.index < val_split)]
    val_df = main_df[(main_df.index >= val_split)]

    print("dev : ", len(dev_df), ", test :", len(test_df), ", valid :", len(val_df))

    return dev_df, test_df, val_df


def classify_sb(current, future):
    if float(future) < float(current * (1.0 - MARGIN_PERCENT)):
        return 0  # sell
    elif float(future) > float(current * (1.0 + MARGIN_PERCENT)):
        return 2  # buy
    else:
        return 1  # keep


def prepare_sequential_data(main_df, sequence_len):
    df = main_df.copy(deep=True)
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:
        if col != "target":  #
            if col in PRICE_HEADERS:
                if col == 'Close':
                    df['Close_backup'] = df[col]

                df[col] = df[col].pct_change()

            if col not in PRICE_HEADERS:
                if VERBOSE:
                    # check for stationarity
                    # https://machinelearningmastery.com/time-series-data-stationary-python/
                    diffed_result = adfuller(df[col].values[1:], autolag="AIC")
                    print('ADF Statistic: %f' % diffed_result[0])
                    print('p-value: %f' % diffed_result[1])
                    print('Critical Values:')
                    for key, value in diffed_result[4].items():
                        print('\t%s: %.3f' % (key, value))

    df.dropna(inplace=True)

    sequential_data = []
    close_backup = []
    prev_days = deque(maxlen=sequence_len)

    print("column used: ", df.columns[:-2])
    print("target column: ", df.columns[-2])

    for i in df.values:
        prev_days.append([n for n in i[:-2]])
        if len(prev_days) == sequence_len:
            sequential_data.append([preprocessing.scale(np.array(prev_days)), i[-2]])
            close_backup.append(i[-1])

    return sequential_data, close_backup


########################################################################################################################
# MODEL
########################################################################################################################

def process_sb_df(df, sequence_len):
    sequential_data, close_backup = prepare_sequential_data(df, sequence_len)
    random.shuffle(sequential_data)

    sells = []
    keeps = []
    buys = []
    for seq, target in sequential_data:
        if target == 0:  # sell
            sells.append([seq, target])
        elif target == 1:  # keep
            keeps.append([seq, target])
        elif target == 2:  # buy
            buys.append([seq, target])

    random.shuffle(sells)
    random.shuffle(keeps)
    random.shuffle(buys)

    lower = min(len(sells), len(keeps), len(buys))

    sells = sells[:lower]
    keeps = keeps[:lower]
    buys = buys[:lower]

    sequential_data = sells + keeps + buys
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def create_model(input_shape,dropout_01,dropout_02):

    model = Sequential()
    model.add(LSTM(NUM_UNITS, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_01))
    model.add(BatchNormalization())

    model.add(LSTM(NUM_UNITS, return_sequences=True))
    model.add(Dropout(dropout_02))
    model.add(BatchNormalization())

    model.add(LSTM(NUM_UNITS))
    model.add(Dropout(dropout_01))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(3, activation='softmax'))

    if VERBOSE:
        model.summary()

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)  # lr=0.001, decay=1e-6

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def build_model(**parameters):
    sequence_len = parameters['sequence_len']
    dropout_01 = parameters['dropout_01']
    dropout_02 = parameters['dropout_02']
    dev_df, test_df, val_df = build_test_validation_df()

    if VERBOSE:
        (dev_df.head())

    train_x, train_y = process_sb_df(dev_df, sequence_len)
    test_x, test_y = process_sb_df(test_df, sequence_len)

    print("train sells: ", train_y.count(0), ", holds: ", train_y.count(1), ", buys: ", train_y.count(2))
    print("test sells: ", test_y.count(0), ", holds:", test_y.count(1), ", buys: ", test_y.count(2))
    print("sequence_len: ", sequence_len," ,dropout_01: ",dropout_01," ,dropout_02: ",dropout_02)

    input_shape = (train_x.shape[1:])
    # Train model
    from keras.wrappers.scikit_learn import KerasClassifier
    model = KerasClassifier(build_fn=create_model, input_shape=input_shape, dropout_01=dropout_01, dropout_02=dropout_02,
                            epochs=10, batch_size=128, verbose=1)

    score = -999.0
    if RUN_WITH_BEST_PARAMETERS:
        model.fit(train_x,train_y)
        model.save(MODEL_PATH)
    else:
        from sklearn.model_selection import cross_val_score
        score = np.mean(cross_val_score(model, train_x, train_y, cv=3))

    print("average score: ", score)

    return -score


########################################################################################################################
# TEST MODEL
########################################################################################################################

def test_model(data_type="val", probability=PROBABILITY_THRESHOLD, model_path=MODEL_PATH, model=None):

    print("data_type: ", data_type)
    dev_df, test_df, val_df = build_test_validation_df()

    if not model:
        print('load model from ', model_path)
        model = tf.keras.models.load_model(model_path)

    if data_type == "dev":
        sequential_data, close_backup = prepare_sequential_data(dev_df, SEQUENCE_LEN)
    if data_type == "test":
        sequential_data, close_backup = prepare_sequential_data(test_df, SEQUENCE_LEN)
    if data_type == "val":
        sequential_data, close_backup = prepare_sequential_data(val_df, SEQUENCE_LEN)

    X_sequential = []
    y_sequential = []
    for seq, target in sequential_data:
        X_sequential.append(seq)
        y_sequential.append(target)

    single_input = np.array(X_sequential)
    yhat = model.predict(single_input, verbose=0)

    yhat_labels = []
    correct = 0
    long = False
    buy_price = -1.0
    fee = 0.25 / 100.0
    eur_capital = 1000.0
    btc_holds = 0.0
    for i in range(0, len(yhat)):

        mp = max(yhat[i])
        sell = yhat[i][0]
        keep = yhat[i][1]
        buy = yhat[i][2]

        # add number of cases
        if sell == mp:
            yhat_labels.append(1)
            if y_sequential[i] == 0:
                correct += 1

        if keep == mp:
            yhat_labels.append(1)
            if y_sequential[i] == 1:
                correct += 1

        elif buy == mp:
            yhat_labels.append(0)
            if y_sequential[i] == 2:
                correct += 1

        if i == 0:
            print("initial price :", close_backup[0])
            btc_hodl = eur_capital / (close_backup[0] * (1.0 + fee))

        # simulate trading
        if not long and buy > probability:
            print("buy event, price :", close_backup[i], " ,sell ", sell, " ,buy ", buy)
            btc_holds = eur_capital / (close_backup[i] * (1.0 + fee))
            buy_price = close_backup[i]
            eur_capital = 0.0
            long = True

        elif long and sell > probability:
            print("sell event, price :", close_backup[i], " ,sell ", sell, " ,buy ", buy,
                  " ,profit :", (close_backup[i] - buy_price) / buy_price * 100.0, " ,btc holds: ", btc_holds)
            eur_capital = btc_holds * close_backup[i] * (1.0 - fee)
            long = False

    import matplotlib.pyplot as plt
    # plt.plot(yhat_labels, y_sequential, 'o', color='black')
    total_capitalization = btc_holds * close_backup[-1] + eur_capital
    print("correct percentage: ", correct / len(yhat))
    print("The capitalization is: ", total_capitalization, " is long?: ", long)
    print("The hodl value  is: ", close_backup[-1] * (1.0 - fee) * btc_hodl, " with n ", btc_hodl, " btc hodl")

    return total_capitalization


########################################################################################################################
# GO LIVE
########################################################################################################################

def cycle_predictions():
    from datetime import datetime
    import time
    import os

    pFile = open("pFile.txt", "r")
    e = pFile.readline()
    p = pFile.readline()
    pFile.close()
    os.remove("pFile.txt")
    sendEmail("btcpred", "starting", e, p)

    modelPath = "models/model"
    print('model path is', modelPath)
    loaded_model = tf.keras.models.load_model(modelPath)

    log_file = open("predictions.txt", "w+")
    results = []
    action = -999
    price = -999
    long = False
    buy_price = 0

    while (1):
        current_time = datetime.now().timetuple()
        if current_time.tm_min == 0:  # check every hour
            date, current_price, sell, keep, buy = predict_next(loaded_model)
            correct = 0
            if current_price > price * (1 + MARGIN_PERCENT) and action == 2:
                correct = 1
            if current_price < price * (1 - MARGIN_PERCENT) and action == 0:
                correct = 1
            if current_price <= price * (1 + MARGIN_PERCENT) and current_price >= price * (
                    1 - MARGIN_PERCENT) and action == 1:
                correct = 1
            row = str("date " + date + " current_price " + str(current_price)
                      + " buy " + str(buy) + " sell " + str(sell) + " correct " + str(correct) + "\n")
            log_file.write(row)
            log_file.flush()
            results.append([date, current_price, buy, sell, correct])

            mp = max(sell, keep, buy)
            price = current_price

            if sell == mp and sell > PROBABILITY_THRESHOLD:
                action = 0
            elif buy == mp and buy > PROBABILITY_THRESHOLD:
                action = 2
            else:
                action = 1

            if not long and action == 2:
                text = "buy: " + str(buy) + ' ' + str(sell) + ' ' + str(current_price)
                long = True
                sendEmail("btcpred", text, e, p)
                buy_price = current_price

            if long and action == 0:  # and current_price > (buy_price * 1.01):
                text = "sell: " + str(buy) + ' ' + str(sell) + ' ' + str(current_price)
                long = False
                sendEmail("btcpred", text, e, p)

        print('sleeping...')
        time.sleep(60)

    log_file.close()


def predict_next(loaded_model=None):
    import cbpro
    from datetime import datetime
    from calendar import timegm
    import time

    granularity = 3600
    end_epoch = timegm(datetime.now().timetuple())
    start_epoch = end_epoch - (granularity * SEQUENCE_LEN * 2)

    start_struct_time = time.gmtime(start_epoch)
    end_struct_time = time.gmtime(end_epoch)

    start = time.strftime('%Y-%m-%dT%H:%M:%SZ', start_struct_time)
    end = time.strftime('%Y-%m-%dT%H:%M:%SZ', end_struct_time)

    public_client = cbpro.PublicClient()
    count = 0
    while (count < 3):
        try:
            print('retriving from coinbase from ', start, ' to ', end)
            rates = public_client.get_product_historic_rates('BTC-EUR', granularity=3600, start=start, end=end)
            break
        except:
            print('exception, retrying...')
            time.sleep(2.0)
            count += 1

    # Headers
    headers = ['Date', 'Low', 'High', 'Open', 'Close', 'VolumeFrom']
    main_df = pd.DataFrame(rates, columns=headers)
    main_df = main_df.sort_values(by=['Date'])
    main_df['Date'] = main_df['Date'].apply(lambda d: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(d)))
    main_df.set_index("Date", inplace=True)

    main_df = main_df[['Open', 'High', 'Low', 'Close', 'VolumeFrom']]
    main_df.fillna(method="ffill", inplace=True)
    main_df.dropna(inplace=True)
    main_df = main_df[main_df.VolumeFrom != 0]

    # print(main_df.head())
    main_df['future'] = main_df['Close'].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify_sb, main_df['Close'], main_df['future']))

    # print some info on the rates

    last_date = main_df.index.tolist()[-1]
    last_close = main_df['Close'][-1]
    last_high = main_df['High'][-1]
    last_low = main_df['Low'][-1]
    print('last_date ', last_date, ' last close ', last_close,
          ' High ', last_high, ' Low ', last_low)

    # prepare data
    sequential_data, close_backup = prepare_sequential_data(main_df, SEQUENCE_LEN)
    X_sequential = []
    y_sequential = []
    for seq, target in sequential_data:
        X_sequential.append(seq)
        y_sequential.append(target)

    input = np.array(X_sequential)

    # load mode
    if not loaded_model:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)

    yhat = loaded_model.predict(input, verbose=0)

    # run classification
    # print('next hour: up', yhat[-1][0], ' next hour: down ', yhat[-1][1])

    return last_date, last_close, yhat[-1][0], yhat[-1][1]


def sendEmail(Subject, text, e, p):
    from email.mime.text import MIMEText
    import smtplib

    msg = MIMEText(text)
    msg['From'] = "info"
    msg['To'] = e
    msg['Subject'] = Subject

    server = smtplib.SMTP('smtp.gmail.com')
    server.starttls()
    server.login(e, p)

    server.sendmail(e, e, msg.as_string())
    server.quit()


if __name__ == '__main__':
    # run_base_model()
    optuna_optimization()
    # run_base_model()
    # cycle_predictions()
