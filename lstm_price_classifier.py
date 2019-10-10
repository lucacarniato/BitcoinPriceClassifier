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
VALIDATION_PERCENT = 0.20
PRICE_HEADERS = ['Open', 'High', 'Low', 'Close', 'VolumeFrom']
PRICE_DATA="training_datas/coinbase-1h-btc-eur-api.csv"
SENTIMENT_DATA="training_datas/sentiment.csv"
SENTIMENT_DATA_TOTAL = None #"training_datas/sentiment_total.csv"
SEQUENCE_LEN = int(45.6193325687566)
NUM_NEURONS = 128
BATCH_SIZE = 128
EPOCHS = 5  # 20 how many passes through our data
SAVE_MODEL = True
SENTIMENT_HEADERS =[]
VERBOSE = False

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
        'Dropout_01': trial.suggest_loguniform('Dropout_01', 0.05, 0.5),
        'Dropout_02': trial.suggest_loguniform('Dropout_02', 0.05, 0.5),
        'Dropout_03': trial.suggest_loguniform('Dropout_03', 0.05, 0.5),
        'Dropout_04': trial.suggest_loguniform('Dropout_04', 0.05, 0.5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.01),
        'decay_adam': trial.suggest_loguniform('decay_adam', 1e-8, 1e-4),
        'sequence_len': int(trial.suggest_int('sequence_len', 6, 24 * 3)),
    }

def run_base_model(**parameters):
    parameters['Dropout_01'] = 0.2
    parameters['Dropout_02'] = 0.1
    parameters['Dropout_03'] = 0.2
    parameters['Dropout_04'] = 0.2
    parameters['learning_rate'] = 0.00247848126819933
    parameters['decay_adam'] = 5.87529568055265e-05
    parameters['sequence_len'] = int(45.6193325687566)
    build_model(**parameters)

def optimize_model(trial):
    parameters = model_parameter_distributions(trial)
    return build_model(**parameters)


########################################################################################################################
# File IO
########################################################################################################################

def get_historical_data():
    import cbpro
    import time
    from datetime import datetime
    from calendar import timegm

    public_client = cbpro.PublicClient()
    start_date_his = '2019-07-14T21:00:00.0' #2019-07-14T21:00:00Z
    end_date_his = '2019-07-27T12:00:00.0'

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
    df.to_csv('training_datas/last_chunk.csv')

def build_test_validation_df(price_data=PRICE_DATA, sentiment_data=None):

    main_df = pd.read_csv(price_data)
    main_df = main_df.sort_values(by=['Date'])
    main_df.set_index("Date", inplace=True)
    main_df = main_df[PRICE_HEADERS]
    main_df.fillna(method="ffill", inplace=True)
    main_df.dropna(inplace=True)

    # drop 0 volume
    main_df = main_df[main_df.VolumeFrom != 0]
    #main_df = main_df.tail(10000)

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
        #main_df.to_csv('training_datas/tests/result.csv')

    main_df['future'] = main_df['Close'].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify_sb, main_df['Close'], main_df['future']))
    main_df.dropna(inplace=True)

    ## here, split away some slice of the future data from the main main_df.
    times = sorted(main_df.index.values)
    last = sorted(main_df.index.values)[-int(VALIDATION_PERCENT * len(times))]

    validation_main_df = main_df[(main_df.index >= last)]
    main_df = main_df[(main_df.index < last)]

    return main_df, validation_main_df

########################################################################################################################

def classify_sb(current, future):
    if float(future) < float(current):
        return 0  # sell
    elif float(future) > float(current):
        return 1  # buy

def prepare_sequential_data(main_df, sequence_len):
    df = main_df.copy(deep=True)
    df = df.drop("future", 1)  # don't need this anymore.
    # print(df.head())

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
    close_backup =[]
    prev_days = deque(maxlen=sequence_len)

    for i in df.values:
        prev_days.append([n for n in i[:-2]]) #i[:-1]
        if len(prev_days) == sequence_len:
            sequential_data.append([preprocessing.scale(np.array(prev_days)), i[-2]])
            close_backup.append(i[-1])

    return sequential_data, close_backup


def process_sb_df(df, sequence_len):
    sequential_data, close_backup = prepare_sequential_data(df, sequence_len)
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:  # sell
            sells.append([seq, target])
        elif target == 1:  # buy
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = sells + buys
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def test_model(sequence_len, test_type="val",model_path=MODEL_PATH):

    main_df, validation_main_df = build_test_validation_df()
    print('load model from ', model_path)
    loaded_model = tf.keras.models.load_model(model_path)

    # cross plot
    if test_type=="val":
        sequential_data, close_backup = prepare_sequential_data(validation_main_df, sequence_len)
    if test_type == "train":
        sequential_data, close_backup = prepare_sequential_data(main_df, sequence_len)

    X_sequential = []
    y_sequential = []
    for seq, target in sequential_data:
        X_sequential.append(seq)
        y_sequential.append(target)

    single_input = np.array(X_sequential)
    yhat = loaded_model.predict(single_input, verbose=0)

    yhat_labels = []
    correct = 0
    long = False
    buy_price = -1.0
    profit = 0.0
    increment_parameter = 0.0
    buy_threshold = 0.50
    sell_threshold = 0.50
    fee = 0.25/100.0
    eur_capital = 1000.0
    btc_holds = 0.0
    for i in range(0, len(yhat)):

        mp = max(yhat[i])
        buy = yhat[i][0]
        sell = yhat[i][1]

        # add number of cases
        if buy == mp:
            yhat_labels.append(1)
            if y_sequential[i] == 1:
                correct += 1

        elif sell == mp:
            yhat_labels.append(0)
            if y_sequential[i] == 0:
                correct += 1

        if i == 0:
            btc_holding = eur_capital/(close_backup[i]*(1.0+fee))
            btc_holds = eur_capital/(close_backup[i]*(1.0+fee))
            buy_price = close_backup[i]
            eur_capital = 0.0
            long = True

        # simulate trading
        if not long and buy > buy_threshold:
            btc_holds = eur_capital/(close_backup[i]*(1.0+fee))
            buy_price = close_backup[i]
            eur_capital = 0.0
            long = True

        if long and sell > sell_threshold and close_backup[i] > (buy_price * 1.01):
            print("profit event :",  (close_backup[i]-buy_price)/buy_price * 100.0, " btc holds ",btc_holds)
            eur_capital = btc_holds * close_backup[i] * (1.0 - fee)
            long = False

    import matplotlib.pyplot as plt
    # plt.plot(yhat_labels, y_sequential, 'o', color='black')
    print("correct percentage: ", correct / len(yhat))
    print("The capitalization is: ", btc_holds * close_backup[-1] + eur_capital, " is long?: ", long)
    print("The hodl value  is: ", close_backup[-1] * btc_holding, " with n ", btc_holding, " initial btc")

def build_model(**parameters):

    Dropout_01 = parameters['Dropout_01']
    Dropout_02 = parameters['Dropout_02']
    Dropout_03 = parameters['Dropout_03']
    Dropout_04 = parameters['Dropout_04']
    learning_rate = parameters['learning_rate']
    sequence_len = parameters['sequence_len']
    decay_adam = parameters['decay_adam']

    main_df, validation_main_df = build_test_validation_df()

    if VERBOSE:
        (main_df.head())

    train_x, train_y = process_sb_df(main_df, sequence_len)
    validation_x, validation_y = process_sb_df(validation_main_df, sequence_len)

    if VERBOSE:
        print(f"train data: {len(train_x)} validation: {len(validation_x)}")
        print(f"Sells: {train_y.count(0)}, holds: {train_y.count(1)}, buys: {train_y.count(2)}")
        print(f"VALIDATION  {validation_y.count(0)}, holds: {validation_y.count(1)}, buys: {validation_y.count(2)}")

    model = Sequential()
    model.add(LSTM(NUM_NEURONS, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(Dropout_01))
    model.add(BatchNormalization())

    model.add(LSTM(NUM_NEURONS, return_sequences=True))
    model.add(Dropout(Dropout_02))
    model.add(BatchNormalization())

    model.add(LSTM(NUM_NEURONS))
    model.add(Dropout(Dropout_03))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(Dropout_04))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_adam)  # lr=0.001, decay=1e-6

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # TensorBoard(log_dir="logs\{}".format(NAME))
    tensorboard = None
    filepath = "RNN_Final-{epoch:02d}"
    checkpoint = ModelCheckpoint(
        "models\{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

    # Train model
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=(validation_x, validation_y))  # callbacks=[tensorboard,checkpoint]

    # Score model
    length_valid = int(len(validation_x)/2)
    score = model.evaluate(validation_x[:length_valid], validation_y[:length_valid], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if SAVE_MODEL:
        model.save(MODEL_PATH)

    return -score[1]

def update_csv_data():
    from update_data import download_data_async
    download_data_async()


def predict_next(sequence_len, loaded_model=None):
    import cbpro
    from datetime import datetime
    from calendar import timegm
    import time

    granularity = 3600
    end_epoch = timegm(datetime.now().timetuple())
    start_epoch = end_epoch - (granularity * sequence_len * 2)

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
    sequential_data, close_backup = prepare_sequential_data(main_df, sequence_len)
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

def cycle_predictions(sequence_len=SEQUENCE_LEN):
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
    previus_action = -1
    previus_price = -1
    long = False
    buy_price = 0

    while (1):
        current_time = datetime.now().timetuple()
        if current_time.tm_min == 0 : #check every hour
            date, current_price, buy, sell = predict_next(sequence_len, loaded_model)
            correct = 0
            if current_price > previus_price and previus_action == 1:
                correct = 1
            if current_price < previus_price and previus_action == 0:
                correct = 1
            row = str("date " + date + " current_price " + str(current_price)
                      + " buy " + str(buy) + " sell " + str(sell) + " correct " + str(correct) + "\n")
            log_file.write(row)
            log_file.flush()
            results.append([date, current_price, buy, sell, correct])

            previus_price = current_price
            if sell > buy:
                previus_action = 0
            else:
                previus_action = 1

            if not long and buy > 0.50:
                text = "buy: " + str(buy) + ' ' + str(sell) + ' ' + str(current_price)
                long = True
                sendEmail("btcpred", text, e, p)
                buy_price = current_price
            if long and buy < 0.50 : #and current_price > (buy_price * 1.01):
                text = "sell: " + str(buy) + ' ' + str(sell) + ' ' + str(current_price)
                long = False
                sendEmail("btcpred", text, e, p)

        print('sleeping...')
        time.sleep(60)

    log_file.close()


if __name__ == '__main__':
    # run_base_model()
    # optuna_optimization()
    # run_base_model()
    cycle_predictions()
