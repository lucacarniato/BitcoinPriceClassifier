# Bitcoin LSTM classifier with email notification

Downloads historical hourly OHLCV data from Coinbase

Prepares test and train data for a LSTM classifier

Trains the model, output are 2 softmax units, the first gives the probability of an higher price in the next hour and the second the probability of a lower price

Sends a notification email buy/sell
