# Bitcoin LSTM classifier with email notification

Downloads historical hourly OHLCV data from Coinbase

Prepares test and train data for a LSTM classifier

Trains the model, output are 2 softmax units, the first gives the probability of an higher price in the next hour and the second the probability of a lower price

Sends a notification email buy/sell


Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 119, 128)          68608     
_________________________________________________________________
dropout (Dropout)            (None, 119, 128)          0         
_________________________________________________________________
batch_normalization (BatchNo (None, 119, 128)          512       
_________________________________________________________________
lstm_1 (LSTM)                (None, 119, 128)          131584    
_________________________________________________________________
dropout_1 (Dropout)          (None, 119, 128)          0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 119, 128)          512       
_________________________________________________________________
lstm_2 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 128)               512       
_________________________________________________________________
dense (Dense)                (None, 32)                4128      
_________________________________________________________________
dropout_3 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 99        
=================================================================
Total params: 337,539
Trainable params: 336,771
Non-trainable params: 768
_________________________________________________________________