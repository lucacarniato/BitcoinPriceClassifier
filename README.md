# Bitcoin LSTM classifier with email notification


This project attempts to predict the next hour close price of bitcoin. In order to do so, it uses an LSTM network and scrapes the price data from Coinbase. 

Data preparation:

•	Downloads historical hourly open, high, low, close and volume price data from Coinbase, saves them to file.
•	Makes the series stationary by applying percent transformations.
•	Creates the sequences composed by all past and present data.
•	Creates three labels for the next closing price: 0 (sell action) if the close price is below the current price minus a percent margin, 1 (keep action) if the close price is within the margin and 2 (buy action) if the next price is above the current price plus the margin.

Model fitting:

•	Computes the 3-fold cross validation score, used in the hyperparameter tuning (see below)

Hyperparameter tuning:

•	By using [Optuna](https://github.com/optuna/optuna). In this case the length of the sequence and the dropout fractions are explored to maximize the cross-validation score.

Live predictions

•	Retrieves hourly data from Coinbase and notifies the subscriber(s) by email when the price is predicted to change more than the prescribed margin.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
