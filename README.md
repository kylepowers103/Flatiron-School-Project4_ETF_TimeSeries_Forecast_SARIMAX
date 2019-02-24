# Flatiron-School-Project4_ETF_TimeSeries_Forecast_SARIMAX

# ETF/STOCK TIME-SERIES FORECASTING

#### Our Objective

The objective of our project is neither to build
a system that makes billions nor to waste billions
But the objective is to develop a system that finds
the direction of change of ETF/STOCK prices based
on the co-relations between STOCK prices and help
the investors in the stock market in taking
a decision whether to buy/sell/hold a ETF or stock by
providing the results in-terms of visualizations


### THE PROCESS

Time series
![png](/readmephotos/process.png)


### CHECKING FOR STATIONARITY
![png](/readmephotos/2.png)


Results –Not so good


### Transformation: ELIMINATING TREND AND SEASONALITY
Results of Dickey-Fuller Test:
![png](/readmephotos/3.png)


### Transformation: EXPONENTIAL MOVING AVERAGE 

Results of Dickey-Fuller Test:
![png](/readmephotos/4.png)
![png](/readmephotos/5.png)


### Transformation: LOG MOVING AVERAGE DIFF

Results of Dickey-Fuller Test:

![png](/readmephotos/6.png)
![png](/readmephotos/7.png)

### Transformation: DECOMPOSITION

Results of Dickey-Fuller Test:

![png](/readmephotos/8.png)
![png](/readmephotos/9.png)

### Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) to IDENTIFY the AR or MA terms in an ARIMA model

plots
![png](/readmephotos/10.png)

### ARIMA MODEL Using ACF and PACF Terms from Above
![png](/readmephotos/11.png)

### Resampled Data
![png](/readmephotos/12.png)

### Model Error Formula
![png](/readmephotos/13.png)

### Forecast EFA Closing Prices and Monthly and Weekly Resample Forecasts
![png](/readmephotos/14.png)


### Other Stock and ETF REDICTIONS

■ Using resampled weekly Data
forecast model

■ Showing Upper and Lower
confidence levels
![png](/readmephotos/15.png)

### Actual Price Comparison
![png](/readmephotos/17.png)


### FUTURE IMPROVEMENTS

■ Add more data from other sources

■ Event driven analysis

■ Extra visualization for close price

■ Recurrent neural network with LSTM

■ Add more variables to predict the adjusted close price


## Thank you

■ Kyle Powers
