<H1 ALIGN =CENTER> Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL...</H1>

### Date: 

### AIM :

To implement SARIMA model using python.

### ALGORITHM :

#### Step 1 :

Explore the dataset.

#### Step 2 :

Check for stationarity of time series.

#### Step 3 :

Determine SARIMA models parameters p, q.

#### Step 4 :

Fit the SARIMA model.

#### Step 5 :

Make time series predictions and Auto-fit the SARIMA model.

#### Step 6 :

Evaluate model predictions.

### PROGRAM :

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Temperature.csv')
data['date'] = pd.to_datetime(data['date'])

plt.plot(data['date'], data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['temp'])

plot_acf(data['temp'])
plt.show()
plot_pacf(data['temp'])
plt.show()

sarima_model = SARIMAX(data['temp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['temp'][:train_size], data['temp'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```

### OUTPUT :
![m6](https://github.com/Vishwarathinam/TSA_EXP10/assets/95266350/a7fc7cb4-1188-453d-a5d6-92d8b33862a0)

![m7](https://github.com/Vishwarathinam/TSA_EXP10/assets/95266350/3c11420a-f2bd-477f-94b7-4850a6fba4f4)

![m8](https://github.com/Vishwarathinam/TSA_EXP10/assets/95266350/e6ee8782-6305-4840-b68d-46b8d5dab530)

![m9](https://github.com/Vishwarathinam/TSA_EXP10/assets/95266350/91552054-f9b3-472a-9ffa-9bd69c40e60f)



### RESULT :

Thus, the program had ran successfully based on the SARIMA model using python.
