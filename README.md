## DEVELOPED BY: Sri Varshan P
## REGISTER NO: 212222240104
## DATE: 
# EX.NO.09        A project on Time series analysis on forecasting using ARIMA model 

### AIM:
To Create a project on Time series analysis on forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of Raw_sales. 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error



def arima_model(data, target_variable, order):
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the ARIMA model on the training data
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast for the length of the test data
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='red')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='green')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()



data = pd.read_csv('russia_losses_equipment.csv')


data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)


arima_model(data, 'tank', order=(5,1,0))

```
### OUTPUT:

![image](https://github.com/user-attachments/assets/4c48f067-9f0a-4af7-98b8-8fd56b1e49f6)


### RESULT:
Thus, the program based on the ARIMA model using python is executed successfully.
