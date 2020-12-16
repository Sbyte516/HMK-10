# HW-10

### TIME_SERIES_ANALYSIS ###

![](https://www.quest.com/community/cfs-filesystemfile/__key/communityserver-components-secureimagefileviewer/communityserver-blogs-components-weblogfiles-00-00-00-00-40/BlogPost_2D00_MPM_2D00_2021Predictions_2D00_1100x500_2D00_US_2D00_LR_2D00_63582.jpg_2D00_1100x500x2.jpg?_=637405488919482545)








Import all depenancies, import the Yen-dollar exchange rate, this is a continuous chain of the futures 
data. The idea is to apply auto-regressive modeling and time series analysis using various methods 
that calculate future values for the yen and predicts future behavior based on past behavior 
using X & Y values that forecast our returns over the next 5 day period. 

After importing the file slice the data to include the particular dates we need for our dataset.

Plot the Settle returns, import adfuller to determine if the data is stationary or non-stationary, observe the p
value to determine the stationarity. Our data is non-stationary, as the p value is above .05, therefore we cannot reject
the null hypothesis.

I've choosen to import autocorrelation, and partial correlation graphic models. The auto-correlation is used to estimate
the number of orders for the MA (moving average). This is trying to quantify the error, past errors plus current errors, 
used to predict future values. 

AR Formula : yt = u + a1yt-1 + Et

The value at time y (yt) is = to (u) which is the avg/mean of the data + coefficient (a1), multiplied by (yt - 1)
* If y = today then (t - 1) is yesterday, if y is this year then (t -1) is last year, if y = 12 oclock then (t-1) is 11.
In Auto regression the value of a future value is dependant on past values
(+ Et) is going to be epsilon, Et is our error or noise over time, in the natural world things are not absolute there are 
variences in which we are accounting for, natural variance.  

Partial Autocorrelation gives us the auto-regressive portion of our data, this pcf gives us the number of lags present
these, specific lagged values are used as predictor variables. Lags are where results from one time period affect following 
periods.  
 

The partial Auto-Correlation can show on a vizual scale how many orders are
present in our model. 

An AR model can have more than one order/ number of lags at one time. 

Applying the Hodrick-Prescott Filter Decompose tool to our Yen exchange rate 
outputs two separate data columns, one containing the trend component, the other 
with Nosie component. 

Using the Settle, trend, noise outputs, I've created a Dataframe,
plot the settle vs trend data, and the noise component on a seperate 
graph. 

I've created a series using the settle price and the percent change function. 
converted the -inf to nan and dropped all na's. 

In Forecasting our returns we will use an ARMA model and fit it to our data, set the parameters (p,q) p=2,q=1, 
and give the order =(2,1). In determining how well the model fits that data we will consider the P value (p>.05).

Import the statsmodels packages, create a variable call on the ARMA function, apply it to the returns.values, set the 
order, describe the number of AR lags present, and the MA moving average component based on the order. 
store the results inside a varaiable that contains the fit model function, and output model summary data. 

The model gives us a number for each lag components, MA component, as well as the coefficients, standard deviation and P 
values for each component described. 

All p values presented in our data are greater than .05, this model is not a good fit for the data.

Next I've placed the results inside a DF, ran the forecast function provided the (steps) 
initialized hvplot and gave a title. Output a forecast for our returns over the next 5 day period.

Based on the forecast the returns are negative with minimal increase over the 5 day period.


Using the ARIMA model will be applied to our data set next, this is just like the ARMA model, the difference
being it takes 3 inputs rather than 2 when given an order. The reason being this basically assumes the percent
change function without having to go through the process manually. This automatically does the stationary to non-
stationary conversion.

Autoregressive Intergrated Moving Average (ARIMA) 
Basically the same as ARMA
* Combines features of AR and MA models
* Past values and erros are used to predict future values.
* This takes 3 inputs (Ar order, Diff, MA order) (#, 1, #)
The Diff = % change, 1 = same as % change, does the non-stationary to stationary coversion.
having a 2 would indicate take the second value and apply the % change etc. (#, 2, #)

* ΔYt = (mu) + a1ΔYt-1 + a2ΔYt-2 + Et 

A (usually small) change in value. Often shown using the "delta symbol": Δ Example: Δx means
 "the change in the value of x" or in this case Y.

Our Final model will assist in predicting the volatility
The GARCH model will be applied in the same fashion as the previous models, begininng with imports 
and calling on the model, then storing the model in a variable. Parameters are p=2,q=1 order=(2,1)
arch_model operates on returns, mean='ZERO" vol='GARCH', (p=2,q=1)
fit to the model and place inside a variable. 
Return a summary.

Based on the p value there is too much volatility present to determine if the modle is a good fit for the data.

We will find the last day of the data set by creating a last_day variable and place the returns inside, use the 
index.max().strftime() on the returns to get the last value of the index.

The GARCH model data has provided us with quantitave data that describes how much volitility is present in our data. 
The data will be used to make a forecast that predicts volitility over a 5 day period. 

Use a forecast_horizon varaible set it equal to the number of days you want to predict.
Create another variable to store the forecast, apply the last_day variable and forecast_horizon variable, run the model,
and forecast operation on the last_day and forecast_horizon. 

Create an Annualized variable use the np.sqrt function operation on the forecast.varience. dropna functions, then mutliply by 252
initialize the data. 

Transpose the the columns h.1-h.5 columns into rows using the .T
(transpose) attached to the annulaized variable. 
Now we can call on a data series that contains the annualized values. 
Use the analized data to lot the forecast. 

Model predicts volatility will increase over the next 5 days. 



### Regression Analysis ###

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/38_blog_image_1.png)



Import dependencies, bring in the Yen_dollar exchanage rate, slice the data.
Create a DF add the Returns, add a column for lagged returns. Shift the data down one row.
This sets up each column side by side with the Lagged Returns and the Return. 

The idea is to have the Lagged_Returns data predict the Return data, utilizing a type of machine 
learning, called supervised learning. Supervised learning consists in learning the link between two data sets: 
observed data X and an external variable Y which we would like to predict, tyically called "target" or "lables". Most often 
y is a 1D array of length n_samples.

After creating our DataFrame with the Lagged Returns, split the data in two sets, one for training the data
and one for testing the data. Create a train vairable that holds our trained data, and we will retrieve all dates 
through 2017, this will be considered the test period, this is the period we will be training our data on. Afterwards we 
will use the training data to create a model, which we then use on our test data to determine how well this model predicts 
future values. 

It's time to create 4 DataFrames two of these will contain the Lagged_returns for the X_train 
and X_test variables, and the remianing two will have the Returns and will be stored in our Y_test and Y_train 
vaiables. 

After we've created our separate DataFrames we'll apply the Linear Regression Model to the X_train 
and Y_train data, fit the model to the training data, this step allows us to train the modle on the data that we have 
provided and also creates the modle we will be using for our comparison.

I've made a predictions varaiable that combine our linear regression model with the predict(x) fuction, applies this 
to the X_test data and uses that data to make predictions for the Y_test data. 

In the next step take the actual Y_test data and compare it to our predictions in a DataFrame. 
Using a variable called Results we will store our actuals and our predictions and compare them inside a DF. 

We are now able to plot the results for the first 20 predictions. 

Following, evaluate the model using "out-sample_data".
"Out-of-sample data is data that the model hasn't seen before (Testing data)".
Import mertics from sklearn as before import mean_squared_error(mse)
Call on the (mse) operation. Apply the (mse) to the Results of the 
Y actuals and Y predictions. Calculate the root_mean_squared_error using the (mse).

Out_Of_Sample Root Mean Squared Error (RMSE): 0.41545437184712763

Evaluate the model using in_sample_data.
In_sample_data is data that the model was trained on (Training data).
I've created a new varaible called in_sample_results to store our Y_train
data/ target data. Then I'll create a column with our in_sample_predictions 
and use this in_sample_predictions to make and store our Y predictions after 
using the model.predict fuction that uses our X_train data to predict our Y_train 
values. Our in_sample_resluts will have the Returns and new Y predictions.
Again we will calculate the mse as before, for the comparison to the out_of_sample
data. Use the (mse) on the in_sample_results that have the Return and Prediction data.
Calculate the rmse using the mse and apply the sqaure root function on the in_sample_mse.

In-sample root mean squared error (RMSE): 0.5962037920929946

Results indicate of the in-sample/ out-of-sample data,
The model is better suited to work with out of sample data.

![](https://hackaday.com/wp-content/uploads/2018/05/main4501.png?w=800)
