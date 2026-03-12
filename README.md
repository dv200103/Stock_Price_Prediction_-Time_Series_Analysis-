**📈 **Stock Market Prediction Using Time Series Analysis****

This project demonstrates how Time Series Analysis can be used to model and analyze historical stock price data. Using the ARIMA (Autoregressive Integrated Moving Average) model, the project analyzes Apple stock price history and compares predicted values with actual prices.

Financial markets are highly dynamic and influenced by many factors, making accurate prediction difficult. However, statistical time series techniques can help identify patterns such as trend, seasonality, and autocorrelation, which can be useful for short-term forecasting.

The goal of this project is to demonstrate a complete time series analysis pipeline, including data preprocessing, stationarity testing, ARIMA modeling, and visualization of predicted stock prices.



**📊 Features**

Load and analyze historical stock price data

Convert raw data into time series format

Perform stationarity testing using the Augmented Dickey-Fuller (ADF) test

Apply log transformation and differencing to stabilize the series

Analyze Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots

Train an ARIMA model on stock price data

Generate predicted values

Visualize actual vs predicted stock prices



**🧠 Time Series Analysis Workflow**

The project follows a structured time series modeling approach:

1️⃣ Load historical stock price dataset

2️⃣ Convert date column into time index

3️⃣ Handle missing values in the dataset

4️⃣ Check stationarity using rolling statistics and ADF test

5️⃣ Apply log transformation to stabilize variance

6️⃣ Apply differencing to remove trend

7️⃣ Analyze ACF and PACF plots to understand lag relationships

8️⃣ Train an ARIMA model (1,1,1)

9️⃣ Generate predictions

🔟 Convert predictions back to the original scale and visualize results



## 📂 Project Structure

```
Stock-Market-Prediction-Using-Time-Series-Analysis
│
├── Dataset
│   ├── AAPL.csv
│   ├── AMZN.csv
│   └── FB.csv
│
├── Images
│   ├── ACF_Plot.png
│   └── Arima_Prediction.png
│
├── TimeSeriesAnalysis.py
├── requirements.txt
└── README.md
```

**Folder Description**

Dataset/
Contains historical stock price datasets used for time series analysis.

AAPL.csv → Apple stock data

AMZN.csv → Amazon stock data

FB.csv → Facebook stock data

Images/
Contains output visualization graphs generated from the analysis.

ACF_Plot.png → Autocorrelation function plot used for ARIMA parameter analysis

Arima_Prediction.png → Actual vs predicted stock price graph

TimeSeriesAnalysis.py
Main Python script that performs:

Data preprocessing

Stationarity testing (ADF test)

Rolling statistics

Log transformation & differencing

ACF & PACF analysis

ARIMA model training

Prediction visualization

requirements.txt
Contains the Python dependencies required to run the project.

README.md
Project documentation including description, workflow, installation steps, and results.



**⚙️ Technologies Used**

Python

NumPy

Pandas

Matplotlib

Statsmodels

Scikit-learn



**📦 Installation**

Clone the repository:
git clone https://github.com/dv200103/Stock_Price_Prediction_-Time_Series_Analysis-.git

cd Stock_Price_Prediction_-Time_Series_Analysis-

Install dependencies:

pip install -r requirements.txt

**▶️ Running the Project**

Run the script using Python:

python TimeSeriesAnalysis.py

The program will:

Load stock data

Perform time series analysis

Train the ARIMA model

Display graphs showing prediction results



**📈 Output Example**

The project generates visualization plots such as:

Autocorrelation Analysis

Helps identify relationships between current and previous values in the time series.

![ARIMA Prediction](images/arima_prediction.png)

ARIMA Model Prediction

Displays actual vs predicted stock prices.

![ACF Plot](images/acf_plot.png)

These graphs help visualize how well the ARIMA model fits the historical stock price data.



**⚠️ Limitations**

The model uses historical price data only and does not include external market factors.

ARIMA performs best for short-term forecasting.

Financial markets are inherently volatile, so predictions should not be used for real trading decisions.



**🚀 Possible Improvements**

Future improvements could include:

Automatic ARIMA parameter selection using AIC optimization

Train/test split with RMSE evaluation

Forecasting future stock prices

Building a Streamlit dashboard for interactive visualization

Using advanced models like LSTM or Prophet



**📜 License**

This project is open-source and available under the MIT License.
