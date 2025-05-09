readme_content = """
# 📊 Tesla Stock Price Direction Prediction

This project applies machine learning techniques to predict the direction (up/down) of Tesla’s stock price using historical price data and engineered features. The goal is to determine whether the closing price of the next day will be higher than the current day.

## 📁 Dataset

The dataset used is a CSV file containing Tesla stock market data including:
- Date
- Open, High, Low, Close, Adj Close prices
- Volume

> **Note:** Make sure the `Tesla.csv` file is correctly placed in your working directory.

## 🧠 Models Used

Three classification models were trained and evaluated:
- Logistic Regression
- Support Vector Machine (Polynomial Kernel)
- Random Forest Classifier

Each model's performance was evaluated using:
- Accuracy
- F1 Score
- Log Loss

## 🧪 Feature Engineering

The following features were engineered:
- `open-close` = Open price - Close price
- `low-high` = Low price - High price
- `is_quarter_end` = 1 if the month is the end of a financial quarter (March, June, September, December), else 0

## 📈 Data Visualization

Several visualizations were created to understand the data:
- Price trend plot
- Distribution and box plots for numerical features
- Correlation heatmap
- Target class distribution (next day up or down)

## 🚀 Prediction on New Data

The trained Logistic Regression model was used to predict whether the stock will go up based on new input features. A confidence message is also shown to indicate certainty level of the prediction.
