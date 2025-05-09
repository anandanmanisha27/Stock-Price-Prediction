import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Load dataset (Ensure the path is correct)
df = pd.read_csv(r"C:\Users\kanna\Downloads\Tesla.csv")

# Display the first few rows
print(df.head())
print(df.shape)


print(df.describe())

df.info()

plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()


#to check if same close and adj column

print(df[df['Close']==df['Adj Close']].shape)


df=df.drop(['Adj Close'],axis=1)


print(df.head())



print(df.isnull().sum())

features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()


plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()


splitted=df['Date'].str.split('/',expand=True)





df['day']=splitted[1].astype('int')
df['month']=splitted[0].astype('int')
df['year']=splitted[2].astype('int')

print(df.head())




df['is_quarter_end']=np.where(df['month']%3==0,1,0)
print(df.head())


data_grouped=df.drop('Date',axis=1).groupby('year').mean()

plt.figure(figsize=(20,10))

for i, col in enumerate(['Open','High','Low','Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
plt.show()


print(df.drop('Date',axis=1).groupby('is_quarter_end').mean())


df['open-close']=df['Open']-df['Close']
df['low-high']=df['Low']-df['High']

df['target']=np.where(df['Close'].shift(-1)>df['Close'],1,0)



plt.pie(df['target'].value_counts().values,labels=[0,1],autopct='%1.1f%%')

plt.show()

plt.figure(figsize=(10,10))

sb.heatmap(df.drop('Date',axis=1).corr()>0.9,annot=True, cmap="Spectral")
plt.show()



features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (Polynomial Kernel)": SVC(kernel='poly', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=2022)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]  # Probability scores

    results[name] = {
        "Accuracy": accuracy_score(Y_valid, y_pred),
        "F1-score": f1_score(Y_valid, y_pred),
        "Log Loss": log_loss(Y_valid, y_proba)
    }

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print(results_df)




new_stock = pd.DataFrame({
    'Open': [850],        # Open at $850
    'High': [900],        # High reaches $900 (big jump)
    'Low': [845],         # Low is close to Open (less volatility)
    'Close': [890],       # Closes much higher than Open (bullish signal)
    'Volume': [5000000],  # High volume indicates strong buying interest
    'Date': ["03/29/2025"] # Near quarter-end (March 29)
})



# Convert date to datetime format
new_stock['Date'] = pd.to_datetime(new_stock['Date'])

# Feature Engineering (Same as training data)
new_stock['open-close'] = new_stock['Open'] - new_stock['Close']
new_stock['low-high'] = new_stock['Low'] - new_stock['High']
new_stock['is_quarter_end'] = np.where(new_stock['Date'].dt.month % 3 == 0, 1, 0)

# Select relevant features
new_stock_features = new_stock[['open-close', 'low-high', 'is_quarter_end']]

# Scale new data using the same scaler
new_stock_scaled = scaler.transform(new_stock_features)

# Make predictions using the best model (Logistic Regression)
predicted_prob = models["Logistic Regression"].predict_proba(new_stock_scaled)[:, 1]
predicted_class = models["Logistic Regression"].predict(new_stock_scaled)

# Set a threshold for uncertainty (0.45 - 0.55 is uncertain)
uncertainty_threshold = 0.05

# Check confidence level
if abs(predicted_prob[0] - 0.5) < uncertainty_threshold:
    confidence_message = "⚠️ Uncertain prediction (Low confidence)."
elif predicted_prob[0] > 0.5:
    confidence_message = "✅ Predicted: Price will go UP 📈 (High confidence)"
else:
    confidence_message = "🔻 Predicted: Price will go DOWN 📉 (High confidence)"

# Print Results
print(f"Predicted Probability of Price Going Up: {predicted_prob[0]:.2f}")
print(f"Predicted Class (0=Down, 1=Up): {predicted_class[0]}")
print(confidence_message)
