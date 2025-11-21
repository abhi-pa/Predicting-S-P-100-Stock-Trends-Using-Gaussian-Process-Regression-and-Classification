# Data Inspection
import pandas as pd
import zipfile

# Load and inspect the dataset
zip_file = "/content/sep100.csv (1).zip"
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("/content/")

file_path = "/content/sep100.csv"  # Update the path if necessary
data = pd.read_csv(file_path)

print("Original Data Shape:", data.shape)
print("Columns in Original Data:", data.columns)
print("Missing Values in Original Data:")
print(data.isnull().sum())
print("Data Head:")
print(data.head())

OUTPUT FOR ABOVE CODE:
Original Data Shape: (5786, 102)
Columns in Original Data: Index(['Date', 'AAPL', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM', 'ABT', 'ADBE', 'AIG',
       ...
       'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'ABBV', 'ACN'],
      dtype='object', length=102)
Missing Values in Original Data:
Date       0
AAPL       0
VZ         0
WBA        0
WFC        0
        ... 
UPS        0
USB        0
V       1930
ABBV    3136
ACN      256
Length: 102, dtype: int64
Data Head:
                  Date      AAPL         VZ       WBA       WFC       WMT  \
0  2000-07-13 00:00:00  1.008929  46.315121  30.93750  21.40625  59.12500   
1  2000-07-14 00:00:00  1.030134  44.516476  31.25000  21.84375  59.50000   
2  2000-07-17 00:00:00  1.041295  44.516476  31.21875  21.68750  60.71875   
3  2000-07-18 00:00:00  1.022321  43.898193  31.18750  21.03125  60.06250   
4  2000-07-19 00:00:00  0.940848  42.268169  31.31250  21.09375  60.00000   

        XOM        ABT       ADBE          AIG  ...  TMUS  TSLA      TXN  \
0  39.00000  19.583302  34.468750  1603.333374  ...   NaN   NaN  70.8750   
1  38.78125  19.078287  34.046875  1598.333374  ...   NaN   NaN  72.5000   
2  39.37500  19.078287  33.765625  1591.666626  ...   NaN   NaN  73.5000   
3  39.09375  18.825781  32.734375  1589.166626  ...   NaN   NaN  70.5625   
4  39.34375  18.461048  32.484375  1590.000000  ...   NaN   NaN  67.2500   

         UNH        UNP      UPS      USB   V  ABBV  ACN  
0  10.718750  10.250000  60.3125  21.5625 NaN   NaN  NaN  
1  10.562500  10.875000  60.8750  21.9375 NaN   NaN  NaN  
2  10.867188  11.015625  60.4375  21.4375 NaN   NaN  NaN  
3  10.406250  10.984375  60.7500  20.8750 NaN   NaN  NaN  
4  10.492188  10.578125  61.2500  21.0625 NaN   NaN  NaN  

[5 rows x 102 columns]

# Inspect Missing Data More Closely
# Summary of missing values by column
missing_values_summary = data.isnull().sum()
print("Missing Values Summary:")
print(missing_values_summary[missing_values_summary > 0])

# Check rows with missing values
print("Rows with Missing Values:")
print(data[data.isnull().any(axis=1)].head())

output for above code:
Missing Values Summary:
AVGO     2279
BRK.B    5786
CHTR     2383
CRM       989
DOW      4699
FB       5786
GM       2604
GOOG     1029
GOOGL    1029
KHC      3766
MA       1474
MDLZ      231
NFLX      465
PM       1928
PYPL     3766
TMUS     1699
TSLA     2504
V        1930
ABBV     3136
ACN       256
dtype: int64
Rows with Missing Values:
                  Date      AAPL         VZ       WBA       WFC       WMT  \
0  2000-07-13 00:00:00  1.008929  46.315121  30.93750  21.40625  59.12500   
1  2000-07-14 00:00:00  1.030134  44.516476  31.25000  21.84375  59.50000   
2  2000-07-17 00:00:00  1.041295  44.516476  31.21875  21.68750  60.71875   
3  2000-07-18 00:00:00  1.022321  43.898193  31.18750  21.03125  60.06250   
4  2000-07-19 00:00:00  0.940848  42.268169  31.31250  21.09375  60.00000   

        XOM        ABT       ADBE          AIG  ...  TMUS  TSLA      TXN  \
0  39.00000  19.583302  34.468750  1603.333374  ...   NaN   NaN  70.8750   
1  38.78125  19.078287  34.046875  1598.333374  ...   NaN   NaN  72.5000   
2  39.37500  19.078287  33.765625  1591.666626  ...   NaN   NaN  73.5000   
3  39.09375  18.825781  32.734375  1589.166626  ...   NaN   NaN  70.5625   
4  39.34375  18.461048  32.484375  1590.000000  ...   NaN   NaN  67.2500   

         UNH        UNP      UPS      USB   V  ABBV  ACN  
0  10.718750  10.250000  60.3125  21.5625 NaN   NaN  NaN  
1  10.562500  10.875000  60.8750  21.9375 NaN   NaN  NaN  
2  10.867188  11.015625  60.4375  21.4375 NaN   NaN  NaN  
3  10.406250  10.984375  60.7500  20.8750 NaN   NaN  NaN  
4  10.492188  10.578125  61.2500  21.0625 NaN   NaN  NaN  

[5 rows x 102 columns]

# Handle Missing Values Carefully
# Drop columns with more than a certain threshold of missing values
threshold = 0.5  # Example: drop columns with more than 50% missing values
cleaned_data = data.dropna(axis=1, thresh=len(data) * (1 - threshold))
print("Cleaned Data Shape after dropping columns:", cleaned_data.shape)
print("Columns in Cleaned Data:", cleaned_data.columns)

output for above code:
Cleaned Data Shape after dropping columns: (5786, 96)
Columns in Cleaned Data: Index(['Date', 'AAPL', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM', 'ABT', 'ADBE', 'AIG',
       'AMGN', 'AMT', 'AMZN', 'AVGO', 'AXP', 'BA', 'BAC', 'BIIB', 'BK', 'BKNG',
       'BLK', 'BMY', 'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST',
       'CRM', 'CSCO', 'CVS', 'CVX', 'DD', 'DHR', 'DIS', 'DUK', 'EMR', 'EXC',
       'F', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD',
       'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW',
       'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'MMM', 'MO', 'MRK', 'MS', 'MSFT',
       'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'QCOM',
       'RTX', 'SBUX', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN',
       'UNH', 'UNP', 'UPS', 'USB', 'V', 'ACN'],
      dtype='object')

# Drop rows with missing values in specific columns
important_columns = ['AAPL', 'VZ']  # Replace with relevant columns
cleaned_data = data.dropna(subset=important_columns)
print("Cleaned Data Shape after dropping rows:", cleaned_data.shape)
print("Columns in Cleaned Data:", cleaned_data.columns)

output for above code:
Cleaned Data Shape after dropping rows: (5786, 102)
Columns in Cleaned Data: Index(['Date', 'AAPL', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM', 'ABT', 'ADBE', 'AIG',
       ...
       'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'ABBV', 'ACN'],
      dtype='object', length=102)

# Recheck Data After Cleaning
print("Cleaned Data Shape:", cleaned_data.shape)
print("Columns in Cleaned Data:", cleaned_data.columns)
print("Cleaned Data Head:")
print(cleaned_data.head())

output for above code:
]
# Recheck Data After Cleaning
print("Cleaned Data Shape:", cleaned_data.shape)
print("Columns in Cleaned Data:", cleaned_data.columns)
print("Cleaned Data Head:")
print(cleaned_data.head())
Cleaned Data Shape: (5786, 102)
Columns in Cleaned Data: Index(['Date', 'AAPL', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM', 'ABT', 'ADBE', 'AIG',
       ...
       'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'ABBV', 'ACN'],
      dtype='object', length=102)
Cleaned Data Head:
                  Date      AAPL         VZ       WBA       WFC       WMT  \
0  2000-07-13 00:00:00  1.008929  46.315121  30.93750  21.40625  59.12500   
1  2000-07-14 00:00:00  1.030134  44.516476  31.25000  21.84375  59.50000   
2  2000-07-17 00:00:00  1.041295  44.516476  31.21875  21.68750  60.71875   
3  2000-07-18 00:00:00  1.022321  43.898193  31.18750  21.03125  60.06250   
4  2000-07-19 00:00:00  0.940848  42.268169  31.31250  21.09375  60.00000   

        XOM        ABT       ADBE          AIG  ...  TMUS  TSLA      TXN  \
0  39.00000  19.583302  34.468750  1603.333374  ...   NaN   NaN  70.8750   
1  38.78125  19.078287  34.046875  1598.333374  ...   NaN   NaN  72.5000   
2  39.37500  19.078287  33.765625  1591.666626  ...   NaN   NaN  73.5000   
3  39.09375  18.825781  32.734375  1589.166626  ...   NaN   NaN  70.5625   
4  39.34375  18.461048  32.484375  1590.000000  ...   NaN   NaN  67.2500   

         UNH        UNP      UPS      USB   V  ABBV  ACN  
0  10.718750  10.250000  60.3125  21.5625 NaN   NaN  NaN  
1  10.562500  10.875000  60.8750  21.9375 NaN   NaN  NaN  
2  10.867188  11.015625  60.4375  21.4375 NaN   NaN  NaN  
3  10.406250  10.984375  60.7500  20.8750 NaN   NaN  NaN  
4  10.492188  10.578125  61.2500  21.0625 NaN   NaN  NaN  

[5 rows x 102 columns]




# Proceed with Data Transformation if Data Exists
from sklearn.preprocessing import MinMaxScaler

# Ensure selected columns exist
selected_columns = ['AAPL', 'VZ']  # Replace with actual column names
if not all(col in cleaned_data.columns for col in selected_columns):
    raise ValueError("One or more selected columns are not in the dataframe")

# Feature Scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(cleaned_data[selected_columns])
scaled_data = pd.DataFrame(scaled_features, columns=[f'{col}_scaled' for col in selected_columns])

# Add scaled features back to the DataFrame
cleaned_data = pd.concat([cleaned_data.reset_index(drop=True), scaled_data], axis=1)

# Create new features
cleaned_data['percent_change'] = (cleaned_data[selected_columns[0]] - cleaned_data[selected_columns[1]]) / cleaned_data[selected_columns[1]] * 100

# Define classification function
def classify_movement(row):
    if row[selected_columns[1]] == 0:
        return 'Undefined'
    percent_change = (row[selected_columns[0]] - row[selected_columns[1]]) / row[selected_columns[1]] * 100
    if percent_change > 2:
        return 'Strong Bullish'
    elif percent_change > 0.5:
        return 'Bullish'
    elif percent_change < -2:
        return 'Strong Bearish'
    elif percent_change < -0.5:
        return 'Bearish'
    else:
        return 'Neutral'

# Apply classification
cleaned_data['movement_class'] = cleaned_data.apply(classify_movement, axis=1)

print("Transformed Data Head:")
print(cleaned_data.head())


output for above code:
Transformed Data Head:
                  Date      AAPL         VZ       WBA       WFC       WMT  \
0  2000-07-13 00:00:00  1.008929  46.315121  30.93750  21.40625  59.12500   
1  2000-07-14 00:00:00  1.030134  44.516476  31.25000  21.84375  59.50000   
2  2000-07-17 00:00:00  1.041295  44.516476  31.21875  21.68750  60.71875   
3  2000-07-18 00:00:00  1.022321  43.898193  31.18750  21.03125  60.06250   
4  2000-07-19 00:00:00  0.940848  42.268169  31.31250  21.09375  60.00000   

        XOM        ABT       ADBE          AIG  ...        UNP      UPS  \
0  39.00000  19.583302  34.468750  1603.333374  ...  10.250000  60.3125   
1  38.78125  19.078287  34.046875  1598.333374  ...  10.875000  60.8750   
2  39.37500  19.078287  33.765625  1591.666626  ...  11.015625  60.4375   
3  39.09375  18.825781  32.734375  1589.166626  ...  10.984375  60.7500   
4  39.34375  18.461048  32.484375  1590.000000  ...  10.578125  61.2500   

       USB   V  ABBV  ACN  AAPL_scaled  VZ_scaled  percent_change  \
0  21.5625 NaN   NaN  NaN     0.003998   0.591289      -97.821599   
1  21.9375 NaN   NaN  NaN     0.004108   0.544629      -97.685949   
2  21.4375 NaN   NaN  NaN     0.004166   0.544629      -97.660877   
3  20.8750 NaN   NaN  NaN     0.004068   0.528590      -97.671155   
4  21.0625 NaN   NaN  NaN     0.003647   0.486304      -97.774098   

   movement_class  
0  Strong Bearish  
1  Strong Bearish  
2  Strong Bearish  
3  Strong Bearish  
4  Strong Bearish  

[5 rows x 106 columns]
















# Data Splitting and Validation
from sklearn.model_selection import train_test_split

# Ensure there are enough rows to split
if cleaned_data.empty or len(cleaned_data) < 2:
    raise ValueError("The cleaned_data DataFrame is either empty or has too few rows to split.")

# Define features and target variable
X = cleaned_data.drop(columns=['movement_class'])
y = cleaned_data['movement_class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Shape:", X_train.shape, y_train.shape)
print("Testing Set Shape:", X_test.shape, y_test.shape)

# Validate data
print("NaN Values in Training Set:")
print(X_train.isnull().sum())

print("Target Variable Distribution in Testing Set:")
print(y_test.value_counts())

output for above code:
Training Set Shape: (4628, 105) (4628,)
Testing Set Shape: (1158, 105) (1158,)
NaN Values in Training Set:
Date                 0
AAPL                 0
VZ                   0
WBA                  0
WFC                  0
                  ... 
ABBV              2496
ACN                192
AAPL_scaled          0
VZ_scaled            0
percent_change       0
Length: 105, dtype: int64
Target Variable Distribution in Testing Set:
movement_class
Strong Bearish    960
Strong Bullish    188
Bullish             4
Bearish             3
Neutral             3
Name: count, dtype: int64

# installing the libraries
!pip install matplotlib-venn
!pip install mplfinance
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install pydot
!pip install cartopy
!pip install yfinance mplfinance seaborn

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # when warning occerr

# Step 1: Data Collection
symbols = ['AAPL', 'MSFT']  # Using Apple and Microsoft as examples
start_date = '2021-01-01'  # Changed to start from 2021
end_date = '2024-01-01'  # Changed to end at 2024

data = {}
for symbol in symbols:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            print(f"No data available for {symbol}")
        else:
            data[symbol] = df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

if not data:
    raise ValueError("No data was successfully retrieved for any symbol")

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Calculate technical indicators
    df['50_day_MA'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])

    # Handle missing values
    df.dropna(inplace=True)

    # Create features and target
    features = ['Close', 'Volume', '50_day_MA', 'RSI']
    target = 'Open'

    X = df[features].values
    y = df[target].shift(-1).values[:-1]  # Next day's open
    X = X[:-1]  # Remove last row to align with y

    return X, y

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

# Preprocess data for each symbol
X_all, y_all = [], []
for symbol, df in data.items():
    if len(df) > 50:  # Ensure we have enough data for 50-day MA
        X, y = preprocess_data(df)
        X_all.append(X)
        y_all.append(y)
    else:
        print(f"Not enough data for {symbol}")

if not X_all or not y_all:
    raise ValueError("No data left after preprocessing")

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.2, random_state=42)

# Step 3: Gaussian Process Regression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Check for NaN or infinite values in training data
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    raise ValueError("X_train contains NaN or infinite values.")
if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
    raise ValueError("y_train contains NaN or infinite values.")
if len(X_train) < 10 or len(y_train) < 10:
    raise ValueError("Not enough training data")

# Try different kernel configurations
kernels = [
    C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
    C(1.0, (1e-5, 1e5)) * RBF(length_scale=[1.0] * X_train.shape[1], length_scale_bounds=(1e-5, 1e5)),
    C(1.0) * Matern(length_scale=1.0, nu=1.5),
    WhiteKernel() + C(1.0) * RBF(length_scale=1.0)
]

best_mse = float('inf')
best_gpr = None

for i, kernel in enumerate(kernels):
    try:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                       alpha=1e-2, normalize_y=True, random_state=42)
        gpr.fit(X_train, y_train)
        y_pred, y_std = gpr.predict(X_test, return_std=True)
        mse = mean_squared_error(y_test, y_pred)

        print(f"Kernel {i+1} - MSE: {mse}")
        print(f"Optimized kernel parameters: {gpr.kernel_}")

        if mse < best_mse:
            best_mse = mse
            best_gpr = gpr
    except Exception as e:
        print(f"Error with kernel {i+1}: {e}")

if best_gpr is not None:
    print("\nBest Gaussian Process Regressor:")
    print(f"MSE: {best_mse}")
    print(f"Kernel: {best_gpr.kernel_}")

    # Make predictions with the best model
    y_pred, y_std = best_gpr.predict(X_test, return_std=True)
else:
    print("No successful Gaussian Process model was found.")
    # You might want to consider a different model entirely at this point
    # For example, you could try a simple linear regression:
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Linear Regression MSE: {mse}")

# Check predictions
print("Predictions:", y_pred)
if 'y_std' in locals():
    print("Standard Deviations:", y_std)

# Step 4: Classification
def classify_movement(actual, predicted):
    change = (predicted - actual) / actual
    if change > 0.02:
        return 'Strong Bullish'
    elif 0.005 < change <= 0.02:
        return 'Bullish'
    elif -0.005 <= change <= 0.005:
        return 'Neutral'
    elif -0.02 <= change < -0.005:
        return 'Bearish'
    else:
        return 'Strong Bearish'

y_class = np.array([classify_movement(a, p) for a, p in zip(y_test, y_pred)])

# Gaussian Process Classification
classification_kernel = RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=classification_kernel, random_state=42, max_iter_predict=100)

try:
    gpc.fit(X_test, y_class)
except Exception as e:
    print(f"Error during GPC fitting: {e}")

# Step 5: Visualizations
# Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.fill_between(range(len(y_pred)), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.2)
plt.title('Actual vs Predicted Prices with Confidence Intervals')
plt.legend()
plt.show()

# Correlation heatmap
corr_matrix = np.corrcoef(X_scaled.T)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Scatter Plot: Actual vs Predicted Prices')
plt.show()

# ROC curve
try:
    y_score = gpc.predict_proba(X_test)
    unique_classes = np.unique(y_class)
    plt.figure(figsize=(10, 6))
    for i, class_label in enumerate(unique_classes):
        y_true_binary = (y_class == class_label).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve {class_label} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
except Exception as e:
    print(f"Error during ROC Curve visualization: {e}")

# Candlestick chart (using the first symbol as an example)
# Assuming 'symbol' and 'df' are already defined
try:
    # Resample data to weekly and monthly for overall trend
    df_weekly = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    df_monthly = df.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))

    # Plot weekly data as line chart
    df_weekly['Close'].plot(ax=ax1)
    ax1.set_title(f'{symbol} Weekly Closing Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')

    # Plot monthly data as line chart
    df_monthly['Close'].plot(ax=ax2)
    ax2.set_title(f'{symbol} Monthly Closing Price')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')

    # Plot last 3 months of daily data as candlestick chart
    df_last_3months = df.last('3M')
    mpf.plot(df_last_3months, type='candle', ax=ax3, volume=ax3.twinx(),
             style='yahoo', warn_too_much_data=10000)
    ax3.set_title(f'{symbol} Last 3 Months Daily Candlestick')

    plt.tight_layout()
    plt.show()

    # Cumulative return chart
    returns = df['Close'].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns)
    plt.title(f'Cumulative Return for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.show()

except Exception as e:
    print(f"Error during Chart visualization: {e}")

This the final output for above code:
Kernel 1 - MSE: 4.4015439341251685
Optimized kernel parameters: 103**2 * RBF(length_scale=257)
Kernel 2 - MSE: 4.43346325492907
Optimized kernel parameters: 9.7**2 * RBF(length_scale=[17, 1e+05, 1e+05, 1e+05])
Kernel 3 - MSE: 4.3907323174618265
Optimized kernel parameters: 316**2 * Matern(length_scale=2.04e+03, nu=1.5)
Kernel 4 - MSE: 4.401562594114813
Optimized kernel parameters: WhiteKernel(noise_level=1e-05) + 103**2 * RBF(length_scale=257)

Best Gaussian Process Regressor:
MSE: 4.3907323174618265
Kernel: 316**2 * Matern(length_scale=2.04e+03, nu=1.5)
Predictions: [250.07172878 260.49941978 178.10967254 134.38257417 343.24877504
 151.25882787 299.14994228 143.71681193 131.18246161 251.55655389
 142.24279378 194.54975656 123.01757237 249.36150726 165.40329881
 291.15827452 251.49365266 146.24668676 153.64052264 236.14994288
 185.91782605 241.88996731 221.21972047 304.76415372 163.78188612
 176.77560187 152.65976884 130.03516144 170.64043252 252.51310584
 327.41153342 137.52827219 124.69446761 271.98390979 131.80261811
 271.37969278 259.92504675 152.4643205  174.11578228 287.65686951
 149.8843772  166.39586643 244.82924694 145.78378914 271.14037916
 248.80777107 191.38885828 251.39706285 307.4324754  160.52550907
 178.42390877 256.41673142 320.16413366 157.75413346 173.96444024
 354.57558462 122.74361782 316.0656081  171.64238328 230.9995191
 246.58985572 282.69672302 224.01458843 139.99708458 151.66765465
 274.88559914 263.24500237 190.92356182 141.91064421 169.20095157
 276.369815   172.82943562 294.99030577 157.10925377 236.45985025
 155.47346802 277.56815739 190.29520302 156.3922703  168.14662559
 124.07403118 190.23583911 163.73758362 194.77728785 175.10480245
 167.76854573 276.75581278 189.05596633 149.87184929 222.50908291
 184.21027591 165.04893894 165.29970767 264.18480296 129.43401042
 327.57318372 249.28571077 143.00712857 290.13066573 284.03433797
 195.22839564 266.48991154 260.05651248 128.68446941 191.94459083
 286.43799292 148.64218359 171.21771181 139.37054112 177.271025
 158.55025244 238.702779   304.66100601 175.49460152 258.47981646
 173.20454409 240.87637239 285.99463407 292.03873039 330.80091506
 326.21167392 312.03504252 316.23458249 287.89278594 177.51404188
 162.68590681 172.26573542 162.22506143 191.44749302 144.0994286
 307.69994488 127.44453984 228.66577218 247.31009678 368.67828298
 235.23142268 225.89104127 168.45494295 130.76700592 148.89766791
 241.14171707 261.86974166 364.84033901 144.58007065 226.19238913
 331.96607729 245.23276558 245.54654891 189.36349142 339.49801625
 267.92594962 239.21585622 179.01281958 218.28754108 273.83195672
 294.01090956 160.35025137 162.67136363 326.06999062 311.29134498
 239.47270381 140.92673482 345.28637292 371.74325697 332.02465717
 153.20826952 237.34755657 300.63605221 285.51479315 169.59272538
 278.69402452 252.12925299 124.56466575 371.03869628 261.23764264
 328.05376894 132.17970415 147.69908067 324.97408489 229.37101629
 195.45217726 279.4175638  189.53304883 289.30784267 294.79780896
 224.30725879 176.73542403 188.27291262 248.90925201 165.33879304
 163.13029488 178.7011024  159.58324451 146.42392034 267.8746464
 141.99203951 321.42548592 290.04701678 233.38880275 273.13319929
 306.97682802 147.29513779 249.6807738  197.44443342 331.81799439
 174.79908642 169.46748243 285.36058085 248.40408757 158.41650956
 164.06998811 240.32094335 161.32032133 333.36976999 185.86366548
 278.02321431 173.7801431  273.21089666 133.70488282 183.02324242
 372.98888946 302.08026152 128.97299484 304.89355532 178.04464361
 163.62787585 173.55976278 144.817229   132.31684658 277.70348745
 186.95108898 310.87210164 276.6478759  279.02087129 153.07030934
 187.16232337 123.70375371 331.36225655 280.51999587 182.12364108
 328.44723419 222.74154405 150.3983917  282.1656988  172.16597159
 139.62055981 249.57416431 151.67766662 319.22223894 150.38656356
 163.57583549 147.08713046 224.46466443 167.82538939 372.78317672
 169.70504962 172.51734602 146.95962975 287.2097295  167.4564716
 145.9804446  156.18262999 318.93006992 157.96157757 146.56520541
 162.94144782 328.64401012 327.57040597 331.60417624 238.53996711
 130.43798963 172.84325255 324.22063489 151.19941599 325.80538659
 277.37106025 251.63498422 176.41476856 303.91525799 149.00368992
 167.34109806 155.35510438]
Standard Deviations: [1.64750209 0.73997358 0.50553221 0.72575531 0.71614539 0.56880136
 1.05058099 0.70697503 0.75002254 0.67357562 0.59678783 0.61225267
 0.69934648 0.54722354 0.51555957 0.48857143 0.83246766 0.54826252
 1.65697525 0.56956506 0.53906782 0.54966355 0.63003359 0.6543746
 0.63503428 0.89669338 0.56083875 0.70573026 0.54196941 0.73847007
 0.71440308 0.82049713 0.63511587 0.66925132 0.64636804 0.68289528
 0.78702645 0.52948556 0.49355666 0.64364975 0.5748247  0.69411589
 0.56401473 0.77875132 0.77728425 0.54548436 0.73621495 0.89850539
 0.63384289 1.13086777 0.51464075 0.70729844 0.89509499 1.35558446
 0.50048356 0.74430631 0.89886362 0.84068915 0.61192446 0.7685759
 0.57542018 0.74456084 0.59956268 0.516957   0.62655718 0.67223323
 0.64180075 0.84921972 1.24000743 1.34085051 0.54640934 0.67893953
 0.54208527 1.20527357 0.78043329 1.13460496 0.52101134 0.93573273
 0.86779095 0.57976545 0.79148094 0.55794422 0.52425395 0.56384421
 0.57468605 0.86362781 0.56071863 0.61373781 0.48738133 0.72722202
 0.52474686 0.59477959 0.56219808 0.62856497 0.75918975 0.68565966
 0.76754611 0.49003317 0.52966027 0.54782629 0.55110868 0.56752216
 0.60170658 0.75902701 0.55654601 0.54858798 0.54001742 0.82566185
 0.78403612 0.56769251 0.49987703 0.68101754 0.77684979 0.75444548
 0.61120881 0.67426915 0.63289612 0.56854237 0.53302543 0.61324143
 0.64227833 0.94072705 0.83953481 0.52984504 0.62786609 0.58501632
 0.50804945 0.6245468  0.52649256 0.56472367 0.69446459 0.65964062
 0.55280619 0.65807182 1.19833879 0.63987286 0.52106323 0.51777372
 0.75143532 0.56839572 0.57650998 0.60441139 1.2172206  0.56220932
 0.82466374 0.6349908  0.5887639  0.7015878  0.71782179 0.93818595
 0.66099461 0.75128415 0.81782904 0.65972525 0.67176731 0.57069161
 0.82039098 0.8005647  0.6231946  0.52570498 0.49577731 0.56291971
 0.86959139 0.98401228 0.63957206 0.74768852 0.5060855  0.84026345
 0.68391976 0.59947849 0.57052195 0.52208122 0.62230074 0.93147052
 0.67109476 0.57056269 0.81960272 0.66649122 0.57495107 0.52405925
 0.6473593  0.65104618 0.53442054 0.91021795 0.71310603 0.52870316
 0.57410249 0.67142188 0.59487023 0.59683949 0.57899232 0.56994286
 0.53196043 0.47064794 0.70941073 0.92058028 0.69396256 0.83262612
 0.86648932 0.95272095 0.5880043  0.52015936 0.64443337 0.63927998
 0.66880543 0.5354848  0.70805207 0.52970986 0.53415069 0.99334
 0.52636039 0.66051752 0.77205437 0.74045802 0.6066372  0.59632179
 1.23393838 0.52433999 0.62669511 0.72277772 1.05886093 0.84037424
 0.70043657 0.61881387 0.923821   0.72146818 0.64759469 0.75659317
 0.55162445 0.55672811 0.63830911 0.81981122 1.15006401 0.54031304
 1.09704343 0.58645962 0.63897416 0.82176937 0.49908934 0.50797045
 0.71696082 0.91816916 0.62174598 0.68045369 0.92779302 0.88656849
 0.55619188 0.55199073 0.68566558 0.63359846 0.49748273 0.52169776
 0.68056185 0.51661939 0.94470517 0.7051001  0.54798669 0.87098376
 0.51768245 1.9752847  0.52984691 0.49925006 0.76307802 1.00654748
 0.72969814 0.56444639 0.70202825 0.58662597 0.6639501  0.54054426
 1.31516431 0.5663058  1.15697424 0.62070155 0.712128   0.60131007
0.58482207 1.01500675 0.77075934 1.58880052 0.63365376 0.71588567]
 
 
 
 
 
 
 

