#| echo: false
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import grangercausalitytests
from arch.unitroot import ADF, PhillipsPerron, KPSS
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category=UserWarning)  # ignoring warnings of type UserWarning

# Download stock data (already a DataFrame)
data1 = pd.DataFrame(yf.download('GM','2024-06-01','2025-09-25'))
data2 = pd.DataFrame(yf.download('AAPL','2024-06-01','2025-09-25'))
data1['log_returns'] = np.log(data1['Close']/data1['Close'].shift(1))
data2['log_returns'] = np.log(data2['Close']/data2['Close'].shift(1))

data1 = data1.dropna()
data2 = data2.dropna()
print(data1.head())
print(data2.head())
# Extract closing prices
y1 = data1['Close']
y2 = data2['Close']


# Compute correlation of log returns
corr_levels = np.round(data1['log_returns'].corr(data2['log_returns']), 2)
print(corr_levels)

y1_norm = y1 / y1.iloc[0]
y2_norm = y2 / y2.iloc[0]

# Spread
spread = y1_norm - y2_norm

# Differences
dy1 = y1_norm.diff().dropna()
dy2 = y2_norm.diff().dropna()

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14,5))

# ---- Left plot: normalized series + spread ----
axes[0].plot(y1_norm, label='y1_norm')
axes[0].plot(y2_norm, label='y2_norm')
axes[0].plot(spread, label='Spread (y1_norm - y2_norm)', linestyle='--', color='black')
axes[0].set_title('Normalized series and spread')
axes[0].legend()
axes[0].grid(True)
axes[0].tick_params(axis='x', rotation=45)

# ---- Right plot: scatter of differences ----
axes[1].scatter(dy1, dy2, alpha=0.6)
axes[1].set_title('Scatter plot of Δy1_norm vs Δy2_norm')
axes[1].set_xlabel('Δy1_norm')
axes[1].set_ylabel('Δy2_norm')
axes[1].grid(True)

plt.tight_layout()
plt.show()


X_price = y1["GM"]
y_price = y2["AAPL"]
X_price = sm.add_constant(X_price)  # adding a constant term for intercept

model_ols_price = sm.OLS(y_price, X_price).fit()
#print(model_ols_price.summary())
residuals = model_ols_price.resid
# and test them for stationarity with the ADF test
adf_test = ADF(residuals)
# by default number of lags is selected automatically based on AIC
# WHICH is not correct as it does not take into account potential autocorrelation
# and trend = 'c' (constant/drift) is used
print(adf_test.summary().as_text())