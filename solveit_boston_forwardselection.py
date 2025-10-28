import pandas as pd
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Read dataset into a DataFrame
df = pd.read_csv("boston.csv")

"""
PERFORM FORWARD VARIABLE SELECTION
"""

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()
model_fwdsel = sfs(model, k_features=5, forward=True, verbose=2, scoring='neg_mean_squared_error')
model_fwdsel = model_fwdsel.fit(X_train, y_train)


"""
REBUILD THE MODEL USING THE SELECTED VARIABLES
"""

# Get top-5 selected explanatory variables
feat_names = list(model_fwdsel.k_feature_names_)

# Include the name of the response variable
feat_names.append("MV")

# Reduce the original dataset to top-5 selected variables, plus the response variable
df_fwd = df[feat_names]

# (Again) Separate explanatory variables (x) from the response variable (y)
x = df_fwd.iloc[:,:-1].values
y = df_fwd.iloc[:,-1].values

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Rebuild a linear regression model 
model = LinearRegression()

# Fit the model using the selected variables
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_train)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_train, "Predicted": y_pred})
print(df_pred)


# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_train, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_train, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_train, y_pred))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_train, y_pred)


print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)
