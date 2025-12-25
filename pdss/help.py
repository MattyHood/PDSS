# Help function to guide users
def help(section=None):
  """
   Help Function for PDSS

   Examples:
      help()
      help("classification")
      help("regression")
      help("timeseries")
      help("preprocessing")
      help("validation")

  Parameters:
  section : str or None (default None)
    The topic to get help on. Options:
      - 'preprocessing'
      - 'regression'
      - 'timeseries'
      - 'classification'
      - 'all'
    If None, prints an overview.

  """

  sections = {
    "preprocessing": preprocessing_help,
    "regression": regression_help,
    "timeseries": timeseries_help,
    "classification": classification_help,
    "validation": validation_help,
    }
 #    "all": ALL_HELP - Too large scale for user experience.

  if section is None:
        print(overview)
        print("\nAvailable sections:")
        for s in sections:
            print(f" - {s}")
        print('\nUse help("<section>") to view details.')
        return

  section = section.lower()

  if section not in sections:
        print(f"Unknown topic '{section}'.")
        print("Available topics:", ", ".join(sections.keys()))
        return

  print(sections[section])


overview = """
PDSS — Python Data Science Summary Tools
----------------------------------------
A unified toolkit for:

• Data preprocessing & cleaning
• Regression modelling and diagnostics
• Classification modelling (with optional optimisation)
• Time series forecasting (ARIMA / Prophet)
• Model evaluation, diagnostics, and validation tools

Designed for fast, repeatable workflows in data science projects.

To view documentation for a section:
    help("classification")
    help("regression")
"""

preprocessing_help = """
Preprocessing Tools
-------------------
Tools available include:

• Automatic dtype fixing  // Not yet added
• Missing value handling (drop, fill, interpolate)  // Not yet added
• Outlier removal (IQR and Z-score)  // Not yet added
• StandardScaler and MinMaxScaler utilities  // Not yet added
• Automatic categorical encoding  // Not yet added
• Train/test split helpers
• Feature selection utilities (correlation, VIF checks)

Example:
    from pdss.preprocessing import fix_dtypes, scale_features
"""

regression_help = """
Regression Models
-----------------
Steps for using regression tools in PDSS

1. Prepare your data
   X must contain NUMERIC columns only.
   y must be the target variable.

   Example:
       X = df[['area', 'rooms', 'bathrooms']]
       y = df['price']

2. Optional feature diagnostics:
   - Pearson correlation matrix
   - VIF (variance inflation factor)
         from pdss.regression import calculate_vif
         calculate_vif(X)

3. Train/test split
   Managed inside PDSS regression functions.

4. Train a model
   Example:
       result = train_regression(X, y)

5. Diagnostics & assumptions checks:
   • Residual plots
   • Q-Q plot
   • Homoscedasticity check
   • Predicted vs Actual plot

   Example:
       plot_regression_diagnostics(result["model"], result["X_test"], result["y_test"])

Metrics reported:
• MAE
• MSE / RMSE
• R² Score
"""



timeseries_help = """
Time Series Forecasting
-----------------------
Supported:
• ARIMA / SARIMA
• Prophet (Facebook Prophet)
• Exponential Smoothing (future support planned)

Typical Workflow:
1. Ensure your index is a DateTimeIndex
       df.index = pd.to_datetime(df.index)

2. Handle missing timestamps:
       df = df.asfreq('D')      # daily frequency
       df = df.interpolate()    # fill gaps

3. Split based on time (automatic inside PDSS)

4. Train a model:
       arima_result = train_arima(df['sales'], order=(1,1,1))

       OR Prophet:
       prophet_result = train_prophet(df)

Tools included / planned:
• ADF Stationarity test
• PACF/ACF plotting
• Seasonal decomposition
• Time series interpolation tools
"""


classification_help = """
Classification Models
---------------------
Supported models:
• Logistic Regression
• Random Forest Classifier
• Gradient Boosting Classifier
• AdaBoost Classifier
• KNN Classifier
• SVM (SVC)

How to use:

1. Prepare your dataset:
       X = df.select_dtypes(include="number")
       y = df["Outcome"]

2. Train one model:
       result = train_classifier(X, y, model_name="RandomForestClassifier")

3. OR train all models:
       results = train_classifier(X, y, all_models=True)

4. Optional optimisation:
       result = train_classifier(X, y, optimise=True, n_iter=20)

Optimisation:
• Uses RandomizedSearchCV
• n_iter automatically capped at total parameter combinations

Evaluation includes:
• Train/Test Accuracy
• Precision
• Recall
• F1 Score
• Confusion Matrix (plotted automatically)
"""


validation_help ="""
Result Validation Help // Pending Development (Include RMSE is interpreted in xyz, f-score, precision score, accuracy score mean so and so, etc...)
----------------------


"""
