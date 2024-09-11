# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

from ingest import *

# +
import shap
import numpy as np

from ydata_profiling import ProfileReport
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import ipywidgets as widgets
import plotly.express as px
import plotly.io as pio
# -

# # High Level Overview

profile = ProfileReport(df, title="Titanic Data Profiling Report")
profile.to_notebook_iframe()

# # Anomaly Detection for Data Quality

# +
df_imputation = df.copy()

threshold = 0.2

for col in df_imputation.columns:
    missing_fraction = df_imputation[col].isna().mean()
    
    if missing_fraction < threshold:
        if df_imputation[col].dtype == 'float64' or df_imputation[col].dtype == 'int64':
            # Impute numerical columns with mean or median
            df_imputation.loc[:,col] = df_imputation[col].fillna(df_imputation[col].median())
        else:
            # Impute categorical columns with mode (most frequent value)
            df_imputation.loc[:,col] = df_imputation[col].fillna(df_imputation[col].mode()[0])
    else:
        # For high missing fraction, fill with 'missing' (categorical) or 0 (numerical)
        if df_imputation[col].dtype == 'float64' or df_imputation[col].dtype == 'int64':
            df_imputation.loc[:,col] = df_imputation[col].fillna(0) 
        else:
            df_imputation.loc[:,col] = df_imputation[col].fillna('missing')
# -

df_imputation = pd.get_dummies(df_imputation, drop_first=True)
clf = IsolationForest(n_estimators=200, n_jobs=-1, contamination='auto', random_state=42)
clf.fit(df_imputation)
anomaly_scores = clf.decision_function(df_imputation)

fig = px.histogram(anomaly_scores, title = 'Distribution of Anomaly Scores')
fig.show()

# # Evaluation of Anomalies

# +
explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(df_imputation)
sorted_scores = np.argsort(anomaly_scores)

def plot_shap(idx):

    shap_exp = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value, 
        data=df_imputation.iloc[idx],
        feature_names=df_imputation.columns 
    )

    plt.figure()

    plt.title(f'Passenger ID: {df_imputation.loc[idx,"passengerid"]} \nAnomaly Score: {round(anomaly_scores[idx], 2)}')

    shap.waterfall_plot(shap_exp)
    
# Create an interactive dropdown for selecting an instance based on sorted anomaly scores
dropdown = widgets.Dropdown(
    options=[(f'Passenger {df_imputation.loc[idx, "passengerid"]} (Anomaly Score: {round(anomaly_scores[idx], 2)})', idx) for idx in sorted_scores[:5]],
    description='Passenger:',
    disabled=False,
)

# Update the plot when the dropdown selection changes
widgets.interact(plot_shap, idx=dropdown)
# -





