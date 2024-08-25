# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
from ydata_profiling import ProfileReport

# %% [markdown]
# # Import Data

# %% [markdown]
# - survival - Survival (0 = No; 1 = Yes)
# - class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - name - Name
# - sex - Sex
# - age - Age
# - sibsp - Number of Siblings/Spouses Aboard
# - parch - Number of Parents/Children Aboard
# - ticket - Ticket Number
# - fare - Passenger Fare
# - cabin - Cabin
# - embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %%
df = pd.read_csv('./data/titanic_raw_data.csv')

# %%
profile = ProfileReport(df, title="Titanic Data Profiling Report")


# %%
profile.to_notebook_iframe()


# %%
