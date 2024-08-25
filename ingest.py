# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
from ydata_profiling import ProfileReport



# %% [markdown]
# # Import Data

# %%
df = pd.read_csv('./data/titanic_raw_data.csv')

# %%
profile = ProfileReport(df, title="Titanic Data Profiling Report")


# %%
profile.to_notebook_iframe()


# %%
