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
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.express as px
import string

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

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# - n_cabin: number of cabins that the passenger is part of
# - n_cabin_deck: number of decks that the passenger is part of
# - family_size: number of parents + siblings + spouses + children + self
# - age_bin: category of age
# - title: associated with honorifics
# - ticket_group_size

# %%
df.columns = df.columns.str.lower()
df['embarked'] = df['embarked'].replace({'C': 'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'})

# %%
df['cabin_cleaned'] = df['cabin'].map(lambda x: x.split(' ') if pd.notna(x) else [])
df['cabin_deck'] = df['cabin'].map(lambda x: x[0] if pd.notna(x) else x)

df['n_cabin'] = df['cabin_cleaned'].map(len)
df['n_cabin_deck'] = df['cabin_cleaned'].map(lambda x: len(set([i[0] for i in x])))

df['family_size'] = df['sibsp'] + df['parch'] + 1
df['age_bin'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior']).astype(str)
df['title'] = df['name'].str.lower().str.extract(r'([a-z]+)\.', expand=False)
df['ticket_group_size'] = df.groupby('ticket')['ticket'].transform('count')
df['title_frequency'] = df.groupby('title')['title'].transform('count')
df['fare_per_person'] = df['fare'] / df['ticket_group_size']
df['name_length'] = df['name'].str.replace(f'[{string.punctuation}]', '', regex=True).str.split().apply(len)


# %%
df.drop(columns = ['cabin','cabin_cleaned','ticket','name'], inplace = True)
