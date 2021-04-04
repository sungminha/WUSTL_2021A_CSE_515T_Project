"""
expect to run on python >3.7
"""
import os
import math
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc # I know folks are switching to "as pm" but I'm just not there yet

DATA_DIR = os.path.join(os.getcwd(), 'data')
data_file = os.path.join(DATA_DIR, 'final_18-19season.csv')

#load data: we assume this is processed data table with indices for team numbers
df = pd.read_csv(data_file, sep=",")
df.head()

print(df)

#select columns we want
observed_home_goals = df['FTHG']
observed_away_goals = df['FTAG']
home_team = df['HomeTeam']
away_team = df['AwayTeam']
num_teams = len(home_team.unique())
num_games = len(home_team)

print(observed_home_goals)
print(observed_away_goals)
print(home_team)
print(away_team)