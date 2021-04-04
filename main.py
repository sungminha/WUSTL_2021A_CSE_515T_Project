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
# observed_home_goals = df.home_score.values
# observed_away_goals = df.away_score.values
# home_team = df.i_home.values
# away_team = df.i_away.values
# num_teams = len(df.i_home.unique())
# num_games = len(home_team)


