# -*- coding: utf-8 -*-

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.getcwd(), 'data')
CHART_DIR = os.path.join(os.getcwd(), 'charts')
data_file = os.path.join(DATA_DIR, 'final_18-19season.csv')
team_file = os.path.join(DATA_DIR, 'team_index.csv')
VERBOSE = True  # more printouts

# sanity check: check if data file exists
if not (os.path.isfile(data_file)):
    print("".join(
        ["ERROR: data file (", str(data_file), ") does not exist."]), flush=True)
    sys.exit()

# sanity check: check if team file exists
if not (os.path.isfile(team_file)):
    print("".join(
        ["ERROR: team file (", str(team_file), ") does not exist."]), flush=True)
    sys.exit()

# load data: we assume this is processed data table with indices for team numbers
df = pd.read_csv(data_file, sep=",")
team_details = pd.read_csv(team_file, sep=",")
# df.drop(df.iloc[:, 21:63], axis = 1, inplace = True) #extra columns

#select corners, shots on target, and goals
corners = df[['HC','AC']]
shots_on_target = df[['HST','AST']]
goals = df[['FTHG','FTAG']]
res = df[['FTR']] #use only res for now

home_away = df[['MatchNo','HomeTeam','AwayTeam']]

streak_data = home_away.merge(res, left_index=True, right_index=True) #380 matches x 4 (match count, home, away, results) 
# streak_data = streak_data.merge(corners, left_index=True, right_index=True)
# streak_data = streak_data.merge(shots_on_target, left_index=True, right_index=True)
# streak_data = streak_data.merge(goals, left_index=True, right_index=True)
#clean up data for memory management
del home_away
del corners, shots_on_target, goals
del df
del res
if (VERBOSE):
  print("streak_data: ")
  print(streak_data, flush=True)

#non-weighted streaks
streak_home = np.zeros(shape=(np.shape(streak_data)[0], np.shape(team_details)[0])) #380 matches by 20 teams
streak_data = np.zeros(shape=(np.shape(streak_data)[0], np.shape(team_details)[0])) #380 matches by 20 teams

for match_index in np.arange(start=0, stop=np.shape(streak_data)[0], step=1):
  if (VERBOSE):
    print("".join(["match_index: (", str(match_index),")"]))
  
  