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
VERBOSE = False  # more printouts

k = 4 #streak from last 4 games

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
streak_home = pd.DataFrame(np.zeros(shape=(np.shape(streak_data)[0], np.shape(team_details)[0]))) #380 matches by 20 teams
streak_away = pd.DataFrame(np.zeros(shape=(np.shape(team_details)[0], np.shape(team_details)[0]))) #380 matches by 20 teams

for team_index in np.arange(start=1, stop=np.shape(team_details)[0]+1, step=1):
  if (VERBOSE):
    print("".join(["team_index: (", str(team_index),")"]))
  
  streak_data_home_index_match = streak_data.loc[streak_data['HomeTeam'] == int(team_index)]
  streak_data_away_index_match = streak_data.loc[streak_data['AwayTeam'] == int(team_index)]
  streak_data_index_match = pd.concat([streak_data_home_index_match, streak_data_away_index_match], axis=0).sort_values(by=['MatchNo'])
  #attach a column for streak
  streak_data_index_match["streak"] = 0
  streak_data_index_match["streak"] = streak_data_index_match["streak"].astype(float) #cast to float as the value may be float not integer when assigned
  del streak_data_home_index_match
  del streak_data_away_index_match

  if (VERBOSE):
    print("streak_data_index_match: ")
    print(streak_data_index_match)
    print(np.shape(streak_data_index_match))

  match_count = 1
  for match_index in streak_data_index_match['MatchNo']:
    if (VERBOSE):
      print("".join(["team_index: (", str(team_index),")\t| match_index: (", str(match_index),")\t| match_count: (", str(match_count),")"]))

    if (match_count > k):
      temp = streak_data_index_match.loc[streak_data_index_match['MatchNo'] < int(match_index)]
      temp = temp.iloc[-k:]
      temp_home = temp.loc[temp["HomeTeam"] == int(team_index)]
      temp_away = temp.loc[temp["AwayTeam"] == int(team_index)]
    
      #intialize as zero
      sum_streak = 0

      #for home team win
      test = temp_home.loc[temp_home["FTR"] == 0]
      if not (test.empty):
        sum_streak = sum_streak + np.shape(test)[0]*3
      del test

      #for home team draw
      test = temp_home.loc[temp_home["FTR"] == 2]
      if not (test.empty):
        sum_streak = sum_streak + np.shape(test)[0]*1
      del test

      #for away team win
      test = temp_away.loc[temp_away["FTR"] == 1]
      if not (test.empty):
        sum_streak = sum_streak + np.shape(test)[0]*3
      del test

      #for away team draw
      test = temp_away.loc[temp_away["FTR"] == 2]
      if not (test.empty):
        sum_streak = sum_streak + np.shape(test)[0]*1
      del test

      #divide by 3
      sum_streak = float(sum_streak) / float(3*k)

      if (VERBOSE):
        print("temp: ")
        print(temp, flush=True)
        print("sum_streak: ")
        print(sum_streak, flush=True)
      streak_data_index_match.at[streak_data_index_match.loc[streak_data_index_match['MatchNo'] == int(match_index)].index.item(), 'streak'] = float(sum_streak)
      del temp

    match_count = match_count + 1

  print(streak_data_index_match)
  print(np.shape(streak_data_index_match))

#   streak_home.at[streak_home.loc[streak_home['MatchNo'] < int(match_index)].index.item()] = sum_streak
#   print("streak_home: ")
#   print(streak_home)

  

  del streak_data_index_match