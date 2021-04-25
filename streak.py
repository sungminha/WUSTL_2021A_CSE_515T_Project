# -*- coding: utf-8 -*-

import os
import sys
#import math
#import warnings
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

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
del team_file
del data_file

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
del res
if (VERBOSE):
  print("streak_data: ")
  print(streak_data, flush=True)

#non-weighted streaks
# streak_home = pd.DataFrame(np.zeros(shape=(np.shape(streak_data)[0], np.shape(team_details)[0]+1))) #380 matches by 20 teams + 1
# streak_away = pd.DataFrame(np.zeros(shape=(np.shape(streak_data)[0], np.shape(team_details)[0]+1))) #380 matches by 20 teams + 1
streak_home = pd.DataFrame(np.zeros(shape=(np.shape(streak_data)[0], 4))) #380 matches by MatchNo, HomeTeam, AwayTeam, streak
streak_away = pd.DataFrame(np.zeros(shape=(np.shape(streak_data)[0], 4))) #380 matches by MatchNo, HomeTeam, AwayTeam, streak

#assign match index to left most column
# streak_columns = ['MatchNo']
# for i in np.arange(start=0, stop=team_details['i'].size, step=1):
#   streak_columns.append(team_details['i'][i])
# del i
streak_columns = ['MatchNo', 'HomeTeam', 'AwayTeam', 'streak']
streak_home.columns =streak_columns
streak_away.columns =streak_columns
streak_home['MatchNo'] = df['MatchNo']
streak_away['MatchNo'] = df['MatchNo']
streak_home['HomeTeam'] = df['HomeTeam']
streak_away['HomeTeam'] = df['HomeTeam']
streak_home['AwayTeam'] = df['AwayTeam']
streak_away['AwayTeam'] = df['AwayTeam']
del df
del streak_columns

for team_index in np.arange(start=1, stop=np.shape(team_details)[0]+1, step=1):
  if (VERBOSE):
    print("".join(["team_index: (", str(team_index),") - Generating streak_data_index_match"]))
  
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
      del temp_away
      del temp_home

      #divide by 3
      sum_streak = float(sum_streak) / float(3*k)

      if (VERBOSE):
        print("temp: ")
        print(temp, flush=True)
        print("sum_streak: ")
        print(sum_streak, flush=True)
      streak_data_index_match.at[streak_data_index_match.loc[streak_data_index_match['MatchNo'] == int(match_index)].index.item(), 'streak'] = float(sum_streak)
      del temp
      del sum_streak
    match_count = match_count + 1

    if (VERBOSE):
      print(streak_data_index_match)
      print(np.shape(streak_data_index_match))

  print("".join(["team_index: (", str(team_index),") - assigning to streak_home"]))
  streak_data_index_match_home = streak_data_index_match[streak_data_index_match['HomeTeam'] == team_index]
  # for match_index_per_team in np.arange(np.shape(streak_data_index_match_home['MatchNo'])[0]-1):
  for match_index_per_team in np.arange(np.shape(streak_data_index_match_home['MatchNo'])[0]):
    
    if (VERBOSE):
      print("".join(["match_index_per_team: (", str(match_index_per_team), ")"]))
      print("".join(["match_index_per_team: (", str(match_index_per_team), ")"]))
    
    # below is for finding matches between two matches of a team
    # index_match_1 = streak_home['MatchNo'] < streak_data_index_match_home['MatchNo'].iloc[match_index_per_team+1]
    # index_match_2 = streak_home['MatchNo'] >= streak_data_index_match_home['MatchNo'].iloc[match_index_per_team]
    # if (match_index_per_team == np.shape(streak_data_index_match_home['MatchNo'])[0]-2):
    #   index_match_1 = streak_home['MatchNo'] <= streak_home.loc[np.shape(streak_home)[0]-1, 'MatchNo'] #has to inlucde last row
    #   index_match_2 = streak_home['MatchNo'] >= streak_data_index_match_home['MatchNo'].iloc[match_index_per_team]

    # index_match = index_match_1 & index_match_2
    # del index_match_2
    # del index_match_1
    index_match = streak_home['MatchNo'] == streak_data_index_match_home['MatchNo'].iloc[match_index_per_team]
    
    if (VERBOSE):
      print("Before applying streak information")
      print(streak_home[['MatchNo', 'streak']][index_match])
    streak_home.loc[index_match, 'streak'] = streak_data_index_match_home['streak'].iloc[match_index_per_team]
    if (VERBOSE):
      print("After applying streak information")
      print(streak_home[['MatchNo', 'streak']][index_match])
    del index_match
    
  print("".join(["team_index: (", str(team_index),") - assigning to streak_away"]))
  streak_data_index_match_away = streak_data_index_match[streak_data_index_match['AwayTeam'] == team_index]
  for match_index_per_team in np.arange(np.shape(streak_data_index_match_away['MatchNo'])[0]):
    
    if (VERBOSE):
      print("".join(["match_index_per_team: (", str(match_index_per_team), ")"]))
      print("".join(["match_index_per_team: (", str(match_index_per_team), ")"]))
    
    
    # below is for finding matches between two matches of a team
    # index_match_1 = streak_away['MatchNo'] < streak_data_index_match_away['MatchNo'].iloc[match_index_per_team+1]
    # index_match_2 = streak_away['MatchNo'] >= streak_data_index_match_away['MatchNo'].iloc[match_index_per_team]
    # if (match_index_per_team == np.shape(streak_data_index_match_away['MatchNo'])[0]-2):
    #   index_match_1 = streak_away['MatchNo'] <= streak_home.loc[np.shape(streak_home)[0]-1, 'MatchNo'] #has to inlucde last row
    #   index_match_2 = streak_away['MatchNo'] >= streak_data_index_match_away['MatchNo'].iloc[match_index_per_team]

    # index_match = index_match_1 & index_match_2
    # del index_match_2
    # del index_match_1
    index_match = streak_away['MatchNo'] == streak_data_index_match_away['MatchNo'].iloc[match_index_per_team]

    if (VERBOSE):
      print("Before applying streak information")
      print(streak_away[['MatchNo', 'streak']][index_match])
    streak_away.loc[index_match, 'streak'] = streak_data_index_match_away['streak'].iloc[match_index_per_team]
    if (VERBOSE):
      print("After applying streak information")
      print(streak_away[['MatchNo', 'streak']][index_match])
    del index_match
    
del team_index
del match_index
del match_count
del match_index_per_team