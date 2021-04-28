"""
expect to run on python >3.7
"""
import os
import sys
import math
# import warnings

# from IPython.display import Image, HTML
import datetime
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pymc  # I know folks are switching to "as pm" but I'm just not there yet

VERBOSE = True
DEBUG = False

DATA_DIR = os.path.join(os.getcwd(), 'data')
CHART_DIR = os.path.join(os.getcwd(), 'charts')
current_datetime_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
data_file = os.path.join(DATA_DIR, 'final_data.csv')
final_data_file = os.path.join(DATA_DIR, 'final_data_original_data_format.csv')
team_file = os.path.join(DATA_DIR, 'team_index.csv')
final_team_file = os.path.join(DATA_DIR, 'team_index_original_data_format.csv')
del current_datetime_string

#load data
df = pd.read_csv(data_file, sep=",")
df_team = pd.read_csv(team_file, sep=",")
print("df:")
print(df)


df = df[['MatchNo','HomeTeam','AwayTeam','FTHG','FTAG']]
df = pd.merge(df, df_team, left_on = 'HomeTeam', right_on = 'i')
df = df.rename(columns = {"i":"i_home"})
df = df.drop(columns="HomeTeam")
df = df.rename(columns = {"Team":"HomeTeam"})
df = df.drop(columns="Unnamed: 0")
if (VERBOSE):
  print("df:")
  print(df)


df = pd.merge(df, df_team, left_on = 'AwayTeam', right_on = 'i')
df = df.rename(columns = {"i":"i_away"})
df = df.drop(columns="AwayTeam")
df = df.rename(columns = {"Team":"AwayTeam"})
df = df.drop(columns="Unnamed: 0")

if (VERBOSE):
  print("df:")
  print(df)

num_teams = np.shape(np.unique(df['HomeTeam'].values))[0]

if (VERBOSE):
  print("num_teams:")
  print(num_teams)
  
final_table_np = np.zeros(shape = (num_teams,num_teams))
final_table = pd.DataFrame(final_table_np)
del final_table_np
final_table.columns = np.sort(np.unique(df['AwayTeam'].values))
final_table.index = np.sort(np.unique(df['HomeTeam'].values))
final_table["Home \ Away[1]"] = np.sort(np.unique(df['HomeTeam'].values))

for home_team in np.sort(np.unique(df['HomeTeam'].values)):
  for away_team in np.sort(np.unique(df['AwayTeam'].values)):
    if (DEBUG):
      print("".join(["home_team: ", str(home_team)]))
      print("".join(["away_team: ", str(away_team)]))

    key1 = df["HomeTeam"] == home_team
    key2 = df["AwayTeam"] == away_team
    match = df[key1 & key2]
  
    if (DEBUG):
      print(match)

    del key1
    del key2
    if (home_team == away_team):
      result = " "

    else:
      home_score = match["FTHG"].values[0]
      away_score = match["FTAG"].values[0]
      
      result = "".join([str(home_score),"-",str(away_score)])
      
    final_table.loc[home_team, away_team] = result

final_table.index = final_table["Home \ Away[1]"]
final_table.index = np.arange(np.shape(final_table)[0])
final_table_columns = np.sort(np.unique(df['AwayTeam'].values))
final_table_columns = np.insert(final_table_columns, 0, "Home \ Away[1]", axis=0)
final_table = final_table[final_table_columns]

print("final_table")
print(final_table)
final_table.to_csv(final_data_file, index=False, sep = "\t")

#prepare premier_league_13_14_table.csv equivalent with Pts, W, L

final_team_columns = (["W","D","L","GF","GA","Pts"])
final_team_np = np.zeros(shape = (num_teams, np.shape(final_team_columns)[0]))
final_team = pd.DataFrame(final_team_np)
del final_team_np
final_team.columns = final_team_columns
final_team.index = np.sort(np.unique(df['HomeTeam'].values))
final_team['Team'] = np.sort(np.unique(df['HomeTeam'].values))

if (VERBOSE):
  print("final_team:")
  print(np.shape(final_team))
  print(final_team)
  print("df_team:")
  print(np.shape(df_team))
  print(df_team)

final_team = pd.merge(final_team, df_team, left_on = "Team", right_on = 'Team')
final_team = final_team.drop(columns="Unnamed: 0")
if (VERBOSE):
  print("final_team:")
  print(np.shape(final_team))
  print(final_team)
# final_team = final_team.drop(columns="Home \ Away[1]")

if (VERBOSE):
  print("final_team:")
  print(np.shape(final_team))
  print(final_team)

final_team = final_team[["Team", 'i', "W","D","L", 'GF', 'GA', 'Pts']]

final_team.index = final_team["Team"]
if (VERBOSE):
  print("final_team:")
  print(np.shape(final_team))
  print(final_team)
for teamname in np.sort(np.unique(df['HomeTeam'].values)):

  home_team_matches = df["HomeTeam"] == teamname
  home_team_matches_GF = np.sum(df[home_team_matches]['FTHG'])
  home_team_matches_GA = np.sum(df[home_team_matches]['FTAG'])
  home_team_win = np.sum((df[home_team_matches]['FTHG'] > df[home_team_matches]['FTAG']))
  home_team_draw = np.sum((df[home_team_matches]['FTHG'] == df[home_team_matches]['FTAG']))
  home_team_lose = np.sum((df[home_team_matches]['FTHG'] < df[home_team_matches]['FTAG']))

  away_team_matches = df["AwayTeam"] == teamname
  away_team_matches_GF = np.sum(df[away_team_matches]['FTAG'])
  away_team_matches_GA = np.sum(df[away_team_matches]['FTHG'])
  away_team_win = np.sum((df[away_team_matches]['FTAG'] > df[away_team_matches]['FTHG']))
  away_team_draw = np.sum((df[away_team_matches]['FTAG'] == df[away_team_matches]['FTHG']))
  away_team_lose = np.sum((df[away_team_matches]['FTAG'] < df[away_team_matches]['FTHG']))

  GF = home_team_matches_GF + away_team_matches_GF
  GA = home_team_matches_GA + away_team_matches_GA
  W = home_team_win + away_team_win
  D = home_team_draw + away_team_draw
  L = home_team_lose + away_team_lose
  Pts = 3 * W + D
  if (DEBUG):
    print("teamname:")
    print(teamname)
    print("final_team['Team']")
    print(final_team['Team'])

  print(final_team['GF'][teamname])
  final_team['GF'][teamname] = GF
  final_team['GA'][teamname] = GA

  final_team['W'][teamname] = W
  final_team['D'][teamname] = D
  final_team['L'][teamname] = L
  final_team['Pts'][teamname] = Pts

final_team = final_team.rename(columns = {"Team":"team"})
final_team = final_team.drop(columns="i")

if (VERBOSE):
  print("final_team:")
  print(np.shape(final_team))
  print(final_team)

final_team.to_csv(final_team_file, index=False, sep = ",")

