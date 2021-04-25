# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:55:59 2021

@author: Kaushik Dutta
"""
"""
Calculating The Form of Each Team for both HOME and AWAY Matches

"""

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

# sanity check: check if data file exists
if not (os.path.isfile(data_file)):
    print("".join(
        ["ERROR: data file (", str(data_file), ") does not exist."]), flush=True)
    sys.exit()

# load data: we assume this is processed data table with indices for team numbers
df = pd.read_csv(data_file, sep=",")
team_details = pd.read_csv(team_file, sep=",")
df.drop(df.iloc[:, 21:63], axis = 1, inplace = True)

no_teams = len(team_details.index) #Number of teams in the league
# Calculating the results for the match i.e. 0 -> Home Win , 1 -> Away Win , 2 -> Draw
res = df['FTR'] #Full Time Result (H=Home Win, D=Draw, A=Away Win)
home_away = df.iloc[:,1:3]
home_away = home_away.merge(res.to_frame(), left_index=True, right_index=True)
#3 columns: Home Team Index, Away Team Index, FTR (0, 1, or 2): size 380 x 3

form_home = np.zeros((no_teams,19)) #20 teams x 19
form_away = np.zeros((no_teams,19)) #20 teams x 19

gamma = 0.33    # Stealing Fraction
home_match_counter = np.zeros((no_teams,1),dtype = int)
away_match_counter = np.zeros((no_teams,1),dtype = int)

## Computation of the form for the dataset
for i in range(len(home_away.index)): #380 teams
    home_ind = home_away.iloc[i,0] - 1
    away_ind = home_away.iloc[i,1] - 1
    result = home_away.iloc[i,2]
    if result == 0:
        if home_match_counter[home_ind] == 0 and away_match_counter[away_ind] == 0:
            form_home[home_ind,home_match_counter[home_ind]] = 1 + gamma
            form_away[away_ind,away_match_counter[away_ind]] = 1 - gamma
        elif home_match_counter[home_ind] == 0 and away_match_counter[away_ind] != 0:
            form_home[home_ind,home_match_counter[home_ind]] = 1 + gamma*form_away[away_ind,away_match_counter[away_ind]-1]
            form_away[away_ind,away_match_counter[away_ind]] = form_away[away_ind,away_match_counter[away_ind]-1] - gamma*form_away[away_ind,away_match_counter[away_ind]-1]
        elif home_match_counter[home_ind] != 0 and away_match_counter[away_ind] == 0:
            form_home[home_ind,home_match_counter[home_ind]] = form_home[home_ind,home_match_counter[home_ind]-1] + gamma
            form_away[away_ind,away_match_counter[away_ind]] = 1 - gamma
        else:
            form_home[home_ind,home_match_counter[home_ind]] = form_home[home_ind,home_match_counter[home_ind]-1] + gamma*form_away[away_ind,away_match_counter[away_ind]-1]
            form_away[away_ind,away_match_counter[away_ind]] = form_away[away_ind,away_match_counter[away_ind]-1] - gamma*form_away[away_ind,away_match_counter[away_ind]-1]
        
        home_match_counter[home_ind] = home_match_counter[home_ind] + 1
        away_match_counter[away_ind] = away_match_counter[away_ind] + 1
    
    
    if result == 1:
        if home_match_counter[home_ind] == 0 and away_match_counter[away_ind] == 0:
            form_home[home_ind,home_match_counter[home_ind]] = 1 - gamma
            form_away[away_ind,away_match_counter[away_ind]] = 1 + gamma
        elif home_match_counter[home_ind] == 0 and away_match_counter[away_ind] != 0:
            form_home[home_ind,home_match_counter[home_ind]] = 1 - gamma
            form_away[away_ind,away_match_counter[away_ind]] = form_away[away_ind,away_match_counter[away_ind]-1] + gamma
        elif home_match_counter[home_ind] != 0 and away_match_counter[away_ind] == 0:
            form_home[home_ind,home_match_counter[home_ind]] = form_home[home_ind,home_match_counter[home_ind]-1]  - gamma*form_home[home_ind,home_match_counter[home_ind]-1]
            form_away[away_ind,away_match_counter[away_ind]] = 1 + gamma*form_home[home_ind,home_match_counter[home_ind]-1]
        else:
            form_home[home_ind,home_match_counter[home_ind]] = form_home[home_ind,home_match_counter[home_ind]-1] - gamma*form_home[home_ind,home_match_counter[home_ind]-1]
            form_away[away_ind,away_match_counter[away_ind]] = form_away[away_ind,away_match_counter[away_ind]-1] + gamma*form_home[home_ind,home_match_counter[home_ind]-1]
        
        home_match_counter[home_ind] = home_match_counter[home_ind] + 1
        away_match_counter[away_ind] = away_match_counter[away_ind] + 1
            
    if result == 2:
        if home_match_counter[home_ind] == 0 and away_match_counter[away_ind] == 0:
            form_home[home_ind,home_match_counter[home_ind]] = 1
            form_away[away_ind,away_match_counter[away_ind]] = 1
        elif home_match_counter[home_ind] == 0 and away_match_counter[away_ind] != 0:
            form_home[home_ind,home_match_counter[home_ind]] = 1 - gamma*(1 - form_away[away_ind,away_match_counter[away_ind]-1])
            form_away[away_ind,away_match_counter[away_ind]] = form_away[away_ind,away_match_counter[away_ind]-1] - gamma*(form_away[away_ind,away_match_counter[away_ind]-1] - 1)
        elif home_match_counter[home_ind] != 0 and away_match_counter[away_ind] == 0:
            form_home[home_ind,home_match_counter[home_ind]] = form_home[home_ind,home_match_counter[home_ind]-1] - gamma*(form_home[home_ind,home_match_counter[home_ind]-1] - 1)
            form_away[away_ind,away_match_counter[away_ind]] = 1 - gamma*(1 - form_home[home_ind,home_match_counter[home_ind]-1])
        else:
            form_home[home_ind,home_match_counter[home_ind]] = form_home[home_ind,home_match_counter[home_ind]-1] - gamma*(form_home[home_ind,home_match_counter[home_ind]-1] - form_away[away_ind,away_match_counter[away_ind]-1])
            form_away[away_ind,away_match_counter[away_ind]] = form_away[away_ind,away_match_counter[away_ind]-1] - gamma*(form_away[away_ind,away_match_counter[away_ind]-1] - form_home[home_ind,home_match_counter[home_ind]-1])
            
        home_match_counter[home_ind] = home_match_counter[home_ind] + 1
        away_match_counter[away_ind] = away_match_counter[away_ind] + 1
            

## Caluclating the per game form for each team

form = np.zeros((len(home_away.index),2))
for i in range(no_teams):
    counter1 = 0
    counter2 = 0
    for j in range(len(home_away.index)):
        if home_away.iloc[j,0] == i+1:
            form[j,0] = form_home[i,counter1]
            counter1 = counter1 + 1
        if home_away.iloc[j,1] == i+1:
            form[j,1] = form_away[i,counter2]
            counter2 = counter2 + 1
    
## Inculcating the home and away form in main table
df['Home Form'] = form[:,0]
df['Away Form'] = form[:,1]
            
            
            
        
            
        
        
            
            
        
        
                
            
    
        
        
        
