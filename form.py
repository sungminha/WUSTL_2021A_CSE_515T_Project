# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:55:59 2021

@author: Kaushik Dutta
"""

import os
import sys
import math
import warnings

# from IPython.display import Image, HTML
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
res = df['FTR']
home_away = df.iloc[:,1:3]
home_away = home_away.merge(res.to_frame(), left_index=True, right_index=True)

form_home = np.zeros((no_teams,19))
form_home[:,0] = 1
form_away = np.zeros((no_teams,19))
form_away[:,0] = 1
gamma = 0.33    # Stealing Fraction
home_match_counter = np.zeros((no_teams,1),dtype = int)
away_match_counter = np.zeros((no_teams,1),dtype = int)
## Computation of the form for the dataset
for i in range(len(home_away.index)):
    home_ind = home_away.iloc[i,0] - 1
    away_ind = home_away.iloc[i,1] - 1
    result = home_away.iloc[i,2]
    if result==0:
        if home_match_counter[home_ind] == 0:
            form_home[home_ind,home_match_counter[home_ind]] = 1
        else:
            form_home[home_ind,home_match_counter[home_ind]] = form_home[home_ind,home_match_counter[home_ind]-1] + gamma*form_away[away_ind,away_match_counter[away_ind]-1]
                
            
    
        
        
        
