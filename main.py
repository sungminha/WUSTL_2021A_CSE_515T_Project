"""
expect to run on python >3.7
"""
import os
import math
import warnings

# from IPython.display import Image, HTML
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc # I know folks are switching to "as pm" but I'm just not there yet

DATA_DIR = os.path.join(os.getcwd(), 'data')
CHART_DIR = os.path.join(os.getcwd(), 'charts')
data_file = os.path.join(DATA_DIR, 'final_18-19season.csv')
VERBOSE = False

#load data: we assume this is processed data table with indices for team numbers
df = pd.read_csv(data_file, sep=",")

if (VERBOSE):
  print("df: ")
  print(df)
print("".join(["Finished loading in data from ", str(data_file)]))

#select columns we want
observed_home_goals = df['FTHG']
observed_away_goals = df['FTAG']
home_team = df['HomeTeam'] -1 #indexing should start at zero not one
away_team = df['AwayTeam'] -1 #indexing should start at zero not one
num_teams = len(home_team.unique())
num_games = len(home_team)

if (VERBOSE):
  print("observed_home_goals: ")
  print(observed_home_goals)
  print("observed_away_goals: ")
  print(observed_away_goals)
  print("home_team: ")
  print(home_team)
  print("away_team: ")
  print(away_team)
print("".join(["Finished finding variables of interest from data"]))

#starting points
g = df.groupby('HomeTeam')
att_starting_points = np.log(g['FTAG'].mean())
g = df.groupby('AwayTeam')
def_starting_points = -np.log(g['FTAG'].mean())

#hyperpriors
home = pymc.Normal('home', 0, .0001, value=0)
tau_att = pymc.Gamma('tau_att', .1, .1, value=10)
tau_def = pymc.Gamma('tau_def', .1, .1, value=10)
intercept = pymc.Normal('intercept', 0, .0001, value=0)

print("".join(["Defined hyperpriors"]))

#team-specific parameters
atts_star = pymc.Normal("atts_star", 
                        mu=0, 
                        tau=tau_att, 
                        size=num_teams, 
                        value=att_starting_points.values)
defs_star = pymc.Normal("defs_star", 
                        mu=0, 
                        tau=tau_def, 
                        size=num_teams, 
                        value=def_starting_points.values) 

print("".join(["Defined team-specific parameters"]))

# trick to code the sum to zero contraint
@pymc.deterministic
def atts(atts_star=atts_star):
    atts = atts_star.copy()
    atts = atts - np.mean(atts_star)
    return atts

@pymc.deterministic
def defs(defs_star=defs_star):
    defs = defs_star.copy()
    defs = defs - np.mean(defs_star)
    return defs

@pymc.deterministic
def home_theta(home_team=home_team, 
               away_team=away_team, 
               home=home, 
               atts=atts, 
               defs=defs, 
               intercept=intercept): 
    return np.exp(intercept + 
                  home + 
                  atts[home_team] + 
                  defs[away_team])
  
@pymc.deterministic
def away_theta(home_team=home_team, 
               away_team=away_team, 
               home=home, 
               atts=atts, 
               defs=defs, 
               intercept=intercept): 
    return np.exp(intercept + 
                  atts[away_team] + 
                  defs[home_team])   

print("".join(["Defined functions for att, def, and thetas"]))

home_goals = pymc.Poisson('home_goals', 
                          mu=home_theta, 
                          value=observed_home_goals, 
                          observed=True)
away_goals = pymc.Poisson('away_goals', 
                          mu=away_theta, 
                          value=observed_away_goals, 
                          observed=True)

print("".join(["Running MCMC"]))

mcmc = pymc.MCMC([home, intercept, tau_att, tau_def, 
                  home_theta, away_theta, 
                  atts_star, defs_star, atts, defs, 
                  home_goals, away_goals])
map_ = pymc.MAP( mcmc )
map_.fit()
iteration = 200000
burn = 40000
thin = 20
mcmc.sample(iter = iteration, burn = burn, thin = thin)

#generate plots
pymc.Matplot.plot(home)
plt.show()
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["home_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()
pymc.Matplot.plot(intercept)
plt.show()
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["intercept_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()
pymc.Matplot.plot(tau_att)
plt.show()
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_att_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()
pymc.Matplot.plot(tau_def)
plt.show()
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_def_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()
# Embed = Image(os.path.join(CHART_DIR, 'atts.png'))
# Embed