"""
expect to run on python >3.7
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
import pymc # I know folks are switching to "as pm" but I'm just not there yet

DATA_DIR = os.path.join(os.getcwd(), 'data')
CHART_DIR = os.path.join(os.getcwd(), 'charts')
data_file = os.path.join(DATA_DIR, 'final_18-19season.csv')

#parameters for model training
iteration = 200 #how many iterations?
burn = 40 #how many to discard from the beginning of the iterations?
thin = 20 #how often to record?

VERBOSE = False #more printouts
USE_MU_ATT_and_MU_DEF = False #use instead of zero for mean of att and def

##Running Code starts here

#sanity check: check if data file exists
if not (os.path.isfile(data_file)):
  print("".join(["ERROR: data file (", str(data_file), ") does not exist."]), flush=True)
  sys.exit()

#load data: we assume this is processed data table with indices for team numbers
df = pd.read_csv(data_file, sep=",")

if (VERBOSE):
  print("df: ", flush=True)
  print(df, flush=True)
print("".join(["Finished loading in data from ", str(data_file)]), flush=True)

#select columns we want
observed_home_goals = df['FTHG']
observed_away_goals = df['FTAG']
home_team = df['HomeTeam'] -1 #indexing should start at zero not one
away_team = df['AwayTeam'] -1 #indexing should start at zero not one
num_teams = len(home_team.unique())
num_games = len(home_team)

if (VERBOSE):
  print("observed_home_goals: ", flush=True)
  print(observed_home_goals, flush=True)
  print("observed_away_goals: ", flush=True)
  print(observed_away_goals, flush=True)
  print("home_team: ", flush=True)
  print(home_team, flush=True)
  print("away_team: ", flush=True)
  print(away_team, flush=True)
print("".join(["Finished finding variables of interest from data"]), flush=True)

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

#original paper without tweaks
if (USE_MU_ATT_and_MU_DEF):
  mu_att = pymc.Normal('mu_att', 0, .0001, value=0)
  mu_def = pymc.Normal('mu_def', 0, .0001, value=0)

print("".join(["Defined hyperpriors"]), flush=True)

#team-specific parameters
if (USE_MU_ATT_and_MU_DEF):
  atts_star = pymc.Normal("atts_star", 
                          mu=mu_att, 
                          tau=tau_att, 
                          size=num_teams, 
                          value=att_starting_points.values)
  defs_star = pymc.Normal("defs_star", 
                          mu=mu_def, 
                          tau=tau_def, 
                          size=num_teams, 
                          value=def_starting_points.values) 
else:
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
print("".join(["Defined team-specific parameters"]), flush=True)

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

#To-Do: try replacing with Skellum
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

print("".join(["Defined functions for att, def, and thetas"]), flush=True)

home_goals = pymc.Poisson('home_goals', 
                          mu=home_theta, 
                          value=observed_home_goals, 
                          observed=True)
away_goals = pymc.Poisson('away_goals', 
                          mu=away_theta, 
                          value=observed_away_goals, 
                          observed=True)

print("".join(["Running MCMC"]), flush=True)

mcmc = pymc.MCMC([home, intercept, tau_att, tau_def, 
                  home_theta, away_theta, 
                  atts_star, defs_star, atts, defs, 
                  home_goals, away_goals])
map_ = pymc.MAP( mcmc )
map_.fit()
mcmc.sample(iter = iteration, burn = burn, thin = thin)

#save statistics
mcmc.write_csv(os.path.join(CHART_DIR, "".join(["stats_", str(iteration), "_", str(burn), "_", str(thin), ".csv"])), variables=["home", "intercept", "tau_att", "tau_def"])

#generate plots

#generate plots: home - all (trace, acorr, hist)
pymc.Matplot.plot(home)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["home_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

#plot individual (trace, hist)
pymc.Matplot.trace(home)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["home_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

pymc.Matplot.histogram(home)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["home_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()


#generate plots: intercept - all (trace, acorr, hist)
pymc.Matplot.plot(intercept)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["intercept_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

#plot individual (trace, hist)
pymc.Matplot.trace(intercept)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["intercept_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

pymc.Matplot.histogram(intercept)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["intercept_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()



#generate plots: tau_att - all (trace, acorr, hist)
pymc.Matplot.plot(tau_att)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_att_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

#plot individual (trace, hist)
pymc.Matplot.trace(tau_att)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_att_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

pymc.Matplot.histogram(tau_att)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_att_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()



#generate plots: tau_def - all (trace, acorr, hist)
pymc.Matplot.plot(tau_def)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_def_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

#plot individual (trace, hist)
pymc.Matplot.trace(tau_def)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_def_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

pymc.Matplot.histogram(tau_def)
plt.savefig(fname = os.path.join(CHART_DIR, "".join(["tau_def_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"]) ))
plt.close()

#making predictions
# observed_season = DATA_DIR + 'premier_league_13_14_table.csv'
observed_season = data_file
df_observed = pd.read_csv(observed_season)

#teams for teams indexing
teams_file = os.path.join(DATA_DIR, 'team_index.csv')
if not (os.path.isfile(teams_file)):
  print("".join(["ERROR: teams file (", str(teams_file), ") does not exist."]), flush=True)
  sys.exit()
teams = pd.read_csv(teams_file)

df_avg = pd.DataFrame({'avg_att': atts.stats()['mean'],
                       'avg_def': defs.stats()['mean']} )
print("df_avg:")
print(df_avg)
print("df_observed")
print(df_observed)
df_avg = pd.merge(df_avg, df_observed, left_index=True, right_on='HomeTeam', how='left')

fig, ax = plt.subplots(figsize=(8,6))

# for label, x, y in zip(df_avg[0].values, df_avg.avg_att.values, df_avg.avg_def.values):
#     ax.annotate(label, xy=(x,y), xytext = (-5,5), textcoords = 'offset points')
# ax.set_title('Attack vs Defense avg effect: 13-14 Premier League')
# ax.set_xlabel('Avg attack effect')
# ax.set_ylabel('Avg defense effect')
# ax.legend()