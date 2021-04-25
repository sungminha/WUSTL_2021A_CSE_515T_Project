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
import pymc  # I know folks are switching to "as pm" but I'm just not there yet

DATA_DIR = os.path.join(os.getcwd(), 'data')
CHART_DIR = os.path.join(os.getcwd(), 'charts')
data_file = os.path.join(DATA_DIR, 'final_data.csv')
team_file = os.path.join(DATA_DIR, 'team_index.csv')

# parameters for model training
iteration = 200000  # how many iterations?
burn = 4000  # how many to discard from the beginning of the iterations?
thin = 20  # how often to record?

VERBOSE = False  # more printouts
USE_MU_ATT_and_MU_DEF = False  # use instead of zero for mean of att and def

# Running Code starts here

# sanity check: check if data file exists
if not (os.path.isfile(data_file)):
    print("".join(
        ["ERROR: data file (", str(data_file), ") does not exist."]), flush=True)
    sys.exit()

# load data: we assume this is processed data table with indices for team numbers
df = pd.read_csv(data_file, sep=",")
team_details = pd.read_csv(team_file, sep=",")

if (VERBOSE):
    print("df: ", flush=True)
    print(df, flush=True)
print("".join(["Finished loading in data from ", str(data_file)]), flush=True)

# select columns we want
observed_home_goals = df['FTHG']
observed_away_goals = df['FTAG']

observed_home_shots = df['HS']
observed_away_shots = df['AS']
observed_home_shots_target = df['HST']
observed_away_shots_target = df['AST']
observed_home_corners = df['HC']
observed_away_corners = df['AC']
observed_home_fouls = df['HF']
observed_away_fouls = df['AF']
observed_home_yc = df['HY']
observed_away_yc = df['AY']
observed_home_rc = df['HR']
observed_away_rc = df['AR']


home_team = df['HomeTeam'] - 1  # indexing should start at zero not one
away_team = df['AwayTeam'] - 1  # indexing should start at zero not one
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

# starting points
g = df.groupby('HomeTeam')
att_starting_points = np.log(g['FTHG'].mean())
att_shots_starting_points = np.log(g['HS'].mean())
att_shots_target_starting_points = np.log(g['HST'].mean())
att_corners_starting_points = np.log(g['HC'].mean())
att_fouls_starting_points = np.log(g['HF'].mean())
att_yc_starting_points = np.log(g['HY'].mean())
att_rc_starting_points = np.log(g['HR'].mean())

g = df.groupby('AwayTeam')
def_starting_points = -np.log(g['FTAG'].mean())
defs_shots_starting_points = -np.log(g['AS'].mean())
defs_shots_target_starting_points = -np.log(g['AST'].mean())
defs_corners_starting_points = -np.log(g['AC'].mean())
defs_fouls_starting_points = -np.log(g['AF'].mean())
defs_yc_starting_points = -np.log(g['AY'].mean())
defs_rc_starting_points = -np.log(g['AR'].mean())



# hyperpriors
home = pymc.Normal('home', 0, .1, value=0)
tau_att = pymc.Gamma('tau_att', .1, .1, value=10)
tau_def = pymc.Gamma('tau_def', .1, .1, value=10)
intercept = pymc.Normal('intercept', 0, .1, value=0)

mu_att_shots = pymc.Normal('mu_att_shots', 0, .0001, value=0)
mu_def_shots = pymc.Normal('mu_def_shots', 0, .0001, value=0)
tau_att_shots = pymc.Gamma('tau_att_shots', .1, .1, value = 10)
tau_def_shots = pymc.Gamma('tau_def_shots', .1, .1, value = 10)

mu_att_shots_target = pymc.Normal('mu_att_shots_target', 0, .0001, value=0)
mu_def_shots_target = pymc.Normal('mu_def_shots_target', 0, .0001, value=0)
tau_att_shots_target = pymc.Gamma('tau_att_shots_target', .1, .1, value = 10)
tau_def_shots_target = pymc.Gamma('tau_def_shots_target', .1, .1, value = 10)

mu_att_corners = pymc.Normal('mu_att_corners', 0, .0001, value=0)
mu_def_corners = pymc.Normal('mu_def_corners', 0, .0001, value=0)
tau_att_corners = pymc.Gamma('tau_att_corners', .1, .1, value = 10)
tau_def_corners = pymc.Gamma('tau_def_corners', .1, .1, value = 10)

mu_att_fouls = pymc.Normal('mu_att_fouls', 0, .0001, value=0)
mu_def_fouls = pymc.Normal('mu_def_fouls', 0, .0001, value=0)
tau_att_fouls = pymc.Gamma('tau_att_fouls', .1, .1, value = 10)
tau_def_fouls = pymc.Gamma('tau_def_fouls', .1, .1, value = 10)

mu_att_yc = pymc.Normal('mu_att_yc', 0, .0001, value=0)
mu_def_yc = pymc.Normal('mu_def_yc', 0, .0001, value=0)
tau_att_yc = pymc.Gamma('tau_att_yc', .1, .1, value = 10)
tau_def_yc = pymc.Gamma('tau_def_yc', .1, .1, value = 10)

mu_att_rc = pymc.Normal('mu_att_rc', 0, .0001, value=0)
mu_def_rc = pymc.Normal('mu_def_rc', 0, .0001, value=0)
tau_att_rc = pymc.Gamma('tau_att_rc', .1, .1, value = 10)
tau_def_rc = pymc.Gamma('tau_def_rc', .1, .1, value = 10)


# original paper without tweaks
if (USE_MU_ATT_and_MU_DEF):
    mu_att = pymc.Normal('mu_att', 0, .0001, value=0)
    mu_def = pymc.Normal('mu_def', 0, .0001, value=0)

print("".join(["Defined hyperpriors"]), flush=True)

# team-specific parameters
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

    
atts_shots_star = pymc.Normal("atts_shots_star",
                              mu = mu_att_shots,
                              tau = tau_att_shots,
                              size = num_teams,
                              value = att_shots_starting_points.values)
defs_shots_star = pymc.Normal("defs_shots_star",
                              mu = mu_def_shots,
                              tau = tau_def_shots,
                              size = num_teams,
                              value = defs_shots_starting_points.values)

atts_shots_target_star = pymc.Normal("atts_shots_target_star",
                              mu = mu_att_shots_target,
                              tau = tau_att_shots_target,
                              size = num_teams,
                              value = att_shots_target_starting_points.values)
defs_shots_target_star = pymc.Normal("defs_shots_target_star",
                              mu = mu_def_shots_target,
                              tau = tau_def_shots_target,
                              size = num_teams,
                              value = defs_shots_target_starting_points.values)


atts_corner_star = pymc.Normal("atts_corner_star",
                              mu = mu_att_corners,
                              tau = tau_att_corners,
                              size = num_teams,
                              value = att_corners_starting_points.values)
defs_corner_star = pymc.Normal("defs_corner_star",
                              mu = mu_def_corners,
                              tau = tau_def_corners,
                              size = num_teams,
                              value = defs_corners_starting_points.values)

atts_fouls_star = pymc.Normal("atts_fouls_star",
                              mu = mu_att_fouls,
                              tau = tau_att_fouls,
                              size = num_teams,
                              value = att_fouls_starting_points.values)
defs_fouls_star = pymc.Normal("defs_fouls_star",
                              mu = mu_def_fouls,
                              tau = tau_def_fouls,
                              size = num_teams,
                              value = defs_fouls_starting_points.values)

atts_yc_star = pymc.Normal("atts_yc_star",
                              mu = mu_att_yc,
                              tau = tau_att_yc,
                              size = num_teams,
                              value = att_yc_starting_points.values)
defs_yc_star = pymc.Normal("defs_yc_star",
                              mu = mu_def_yc,
                              tau = tau_def_yc,
                              size = num_teams,
                              value = defs_yc_starting_points.values)

atts_rc_star = pymc.Normal("atts_rc_star",
                              mu = mu_att_rc,
                              tau = tau_att_rc,
                              size = num_teams,
                              value = att_rc_starting_points.values)
defs_rc_star = pymc.Normal("defs_rc_star",
                              mu = mu_def_rc,
                              tau = tau_def_rc,
                              size = num_teams,
                              value = defs_rc_starting_points.values)

                              
print("".join(["Defined team-specific parameters"]), flush=True)

# trick to code the sum to zero contraint


@pymc.deterministic
def atts(atts_star=atts_star):
    atts = atts_star.copy()
    atts = atts - np.mean(atts_star)
    return atts

@pymc.deterministic
def atts_shots(atts_shots_star = atts_shots_star):
    atts_shots = atts_shots_star.copy()
    atts_shots = atts_shots - np.mean(atts_shots_star)
    return atts_shots

@pymc.deterministic
def atts_shots_target(atts_shots_target_star = atts_shots_target_star):
    atts_shots_target = atts_shots_target_star.copy()
    atts_shots_target = atts_shots_target - np.mean(atts_shots_target_star)
    return atts_shots_target

@pymc.deterministic
def atts_corners(atts_corner_star = atts_corner_star):
    atts_corners = atts_corner_star.copy()
    atts_corners = atts_corners - np.mean(atts_corner_star)
    return atts_corners

@pymc.deterministic
def atts_fouls(atts_fouls_star = atts_fouls_star):
    atts_fouls = atts_fouls_star.copy()
    atts_fouls = atts_fouls - np.mean(atts_fouls_star)
    return atts_fouls

@pymc.deterministic
def atts_yc(atts_yc_star = atts_yc_star):
    atts_yc = atts_yc_star.copy()
    atts_yc = atts_yc - np.mean(atts_yc_star)
    return atts_yc

@pymc.deterministic
def atts_rc(atts_rc_star = atts_rc_star):
    atts_rc = atts_rc_star.copy()
    atts_rc = atts_rc - np.mean(atts_rc_star)
    return atts_rc



@pymc.deterministic
def defs(defs_star=defs_star):
    defs = defs_star.copy()
    defs = defs - np.mean(defs_star)
    return defs

@pymc.deterministic
def defs_shots(defs_shots_star = defs_shots_star):
    defs_shots = defs_shots_star.copy()
    defs_shots = defs_shots - np.mean(defs_shots_star)
    return defs_shots

@pymc.deterministic
def defs_shots_target(defs_shots_target_star = defs_shots_target_star):
    defs_shots_target = defs_shots_target_star.copy()
    defs_shots_target = defs_shots_target - np.mean(defs_shots_target_star)
    return defs_shots_target

@pymc.deterministic
def defs_corners(defs_corner_star = defs_corner_star):
    defs_corners = defs_corner_star.copy()
    defs_corners = defs_corners - np.mean(defs_corner_star)
    return defs_corners

@pymc.deterministic
def defs_fouls(defs_fouls_star = defs_fouls_star):
    defs_fouls = defs_fouls_star.copy()
    defs_fouls = defs_fouls - np.mean(defs_fouls_star)
    return defs_fouls

@pymc.deterministic
def defs_yc(defs_yc_star = defs_yc_star):
    defs_yc = defs_yc_star.copy()
    defs_yc = defs_yc - np.mean(defs_yc_star)
    return defs_yc

@pymc.deterministic
def defs_rc(defs_rc_star = defs_rc_star):
    defs_rc = defs_rc_star.copy()
    defs_rc = defs_rc - np.mean(defs_rc_star)
    return defs_rc
# To-Do: try replacing with Skellum


@pymc.deterministic
def home_theta(home_team=home_team,
               away_team=away_team,
               home=home,
               atts=atts,
               defs=defs,
               atts_shots=atts_shots,
               defs_shots=defs_shots,
               atts_shots_target = atts_shots_target,
               defs_shots_target = defs_shots_target,
               atts_corners = atts_corners,
               defs_corners = defs_corners,
               atts_fouls = atts_fouls,
               atts_yc = atts_yc,
               atts_rc = atts_rc,
               intercept=intercept):
    return np.exp(intercept +
                  home +
                  atts[home_team] +
                  defs[away_team] +
                  atts_shots[home_team] + 
                  defs_shots[away_team]+
                  atts_shots_target[home_team]+
                  defs_shots_target[away_team]+
                  atts_corners[home_team]+
                  defs_corners[away_team]+
                  atts_fouls[home_team]+
                  defs_fouls[away_team]+
                  atts_yc[home_team]+
                  defs_yc[away_team]+
                  atts_rc[home_team]+
                  defs_rc[away_team])


@pymc.deterministic
def away_theta(home_team=home_team,
               away_team=away_team,
               home=home,
               atts=atts,
               defs=defs,
               atts_shots=atts_shots,
               defs_shots=defs_shots,
               atts_shots_target = atts_shots_target,
               defs_shots_target = defs_shots_target,
               atts_corners = atts_corners,
               defs_corners = defs_corners,
               atts_fouls = atts_fouls,
               atts_yc = atts_yc,
               atts_rc = atts_rc,
               intercept=intercept):
    return np.exp(intercept +
                  atts[away_team] +
                  defs[home_team] +
                  atts_shots[away_team] + 
                  defs_shots[home_team]+
                  atts_shots_target[away_team]+
                  defs_shots_target[home_team]+
                  atts_corners[away_team]+
                  defs_corners[home_team]+
                  atts_fouls[away_team]+
                  defs_fouls[home_team]+
                  atts_yc[away_team]+
                  defs_yc[home_team]+
                  atts_rc[away_team]+
                  defs_rc[home_team])


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
                  atts_shots_star, defs_shots_star, atts_shots,defs_shots,
                  atts_shots_target_star, defs_shots_target_star, atts_shots_target, defs_shots_target,
                  atts_corner_star, defs_corner_star, atts_corners, defs_corners,
                  atts_fouls_star, defs_fouls_star, atts_fouls, defs_fouls,
                  atts_yc_star, defs_yc_star, atts_yc, defs_yc,
                  atts_rc_star, defs_rc_star, atts_rc, defs_rc,
                  home_goals, away_goals])
map_ = pymc.MAP(mcmc)
map_.fit()
mcmc.sample(iter=iteration, burn=burn, thin=thin)

# save statistics
mcmc.write_csv(os.path.join(CHART_DIR, "".join(["stats_", str(iteration), "_", str(
    burn), "_", str(thin), ".csv"])), variables=["home", "intercept", "tau_att", "tau_def"])

# generate plots

# generate plots: home - all (trace, acorr, hist)
pymc.Matplot.plot(home)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["home_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(home)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["home_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(home)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["home_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()


# generate plots: intercept - all (trace, acorr, hist)
pymc.Matplot.plot(intercept)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["intercept_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(intercept)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["intercept_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(intercept)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["intercept_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()


# generate plots: tau_att - all (trace, acorr, hist)
pymc.Matplot.plot(tau_att)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["tau_att_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(tau_att)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["tau_att_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(tau_att)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["tau_att_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()


# generate plots: tau_def - all (trace, acorr, hist)
pymc.Matplot.plot(tau_def)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["tau_def_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(tau_def)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["tau_def_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(tau_def)
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["tau_def_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

total_stat = mcmc.stats()
attack_param = total_stat['atts']
mean_attack_team = attack_param['mean']

defence_param = total_stat['defs']
mean_defence_team = defence_param['mean']

# making predictions
# observed_season = DATA_DIR + 'premier_league_13_14_table.csv'
# observed_season = data_file
# df_observed = pd.read_csv(observed_season)

# teams for teams indexing
teams_file = os.path.join(DATA_DIR, 'team_index.csv')
if not (os.path.isfile(teams_file)):
    print("".join(
        ["ERROR: teams file (", str(teams_file), ") does not exist."]), flush=True)
    sys.exit()
teams = pd.read_csv(teams_file)

df_avg = pd.DataFrame({
                        'team': teams.Team.values,
                        'avg_att': atts.stats()['mean'],
                        'avg_def': defs.stats()['mean'],
                      },
                      index=teams.index)
if (VERBOSE):
  print("df_avg:")
  print(df_avg)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_avg.avg_att,  df_avg.avg_def, 'o')

for label, x, y in zip(df_avg.team.values, df_avg.avg_att.values, df_avg.avg_def.values):
    ax.annotate(label, xy=(x, y), xytext = (-5,5), textcoords='offset points')
ax.set_title('Attack vs Defense average effect: 18-19 Premier League')
ax.set_xlabel('Average attack effect')
ax.set_ylabel('Average defense effect')
ax.legend()
#plt.show()

#save figure of scatterplot of att vs def
plt.savefig(fname=os.path.join(CHART_DIR, "".join(
    ["scatter_avg_att_avg_def_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()