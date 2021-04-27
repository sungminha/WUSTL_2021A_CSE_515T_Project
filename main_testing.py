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
import seaborn as sns
import matplotlib.pyplot as plt
import pymc  # I know folks are switching to "as pm" but I'm just not there yet

DATA_DIR = os.path.join(os.getcwd(), 'data')
CHART_DIR = os.path.join(os.getcwd(), 'charts')
current_datetime_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.getcwd(), "".join(["charts_", str(current_datetime_string)]))
os.mkdir(OUTPUT_DIR) #so that outputs will not be overwritten
data_file = os.path.join(DATA_DIR, 'final_data.csv')
# final_data_file = os.path.join(DATA_DIR, 'final_data.csv')
team_file = os.path.join(DATA_DIR, 'team_index.csv')
goal_scored_path = os.path.join(DATA_DIR, 'team_index_totalGoalsScored_totalGoalsTaken.csv')
del current_datetime_string

# parameters for model training
iteration = 200000  # how many iterations?
burn = 4000  # how many to discard from the beginning of the iterations?
thin = 20  # how often to record?

#for testing
iteration = 200  # how many iterations?
burn = 40  # how many to discard from the beginning of the iterations?
thin = 2  # how often to record?


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
test1 = g['HR'].mean()
test1[test1==0] = 1
att_rc_starting_points = np.log(test1)

g = df.groupby('AwayTeam')
def_starting_points = -np.log(g['FTAG'].mean())
defs_shots_starting_points = -np.log(g['AS'].mean())
defs_shots_target_starting_points = -np.log(g['AST'].mean())
defs_corners_starting_points = -np.log(g['AC'].mean())
defs_fouls_starting_points = -np.log(g['AF'].mean())
defs_yc_starting_points = -np.log(g['AY'].mean())
test2 = g['AR'].mean()
test2[test2==0] = 1
defs_rc_starting_points = -np.log(test2)



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
               defs_fouls = defs_fouls,
               atts_yc = atts_yc,
               defs_yc = defs_yc,
               atts_rc = atts_rc,
               defs_rc = defs_rc,
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
                  defs_corners[away_team]-
                  (atts_fouls[home_team]+
                  defs_fouls[away_team]+
                  atts_yc[home_team]+
                  defs_yc[away_team]+
                  atts_rc[home_team]+
                  defs_rc[away_team]))


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
               defs_fouls = defs_fouls,
               atts_yc = atts_yc,
               defs_yc = defs_yc,
               atts_rc = atts_rc,
               defs_rc = defs_rc,
               intercept=intercept):
    return np.exp(intercept +
                  atts[away_team] +
                  defs[home_team] +
                  atts_shots[away_team] + 
                  defs_shots[home_team]+
                  atts_shots_target[away_team]+
                  defs_shots_target[home_team]+
                  atts_corners[away_team]+
                  defs_corners[home_team]-
                  (atts_fouls[away_team]+
                  defs_fouls[home_team]+
                  atts_yc[away_team]+
                  defs_yc[home_team]+
                  atts_rc[away_team]+
                  defs_rc[home_team]))


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
mcmc.write_csv(os.path.join(OUTPUT_DIR, "".join(["stats_", str(iteration), "_", str(
    burn), "_", str(thin), ".csv"])), variables=["home", "intercept", "tau_att", "tau_def"])

## generate plots

# generate plots: home - all (trace, acorr, hist)
pymc.Matplot.plot(home)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["home_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(home)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["home_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(home)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["home_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()


# generate plots: intercept - all (trace, acorr, hist)
pymc.Matplot.plot(intercept)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["intercept_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(intercept)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["intercept_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(intercept)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["intercept_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()


# generate plots: tau_att - all (trace, acorr, hist)
pymc.Matplot.plot(tau_att)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["tau_att_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(tau_att)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["tau_att_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(tau_att)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["tau_att_histogram_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()


# generate plots: tau_def - all (trace, acorr, hist)
pymc.Matplot.plot(tau_def)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["tau_def_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

# plot individual (trace, hist)
pymc.Matplot.trace(tau_def)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["tau_def_trace_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

pymc.Matplot.histogram(tau_def)
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
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
plt.savefig(fname=os.path.join(OUTPUT_DIR, "".join(
    ["scatter_avg_att_avg_def_", str(iteration), "_", str(burn), "_", str(thin), ".png"])))
plt.close()

#df_observed append pts if it does not exist
df_observed = pd.read_csv(goal_scored_path, sep=",")
df_data = pd.read_csv(data_file, sep=",")
df_team = pd.read_csv(team_file, sep=",")

if not 'Pts' in df_observed.columns: #if it already exists, no need to calculate and append
  points = np.zeros(shape = (np.shape(df_team)[0]))
  for team_index in np.arange(start=1, stop=np.shape(team_details)[0]+1, step=1): 
    match_home = df_data[df_data['HomeTeam'] == team_index]
    win_home = match_home[match_home['FTHG'] > match_home['FTAG']]
    del match_home
    match_away = df_data[df_data['AwayTeam'] == team_index]
    win_away = match_away[match_away['FTHG'] < match_away['FTAG']]
    del match_away
    match_draw_home = df_data[df_data['HomeTeam'] == team_index]
    draw_home = match_draw_home[match_draw_home['FTHG'] == match_draw_home['FTAG']]
    del match_draw_home
    match_draw_away = df_data[df_data['AwayTeam'] == team_index]
    draw_away = match_draw_away[match_draw_away['FTHG'] == match_draw_away['FTAG']]
    del match_draw_away
    match_lose_home = df_data[df_data['HomeTeam'] == team_index]
    lose_home = match_lose_home[match_lose_home['FTHG'] < match_lose_home['FTAG']]
    del match_lose_home
    match_lose_away = df_data[df_data['AwayTeam'] == team_index]
    lose_away = match_lose_away[match_lose_away['FTHG'] > match_lose_away['FTAG']]
    del match_lose_away
    win = pd.concat([win_home, win_away], ignore_index = True)
    win.sort_values(by='MatchNo')
    
    draw = pd.concat([draw_home, draw_away], ignore_index = True)
    draw.sort_values(by='MatchNo')
    
    lose = pd.concat([lose_home, lose_away], ignore_index = True)
    lose.sort_values(by='MatchNo')
    
    points[team_index-1] = 3 * np.shape(win)[0] + 1 * np.shape(draw)[0]
  
  df_observed['Pts'] = points
  df_observed.to_csv(goal_scored_path, index=False) #save

###
# simulate the seasons
def simulate_season():
    """
    Simulate a season once, using one random draw from the mcmc chain. 
    """
    num_samples = atts.trace().shape[0] #atts.trace() is [80, 20] shape
    draw = np.random.randint(0, num_samples) #9800, for (iteration - burn) / thin
    atts_draw = pd.DataFrame({'att': atts.trace()[draw, :],})
    defs_draw = pd.DataFrame({'def': defs.trace()[draw, :],})
    home_draw = home.trace()[draw] #home.trace() is [80,] shape
    intercept_draw = intercept.trace()[draw]
    season = df.copy()
    season = pd.merge(season, atts_draw, left_on='HomeTeam', right_index=True)
    season = pd.merge(season, defs_draw, left_on='HomeTeam', right_index=True)
    season = season.rename(columns = {'att': 'att_home', 'def': 'def_home'})
    season = pd.merge(season, atts_draw, left_on='AwayTeam', right_index=True)
    season = pd.merge(season, defs_draw, left_on='AwayTeam', right_index=True)
    season = season.rename(columns = {'att': 'att_away', 'def': 'def_away'})
    season['home'] = home_draw
    season['intercept'] = intercept_draw
    season['home_theta'] = season.apply(lambda x: math.exp(x['intercept'] + 
                                                           x['home'] + 
                                                           x['att_home'] + 
                                                           x['def_away']), axis=1)
    season['away_theta'] = season.apply(lambda x: math.exp(x['intercept'] + 
                                                           x['att_away'] + 
                                                           x['def_home']), axis=1)
    season['home_goals'] = season.apply(lambda x: np.random.poisson(x['home_theta']), axis=1)
    season['away_goals'] = season.apply(lambda x: np.random.poisson(x['away_theta']), axis=1)
    season['home_outcome'] = season.apply(lambda x: 'win' if x['home_goals'] > x['away_goals'] else 
                                                    'loss' if x['home_goals'] < x['away_goals'] else 'draw', axis=1)
    season['away_outcome'] = season.apply(lambda x: 'win' if x['home_goals'] < x['away_goals'] else 
                                                    'loss' if x['home_goals'] > x['away_goals'] else 'draw', axis=1)
    season = season.join(pd.get_dummies(season.home_outcome, prefix='home'))
    season = season.join(pd.get_dummies(season.away_outcome, prefix='away'))
    return season


def create_season_table(season):
    """
    Using a season dataframe output by simulate_season(), create a summary dataframe with wins, losses, goals for, etc.
    
    """
    g = season.groupby('HomeTeam')    
    home = pd.DataFrame({'home_goals': g.home_goals.sum(),
                         'home_goals_against': g.away_goals.sum(),
                         'home_wins': g.home_win.sum(),
                         'home_draws': g.home_draw.sum(),
                         'home_losses': g.home_loss.sum()
                         })
    g = season.groupby('AwayTeam')    
    away = pd.DataFrame({'away_goals': g.away_goals.sum(),
                         'away_goals_against': g.home_goals.sum(),
                         'away_wins': g.away_win.sum(),
                         'away_draws': g.away_draw.sum(),
                         'away_losses': g.away_loss.sum()
                         })
    df = home.join(away)
    df['wins'] = df.home_wins + df.away_wins
    df['draws'] = df.home_draws + df.away_draws
    df['losses'] = df.home_losses + df.away_losses
    df['points'] = df.wins * 3 + df.draws
    df['gf'] = df.home_goals + df.away_goals
    df['ga'] = df.home_goals_against + df.away_goals_against
    df['gd'] = df.gf - df.ga
    df = pd.merge(teams, df, left_on='i', right_index=True)
    # df = df.sort_index(key='points', ascending=False)
    df = df.sort_values(by='points', ascending=False)
    df = df.reset_index()
    df['position'] = df.index + 1
    df['champion'] = (df.position == 1).astype(int)
    df['qualified_for_CL'] = (df.position < 5).astype(int)
    df['relegated'] = (df.position > 17).astype(int)
    return df  
    
def simulate_seasons(n=100):
    dfs = []
    for i in range(n):
        s = simulate_season()
        t = create_season_table(s)
        t['iteration'] = i
        dfs.append(t)
    return pd.concat(dfs, ignore_index=True)

num_simul=1000
print("".join(["Running simulation for ", str(num_simul), " seasons"]))
simuls = simulate_seasons(num_simul)

# team_name_test='Man United'
# ax = simuls.points[simuls['Team'] == team_name_test].hist(figsize=(7,5))
# median = simuls.points[simuls['Team'] == team_name_test].median()
# ax.set_title("".join([team_name_test, "2017-18 points, ", str(num_simul), " simulations"]))
# ax.plot([median, median], ax.get_ylim())
# plt.annotate('Median: %s' % median, xy=(median + 1, ax.get_ylim()[1]-10))


g = simuls.groupby('Team')
season_hdis = pd.DataFrame({'points_lower': g.points.quantile(.05),
                            'points_upper': g.points.quantile(.95),
                            'goals_for_lower': g.gf.quantile(.05),
                            'goals_for_median': g.gf.median(),
                            'goals_for_upper': g.gf.quantile(.95),
                            'goals_against_lower': g.ga.quantile(.05),
                            'goals_against_upper': g.ga.quantile(.95),
                            })
season_hdis = pd.merge(season_hdis, df_observed, left_index=True, right_on='Team')
column_order = ['Team', 'points_lower', 'Pts', 'points_upper', 
                'goals_for_lower', 'goals_scored', 'goals_for_median', 'goals_for_upper',
                'goals_against_lower', 'goals_lost', 'goals_against_upper',]
season_hdis = season_hdis[column_order]
season_hdis['relative_goals_upper'] = season_hdis.goals_for_upper - season_hdis.goals_for_median
season_hdis['relative_goals_lower'] = season_hdis.goals_for_median - season_hdis.goals_for_lower
# season_hdis = season_hdis.sort_index(by='goals_scored')
season_hdis = season_hdis.sort_values(by='goals_scored')
season_hdis = season_hdis.reset_index()
season_hdis['x'] = season_hdis.index + .5
season_hdis

fig, axs = plt.subplots(figsize=(10,6))
axs.scatter(season_hdis.x, season_hdis.goals_scored, c=sns.palettes.color_palette()[4], zorder = 10, label='Actual Goals For')
axs.errorbar(season_hdis.x, season_hdis.goals_for_median, 
             yerr=(season_hdis[['relative_goals_lower', 'relative_goals_upper']].values).T, 
             fmt='s', c=sns.palettes.color_palette()[5], label='Simulations')
axs.set_title('Actual Goals For, and 90% Interval from Simulations, by Team')
axs.set_xlabel('Team')
axs.set_ylabel('Goals Scored')
axs.set_xlim(0, 20)
axs.legend()
_= axs.set_xticks(season_hdis.index + .5)
_= axs.set_xticklabels(season_hdis['Team'].values, rotation=45)
