"""
expect to run on python >3.7
"""
import os
import math
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc # I know folks are switching to "as pm" but I'm just not there yet

DATA_DIR = os.path.join(os.getcwd(), 'data')
data_file = DATA_DIR + 'final_18-19season.csv'

df = pd.read_csv(data_file, sep='\t', index_col=0,)
df.head()

