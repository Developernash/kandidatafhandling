#system imports
from pathlib import Path
from typing import Tuple
from typing import Dict
import os
import sys

#fundamental imports
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import math

#from first_stage estimation
from first_stage.wage import wage_estimation
from first_stage.load_params import load_params_txt
from first_stage.mortality import prob_survival

#Project imports
from functions.utility import old_utility_functions
from functions.utility import new_utility_functions
from functions.final_period_utility import final_period_utility
from functions.budget import budget_dcegm_initial
from functions.state_space_functions import create_state_space_function_dict
from functions.compute_moments import compute_simulation_moments
from functions.compute_moments import compute_simulation_moments_with_ci
# from functions.estimation import crit_func_scipy
# from functions.estimation import theta_to_params
# from functions.estimation import compute_empirical_share_variances

from plots.plots import plot_empirical_vs_simulated_with_ci
from plots.plots import plot_empirical_vs_simulated_with_ci_grid

#DC-EGM imports
import dcegm
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.interfaces.model_class import setup_model

#Statistical imports
import statsmodels.api as sm
from scipy.optimize import minimize

