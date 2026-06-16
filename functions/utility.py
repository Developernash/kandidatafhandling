from project_imports import *

def flow_util_phi(consumption, choice, params, period, model_specs, lagged_choice):
    """
    Flow utility with asymmetric switching costs.

    Switching-cost structure:
        phi_entry  : cost of moving from non-work to work
        phi_exit   : cost of moving from work to non-work
        phi_switch : cost of switching between positive hours levels
    """

    # ------------------------------------------------------------
    # Make choice variables JAX-safe integer types
    # ------------------------------------------------------------
    choice = choice.astype(jnp.int32)
    lagged_choice = lagged_choice.astype(jnp.int32)

    # ------------------------------------------------------------
    # Utility from consumption
    # ------------------------------------------------------------
    rho = params["rho"]

    cons_util = ((consumption ** (1 - rho)) - 1) / (1 - rho)

    ##############################################
    ########## Disutility of working #############
    ##############################################

    # Disutility parameters
    age = (model_specs["start_age"] + period).astype(float)
    gamma = params["gamma"][choice - 1]  # remove the first element

    working = choice > 0  # 1 if choice is working, 0 if not
    # -------  Zero disutility from working if unemployed

    # Age component of disutility
    age_linear = jnp.where(age > 50, params["kappa1"] * (age - 50), 0.0)
    age_quadratic = jnp.where(age > 50, params["kappa2"] * (age - 50) ** 2, 0.0)

    # estimate total disutility
    disutil_0 = working * (1.0 + age_linear + age_quadratic) * gamma

    # No disutility from working, if you dont work.
    disutil = jnp.where(working, disutil_0, 0.0)  # if working == 0, disutility 

    # ------------------------------------------------------------
    # Asymmetric switching costs
    # ------------------------------------------------------------

    was_not_working = lagged_choice == 0
    was_working = lagged_choice > 0

    is_not_working = choice == 0
    is_working = choice > 0

    # 0 -> positive hours
    entry = was_not_working & is_working

    # positive hours -> 0
    exit = was_working & is_not_working

    # positive hours -> different positive hours
    positive_hours_switch = was_working & is_working & (choice != lagged_choice)

    trans_cost = (
        params["phi_entry"] * entry.astype(float)
        + params["phi_exit"] * exit.astype(float)
        + params["phi_switch"] * positive_hours_switch.astype(float)
    )

    # ------------------------------------------------------------
    # Total flow utility
    # ------------------------------------------------------------
    u = cons_util - disutil - trans_cost

    return u


def flow_util(consumption, choice, params, period, model_specs, lagged_choice):
    # Utility parameter
    rho = params["rho"]

    # Disutility parameters
    age = (model_specs["start_age"] + period).astype(float)
    gamma = params["gamma"][choice - 1]  # remove the first element

    ##############################################
    ########## Disutility of working #############
    ##############################################
    working = choice > 0  # 1 if choice is working, 0 if not
    # -------  Zero disutility from working if unemployed

    # Age component of disutility
    # age_linear = jnp.where(age > 50, params["kappa1"] * (age - 50), 0.0)
    # age_quadratic = jnp.where(age > 50, params["kappa2"] * (age - 50) ** 2, 0.0)

    age_linear = jnp.where(age >= 55, params["kappa1"] * (age - 55), 0.0)
    age_quadratic = jnp.where(age >= 55, params["kappa2"] * (age - 55) ** 2, 0.0)   
    # + age_linear

    # estimate total disutility
    disutil_0 = working * (1.0 + age_linear + age_quadratic) * gamma

    # No disutility from working, if you dont work.
    disutil = jnp.where(working, disutil_0, 0.0)  # if working == 0, disutility = 0

    ##############################################
    ############# Transaction costs  #############
    ##############################################

    # transition cost when going from working to unemployed, and vice versa
    trans_cost = jnp.where(
        choice != lagged_choice, params["phi"], 0.0
    )  # from changing choice

    # Utility for agents that are alive.
    u = (
        ((consumption ** (1 - rho)) - 1) / (1 - rho) - disutil - trans_cost
    )  # working*gamma*hours*(1+(kappa1*age)*age_1+(kappa2*age_2)**2*age_2) #jax.lax.select(working, gamma, 0) - if a NaN included

    # Utility for agents that are dead. no utility from consumption, only utility from bequest.
    # u_dead = jnp.where(first_time_dead, bequest, 0.0) # if first time dead, utility is -inf

    # u = jnp.where(survival == 1, u_alive, u_dead) # if survival == 1, utility is alive utility, else dead utility

    return u


def marginal_utility(consumption, params):
    rho = params["rho"]
    u_prime = consumption ** (-rho)
    return u_prime


def inverse_marginal_utility(marginal_utility, params):
    rho = params["rho"]
    return marginal_utility ** (-1 / rho)


old_utility_functions = {
    "utility": flow_util,
    "inverse_marginal_utility": inverse_marginal_utility,
    "marginal_utility": marginal_utility,
}

new_utility_functions = {
    "utility": flow_util_phi,
    "inverse_marginal_utility": inverse_marginal_utility,
    "marginal_utility": marginal_utility,
}