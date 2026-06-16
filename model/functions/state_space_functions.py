from project_imports import *

def  next_period_experience(period, lagged_choice, experience, model_specs):
    """Calculate next period's experience based on current period's experience and last period labor choice."""
    experience = experience.astype(float)
    period = period.astype(float)

    # grab hours and max_hours as
    hours = model_specs["hours"][lagged_choice]
    max_hours = jnp.array(model_specs["max_hours"])
    init_exp = jnp.array(model_specs["max_init_experience"])

    # t+init_experience
    max_experience_period = period + init_exp

    # E_{t+1} = ((t * E_t) + h_t / max_hours) / (t+1)
    next_exp = (
        (max_experience_period - 1.0) * experience + (hours / max_hours)
    ) / max_experience_period

    return next_exp

def get_state_specific_feasible_choice_set(
    lagged_choice: int,
    model_specs: Dict,
    period: int,
) -> jnp.ndarray:
    return jnp.asarray(model_specs["choices"])

def next_period_deterministic_state(period, choice, model_specs):
    """Update discrete endogenous states.

    Current choice becomes next period's lagged choice.
    """
    return {
        "period": period + 1,
        "lagged_choice": choice,
    }

def sparsity_condition(period, lagged_choice, survival, model_specs):
    """Determine the sparsity condition for the current state."""
    final_period = model_specs["n_periods"] - 1

    if survival == 0:
        return {
            "period": final_period,
            "lagged_choice": lagged_choice,
            "survival": survival,
        }
    else:
        return {
            "period": period,
            "lagged_choice": lagged_choice,
            "survival": survival,
        }

def create_state_space_function_dict():
    """Create dictionary with state space functions.

    Returns:
        state_space_functions (dict): Dictionary with state space functions.

    """
    return {
        "next_period_deterministic_state": next_period_deterministic_state,
        "next_period_experience": next_period_experience,
        "sparsity_condition": sparsity_condition,
        "state_specific_choice_set": get_state_specific_feasible_choice_set,
    }



#######################################################################################
######################### Alternative functions #######################################
#######################################################################################

# def state_specific_choice_set(period, lagged_choice, model_specs):
#     """Determine the feasible choice set for the current state."""

#     age = (model_specs["start_age"] + period).astype(float)

#     # Retirement is absorbing
#     if (lagged_choice == 0) and (age > model_specs["retirement_age"]):
#         return [0]
#     # If period equal or larger max ret age you have to choose retirement
#     elif period >= model_specs["max_ret_period"]:
#         return [0]
#     # If above minimum retirement period, retirement is possible
#     else:
#         return model_specs["choices"]

# # state_choice_set where retirement is absorbing
# def get_state_specific_feasible_choice_set(
#     lagged_choice: int,
#     model_specs: Dict,
#     period: int,
# ) -> np.ndarray:

#     age = model_specs["start_age"] + period

#     # Once the agent choses retirement, she can only choose retirement thereafter.
#     # Hence, retirement is an absorbing state.
#     if lagged_choice == 0 and (age >= model_specs["retirement_age"]):
#         feasible_choice_set = jnp.array([0])
#     else:
#         feasible_choice_set = model_specs["choices"]

#     return feasible_choice_set


# Alternative: Force retirement after max retirement age, but allow all choices before that.
# def get_state_specific_feasible_choice_set(
#     lagged_choice: int,
#     model_specs: Dict,
#     period: int,
# ) -> np.ndarray:
#     # Force everyone into non-work after max retirement age, e.g. age 75
#     if period >= model_specs["max_ret_period"]:
#         return np.array([0], dtype=np.int32)

#     # Otherwise allow all choices
#     return np.asarray(model_specs["choices"], dtype=np.int32)
