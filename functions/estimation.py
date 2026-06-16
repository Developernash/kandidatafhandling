import numpy as np
import jax.numpy as jnp


def theta_to_params(theta_array, base_params, beta0, beta1, beta2):
    p = base_params.copy()

    p["interest_rate"] = 0.01
    p["beta0"] = beta0
    p["beta1"] = beta1
    p["beta2"] = beta2

    p["income_shock_mean"] = theta_array[0]
    p["income_shock_std"] = theta_array[1]
    p["taste_shock_scale"] = theta_array[2]
    p["discount_factor"] = theta_array[3]
    p["rho"] = theta_array[4]

    p["gamma"] = jnp.array([
        theta_array[5],
        theta_array[6],
        theta_array[7],
        theta_array[8],
    ])

    p["kappa1"] = theta_array[9]
    p["kappa2"] = theta_array[10]

    p["phi_entry"] = theta_array[11]
    p["phi_exit"] = theta_array[12]
    p["phi_switch"] = theta_array[13]

    p["b_scale"] = theta_array[14]
    p["xi"] = theta_array[15]

    p.pop("phi", None)

    return p


def make_crit_func_scipy(
    model,
    base_params,
    beta0,
    beta1,
    beta2,
    states_initial,
    seed,
    compute_simulation_moments,
    start_age,
    hours_map,
    df_edu,
    keep_cols,
    verbose=False,
):
    """
    Creates the MSM criterion function used by scipy.optimize.minimize.

    All notebook-specific objects are passed in once here, so the function
    returned below only depends on theta_array.
    """

    def crit_func_scipy(theta_array):
        p = theta_to_params(
            theta_array=theta_array,
            base_params=base_params,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
        )

        if verbose:
            print("Parameter dictionary used:")
            for key, value in p.items():
                print(f"{key}: {value}")

        model.validate_exogenous(p)

        model_solved = model.solve(p)

        sim = model_solved.simulate(
            states_initial=states_initial,
            seed=seed,
        )

        sim_moments = compute_simulation_moments(
            sim,
            start_age,
            hours_map,
        )

        empirical_moms = df_edu.copy()

        if "pens" in sim_moments.columns:
            sim_moments = sim_moments.drop(columns=["pens"])

        if "pens" in empirical_moms.columns:
            empirical_moms = empirical_moms.drop(columns=["pens"])

        sim_moments = sim_moments[keep_cols]
        empirical_moms = empirical_moms[keep_cols]

        sim_vals = sim_moments.to_numpy()
        emp_vals = empirical_moms.to_numpy()

        if verbose:
            print("Simulated moments shape:", sim_vals.shape)
            print("Empirical moments shape:", emp_vals.shape)

        diff = sim_vals - emp_vals

        emp_var = np.nanvar(emp_vals, axis=0, ddof=1)

        epsilon = 1e-6
        weights = 1.0 / (emp_var + epsilon)

        crit_val = 0.0
        for i in range(diff.shape[1]):
            crit_val += weights[i] * np.nansum(diff[:, i] ** 2)

        if verbose:
            print("Crit value:", crit_val)
            print("")

        return float(crit_val)

    return crit_func_scipy

# Helper function for determining variance in empirical moments
def compute_empirical_share_variances(df, share_cols, freq_col="_FREQ_"):
    df_var = df[["age", freq_col] + share_cols].copy()

    for col in share_cols:
        p = df_var[col]
        n = df_var[freq_col]
        df_var[col] = p * (1 - p) / n

    return df_var