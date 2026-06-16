import numpy as np
import pandas as pd
import statsmodels.api as sm
import project_paths as pp

##################################################################
########## 3 NEW WLS VS OLS ESTIMATION ############################
##################################################################


def wage_estimation(estimator="ols", save=True):
    estimator = estimator.lower()

    if estimator not in ["ols", "wls"]:
        raise ValueError("estimator must be either 'ols' or 'wls'.")

    files = {
        1: pp.MOMENTS_DIR / "moments_udd1.txt",
        2: pp.MOMENTS_DIR / "moments_udd2.txt",
        3: pp.MOMENTS_DIR / "moments_udd3.txt",
    }

    dfs = {
        udd: pd.read_csv(path)
        for udd, path in files.items()
    }

    models = {}
    betas = {}

    for udd, df in dfs.items():
        df = df.copy()

        if "ALDER" in df.columns:
            df.rename(columns={"ALDER": "age"}, inplace=True)

        df["log_wage"] = np.log(df["avg_wage"])

        X = sm.add_constant(
            np.column_stack((df["age"], df["age"] ** 2))
        )

        y = df["log_wage"]

        if estimator == "ols":
            model = sm.OLS(y, X, missing="drop").fit()

        elif estimator == "wls":
            if "_FREQ_" not in df.columns:
                raise ValueError(f"_FREQ_ missing for education group {udd}")

            weights = df["_FREQ_"]
            model = sm.WLS(y, X, weights=weights, missing="drop").fit()

        beta0, beta1, beta2 = model.params

        models[udd] = model
        betas[udd] = {
            "beta0": beta0,
            "beta1": beta1,
            "beta2": beta2,
        }

        df["predicted"] = np.exp(
            beta0 + beta1 * df["age"] + beta2 * df["age"] ** 2
        )

        dfs[udd] = df

        print(
            f"{estimator.upper()} UDD {udd}: "
            f"β0={beta0:.4f}, β1={beta1:.4f}, β2={beta2:.4f}"
        )

        if save:
            out_path = (
                pp.FIRST_STAGE_RESULTS_DIR
                / f"wage_params_udd{udd}_{estimator}.txt"
            )

            np.savetxt(out_path, [beta0, beta1, beta2])

    return betas, models, dfs