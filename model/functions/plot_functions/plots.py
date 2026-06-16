import matplotlib.pyplot as plt
import project_paths as pp

    
def plot_empirical_vs_simulated_with_ci(
    edu,
    moments_sim,
    out_subfolder=None,
    figsize=(4, 4),
    dpi=100,
):
    
    var_labels = {
        "avg_wealth": "Average Wealth",
        "hours_0": "Unemployed",
        "hours_1": "Below 10 hours",
        "hours_2": "10-20 hours",
        "hours_3": "20-30 hours",
        "hours_4": "Above 30 hours",
        "avg_hours": "Average Hours",
        "prob_work": "Employment Rate",
        "avg_wage": "Average Wage",
        "avg_experience": "Average Experience",
        "work_work": "Work to Work Transition Rate",
        "nowork_nowork": "No Work to No Work Transition Rate",
    }

    var_scales = {
        "avg_wealth": 100_000,
        "avg_wage": 100_000,
    }

    ylims={
        "avg_wealth": (0, 3_000_000),
        "avg_hours": (0, 2_000),
        "prob_work": (0, 1),
        "hours_0": (0, 1),
        "hours_1": (0, 0.2),
        "hours_2": (0, 0.2),
        "hours_3": (0, 0.3),
        "hours_4": (0, 1),
        "avg_wage": (0, 500),
    }


    if var_labels is None:
        var_labels = {}
    if var_scales is None:
        var_scales = {}
    if ylims is None:
        ylims = {}

    plot_vars = [
        v for v in edu.columns
        if v != "age" and v in moments_sim.columns
    ]

    if out_subfolder is None:
        out_dir = pp.SIM_PLOTS_DIR
    else:
        out_dir = pp.SIM_PLOTS_DIR / out_subfolder

    out_dir.mkdir(parents=True, exist_ok=True)

    for var in plot_vars:
        pretty = var_labels.get(var, var)
        scale = var_scales.get(var, 1.0)
        ymin, ymax = ylims.get(var, (None, None))

        x_emp = edu["age"]
        y_emp = edu[var] * scale

        x_sim = moments_sim["age"]
        y_sim = moments_sim[var] * scale

        low_col = f"{var}_lower"
        high_col = f"{var}_upper"
        has_ci = low_col in moments_sim.columns and high_col in moments_sim.columns

        if has_ci:
            y_low = moments_sim[low_col] * scale
            y_high = moments_sim[high_col] * scale

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x_emp, y_emp, "-", label="Empirical")
        ax.plot(x_sim, y_sim, "--", label="Simulated")

        if has_ci:
            ax.fill_between(
                x_sim,
                y_low,
                y_high,
                alpha=0.2,
                label="95% CI",
            )

        ax.set_title(pretty)
        ax.set_xlabel("Age")
        ax.grid(True)
        ax.set_ylim(ymin, ymax)
        ax.legend()

        plt.tight_layout()

        out_dir = pp.SIM_PLOTS_DIR

        if out_subfolder is not None:
            out_dir = out_dir / out_subfolder

        out_dir.mkdir(parents=True, exist_ok=True)

        save_path = out_dir / f"{var}_over_age.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        print(f"Saved {save_path}")

        # plt.show()

def plot_empirical_vs_simulated_with_ci_grid(
    edu,
    moments_sim,
    out_subfolder=None,
    ncols=3,
    figsize_per_plot=(4, 4),
    dpi=100,
    filename="empirical_vs_simulated_grid.png",
):
    import math
    
    var_labels = {
        "avg_wealth": "Average Wealth",
        "avg_labor_income": "Average Income",
        "hours_0": "Unemployed",
        "hours_1": "Below 10 hours",
        "hours_2": "10-20 hours",
        "hours_3": "20-30 hours",
        "hours_4": "Above 30 hours",
        "avg_hours": "Average Hours",
        "prob_work": "Employment Rate",
        "avg_wage": "Average Wage",
        "avg_experience": "Average Experience",
        "work_work": "Work to Work Transition Rate",
        "nowork_nowork": "No Work to No Work Transition Rate",
    }

    var_scales = {
        "avg_wealth": 100_000,
        "avg_wage": 100_000,
    }

    ylims={
        "avg_wealth": (0, 3_000_000),
        "avg_hours": (0, 2_000),
        "prob_work": (0, 1),
        "hours_0": (0, 1),
        "hours_1": (0, 0.2),
        "hours_2": (0, 0.2),
        "hours_3": (0, 0.3),
        "hours_4": (0, 1),
        "avg_wage": (0, 500),
    }

    if var_labels is None:
        var_labels = {}
    if var_scales is None:
        var_scales = {}
    if ylims is None:
        ylims = {}

    plot_vars = [
        v for v in edu.columns
        if v != "age" and v in moments_sim.columns
    ]

    n_plots = len(plot_vars)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax, var in zip(axes_flat, plot_vars):
        pretty = var_labels.get(var, var)
        scale = var_scales.get(var, 1.0)
        ymin, ymax = ylims.get(var, (None, None))

        x_emp = edu["age"]
        y_emp = edu[var] * scale

        x_sim = moments_sim["age"]
        y_sim = moments_sim[var] * scale

        low_col = f"{var}_lower"
        high_col = f"{var}_upper"
        has_ci = low_col in moments_sim.columns and high_col in moments_sim.columns

        ax.plot(x_emp, y_emp, "-", label="Empirical")
        ax.plot(x_sim, y_sim, "--", label="Simulated")

        if has_ci:
            y_low = moments_sim[low_col] * scale
            y_high = moments_sim[high_col] * scale

            ax.fill_between(
                x_sim,
                y_low,
                y_high,
                alpha=0.2,
                label="95% CI",
            )

        ax.set_title(pretty)
        ax.set_xlabel("Age")
        ax.grid(True)
        ax.set_ylim(ymin, ymax)

    # Remove empty subplots if number of variables is not divisible by ncols
    for ax in axes_flat[n_plots:]:
        ax.remove()

    # One shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()

    if out_subfolder is None:
        out_dir = pp.SIM_PLOTS_DIR
    else:
        out_dir = pp.SIM_PLOTS_DIR / out_subfolder

    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved {save_path}")

    plt.show()

    return fig, axes