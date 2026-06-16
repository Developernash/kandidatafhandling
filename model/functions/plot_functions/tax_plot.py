from project_imports import *
import project_paths as pp

# Tax parameters
th1 = 54000      # First threshold in DKK
th2 = 641_200    # Second threshold in DKK
th3 = 777_900    # Third threshold in DKK
th4 = 2_592_700  # Fourth threshold in DKK

r2  = 0.362      # Tax rate between th1 and th2
r3  = 0.587      # Tax rate above th2
r4  = 0.662      # Tax rate above th3
r5  = 0.712      # Tax rate above th4

# Create a grid of incomes from 0 to 3,000,000 DKK
income = np.linspace(0, 3_000_000, 5000)

# Compute the portions in each bracket
inc2 = np.minimum(np.maximum(income - th1, 0.0), th2 - th1)
inc3 = np.minimum(np.maximum(income - th2, 0.0), th3 - th2)
inc4 = np.minimum(np.maximum(income - th3, 0.0), th4 - th3)
inc5 = np.minimum(np.maximum(income - th4, 0.0), 3_000_000 - th4)

# Compute total tax liability
tax = r2 * inc2 + r3 * inc3 + r4 * inc4 + r5 * inc5

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(income, tax, linewidth=2, label="Tax Liability")
ax.axvline(th1, color='gray', linestyle='--', label='Base Threshold 54.000 DKK)')
ax.axvline(th2, color='gray', linestyle=':', label='Mellemskat Threshold 641.200 DKK)')
ax.axvline(th3, color='gray', linestyle='-.', label='Topskat Threshold 777.900 DKK)')
ax.axvline(th4, color='gray', linestyle='-', label='Toptopskat Threshold 2.592.700 DKK)')
ax.set_xlabel("Labor Income (DKK)")
ax.set_ylabel("Tax Liability (DKK)")
ax.set_title("Piecewise Linear Tax Function")
ax.grid(True)
ax.legend()

# Disable scientific notation on both axes
fmt = ScalarFormatter(useOffset=False)
fmt.set_scientific(False)
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)
plt.savefig(pp.PLOTS_DIR + "/tax_function_plot.png", dpi=300)

plt.tight_layout()
plt.show()