from project_imports import *

def tax_liability(labor_income):
    """
    Piecewise tax liability with
      – 0% på indkomst [0, th1]
      – rate2 på indkomst (th1, th2]
      – rate3 på indkomst > th2
    th1, th2, rate2, rate3 kan hentes fra model_specs eller params.
    """
    # Tax parameters
    th1 = 54000      # First threshold in DKK
    th2 = 641_200    # Second threshold in DKK
    th3 = 777_900    # Third threshold in DKK
    th4 = 2_592_700  # Fourth threshold in DKK

    r1 = 0.0        # Tax rate between 0 and th1
    r2 = 0.362      # Tax rate between th1 and th2
    r3 = 0.587      # Tax rate above th2
    r4 = 0.662      # Tax rate above th3
    r5 = 0.712      # Tax rate above th4

    # Create a grid of incomes from 0 to 3,000,000 DKK
    labour_income = jnp.linspace(0, 3_000_000, 5000)

    # Compute the portions in each bracket
    inc1 = jnp.minimum(labour_income, th1)
    inc2 = jnp.minimum(jnp.maximum(labour_income - th1, 0.0), th2 - th1)
    inc3 = jnp.minimum(jnp.maximum(labour_income - th2, 0.0), th3 - th2)
    inc4 = jnp.minimum(jnp.maximum(labour_income - th3, 0.0), th4 - th3)
    inc5 = jnp.minimum(jnp.maximum(labour_income - th4, 0.0), 3_000_000 - th4)

    # Compute total tax liability
    return  r1 * inc1 + r2 * inc2 + r3 * inc3 + r4 * inc4 + r5 * inc5