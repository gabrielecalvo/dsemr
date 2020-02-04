"""
Equivalent to R Segmented package.
code source: https://datascience.stackexchange.com/questions/8457/python-library-for-segmented-regression-a-k-a-piecewise-regression
"""

import numpy as np
from numpy.linalg import lstsq

ramp = lambda u: np.maximum(u, 0)
step = lambda u: (u > 0).astype(float)


def segmented_linear_regression(x, y, breakpoints, n_iteration_max=10):
    breakpoints = np.sort(np.array(breakpoints))
    dt = np.min(np.diff(x))
    ones = np.ones_like(x)

    for i in range(n_iteration_max):
        # Linear regression:  solve A*p = Y
        Rk = [ramp(x - xk) for xk in breakpoints]
        Sk = [step(x - xk) for xk in breakpoints]
        A = np.array([ones, x] + Rk + Sk)
        p = lstsq(A.transpose(), y, rcond=None)[0]

        # Parameters identification:
        a, b = p[0:2]
        ck = p[2:2 + len(breakpoints)]
        dk = p[2 + len(breakpoints):]

        # Estimation of the next break-points:
        new_breakpoints = breakpoints - dk / ck

        # Stop condition
        if np.max(np.abs(new_breakpoints - breakpoints)) < dt / 5:
            break

        breakpoints = new_breakpoints
    else:
        print('maximum iteration reached')

    # Compute the final segmented fit:
    x_solution = np.insert(np.append(breakpoints, max(x)), 0, min(x))
    ones = np.ones_like(x_solution)
    Rk = [c * ramp(x_solution - x0) for x0, c in zip(breakpoints, ck)]

    y_solution = a * ones + b * x_solution + np.sum(Rk, axis=0)
    return x_solution, y_solution
