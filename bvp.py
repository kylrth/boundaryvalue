# This module will contain the boundary value solver code, which we can reference in the Jupyter Notebook.
import numpy as np


def fun(cf, a, b, deg, ode, bc):

    # interpolate
    # thing = np.polynomial.chebyshev.Chebyshev((1))
    # thing = thing.interpolate(fun, deg, (a, b))

    # nodes (Chebyshev extrema)
    x = np.cos((np.pi * np.arange(2 * n)) / n)

    # polynomial evaluated at nodes
    p = np.polynomial.chebyshev.chebval(x, cf)

    # polynomial at the end points
    ya = p[0]
    yb = p[-1]

    # coefficients of the derivative of the polynomial
    # (Source: Chebyshev collocation methods for ordinary differential equations, K. Wright)
    


if __name__ == '__main__':
    print('This is a module for the BVP solver. To run, call one of the functions.')
