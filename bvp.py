# This module contains the boundary value solver code, which we call in the Jupyter Notebook.
import numpy as np
from scipy.optimize import root
from matplotlib import pyplot as plt


def fun(cf, a, b, dim, deg, ode, bc):
    cf = cf.reshape((dim, deg+1))
    cf = np.array(cf)
    
    # nodes (Chebyshev extrema)
    x = np.array(list(reversed(np.cos((np.pi * np.arange(deg + 1)) / deg))))
    # scale to (a, b)
    x = (x * (b - a) + b + a) / 2
    
    # polynomial evaluated at nodes
    cheb_poly = [np.polynomial.chebyshev.Chebyshev(coeffs, (a, b)) for coeffs in cf]
    
    p = np.array([cheb_poly[i](x) for i in range(dim)])

    # polynomial at the end points
    ya = p[:, 0]
    yb = p[:, -1]

    # coefficients of the derivative of the polynomial
    cheb_poly_der = [cheb_poly[i].deriv() for i in range(dim)]

    # derivative of polynomial evaluated at the nodes
    p_der = np.array([cheb_poly_der[i](x) for i in range(dim)])

    ya_prime = p_der[:, 0]
    yb_prime = p_der[:, -1]

    # output for fsolve
    return [
        *np.ravel(bc([ya, ya_prime], [yb, yb_prime])),  # boundary conditions
        *np.ravel((p_der - ode(x,p)))  # ODE conditions
    ]


def our_own_bvp_solve(f, a, b, n, y0, dim, bc, tol=1e-10):
    """Solves a boundary value problem using Chebyshev colocation. Returns a list of functions that form the solution to
    the problem."""

    cf0 = np.polynomial.chebyshev.chebfit(np.linspace(a, b, len(y0)), y0, n)

    solution = root(lambda u: fun(u, a, b, dim, n, f, bc), cf0, method='lm', tol=tol)
    if not solution.success:
        print('root finding failed')

    cf = solution.x
    cf = cf.reshape((dim, cf.size // dim))

    return [np.polynomial.chebyshev.Chebyshev(cf[i], (a, b)) for i in range(dim)]
