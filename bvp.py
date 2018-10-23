# This module will contain the boundary value solver code, which we can reference in the Jupyter Notebook.
import numpy as np
from scipy.optimize import fsolve


def fun(cf, a, b, deg, ode, bc):

    # interpolate
    # thing = np.polynomial.chebyshev.Chebyshev((1))
    # thing = thing.interpolate(fun, deg, (a, b))

    # nodes (Chebyshev extrema)
    x = np.cos((np.pi * np.arange(2 * deg)) / deg)

    # polynomial evaluated at nodes
    cheb_poly = np.polynomial.chebyshev.Chebyshev(cf, (a, b))
    p = cheb_poly(x)

    # polynomial at the end points
    ya = p[0]
    yb = p[-1]

    # coefficients of the derivative of the polynomial
    # (Source: Chebyshev collocation methods for ordinary differential equations, K. Wright)
    cheb_poly_der = cheb_poly.deriv()

    # derivative of polynomial evaluated at the nodes
    p_der = cheb_poly_der(x)

    ya_prime = p_der[0]
    yb_prime = p_der[-1]

    # output for fsolve
    return [
        bc([ya, ya_prime], [yb, yb_prime]),  # boundary conditions
        p_der[1:-1] - ode(x[1:-1], p[1:-1])  # ODE conditions
    ]


def our_own_bvp_solve():
    f = lambda y: y

    a = 0
    b = 1
    n = 2

    bc = lambda ya, yb: ya[1] - 1

    u0 = np.random.rand(n + 1)

    cf = fsolve(lambda u: fun(u, a, b, n, f, bc), u0)

    




if __name__ == '__main__':
    print('This is a module for the BVP solver. To run, call one of the functions.')
