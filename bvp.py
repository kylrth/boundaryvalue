# This module will contain the boundary value solver code, which we can reference in the Jupyter Notebook.
import numpy as np
from scipy.optimize import root
from matplotlib import pyplot as plt

coeffs = []


def fun(cf, a, b, deg, ode, bc):

    # nodes (Chebyshev extrema)
    x = np.cos((np.pi * np.arange(deg + 1)) / deg)
    # print(x)
    # scale to (a, b)
    x = (x * (b - a) + b + a) / 2

    # polynomial evaluated at nodes
    cheb_poly = np.polynomial.chebyshev.Chebyshev(cf, (a, b))
    p = cheb_poly(x)

    # polynomial at the end points
    ya = p[-1]
    yb = p[0]

    # coefficients of the derivative of the polynomial
    cheb_poly_der = cheb_poly.deriv()

    # derivative of polynomial evaluated at the nodes
    p_der = cheb_poly_der(x)

    ya_prime = p_der[0]
    yb_prime = p_der[-1]

    # plot it
    dom = np.linspace(a, b, 1000)
    coeffs.append(cf)

    # output for fsolve
    return [
        *bc([ya, ya_prime], [yb, yb_prime]),  # boundary conditions
        *(p_der - ode(x, p))  # ODE conditions
    ]


def our_own_bvp_solve():
    f = lambda x, y: y

    a = -1
    b = 1
    n = 10

    bc = lambda ya, yb: [ya[0] - np.exp(-1)]

    u0 = np.random.rand(n + 1)

    solution = root(lambda u: fun(u, a, b, n, f, bc), u0, method='lm')

    fudge_factor = 1

    cf = solution.x
    print(solution.success)  # flesh this out

    # plotting
    dom = np.linspace(a, b, 1000)
    plt.plot(dom, np.exp(dom), label='$e^x$')
    plt.plot(dom, np.polynomial.chebyshev.chebval(dom, cf) * fudge_factor, label='estimate')
    # for i, cf_temp in enumerate(coeffs):
    #     plt.plot(dom, np.polynomial.chebyshev.chebval(dom, cf_temp), label=str(i))
    plt.legend()
    plt.show()




if __name__ == '__main__':
    print('This is a module for the BVP solver. To run, call one of the functions.')
