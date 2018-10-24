# This module contains the boundary value solver code, which we can reference in the Jupyter Notebook.
import numpy as np
from scipy.optimize import root
from matplotlib import pyplot as plt

coeffs = []


def fun(cf, a, b, dim, deg, ode, bc):
    cf = cf.reshape((dim,deg+1))
    cf = np.array(cf)
    
    # nodes (Chebyshev extrema)
    x = np.cos((np.pi * np.arange(deg + 1)) / deg)
    # print(x)
    # scale to (a, b)
    x = (x * (b - a) + b + a) / 2
    
    # polynomial evaluated at nodes
    cheb_poly = [np.polynomial.chebyshev.Chebyshev(coeffs, (a, b)) for coeffs in cf]
    
    p = np.array([cheb_poly[i](x) for i in range(dim)])

    # polynomial at the end points
    ya = p[:, -1]
    yb = p[:, 0]

    # coefficients of the derivative of the polynomial
    cheb_poly_der = [cheb_poly[i].deriv() for i in range(dim)]

    # derivative of polynomial evaluated at the nodes
    p_der = np.array([cheb_poly_der[i](x) for i in range(dim)])

    ya_prime = p_der[:, 0]
    yb_prime = p_der[:, -1]

    # plot it
    dom = np.linspace(a, b, 1000)
    coeffs.append(cf)

    # output for fsolve
    stuff = [
        *np.ravel(bc([ya, ya_prime], [yb, yb_prime])),  # boundary conditions
        *np.ravel((p_der - ode(x, p)))  # ODE conditions
    ]
    
    print(stuff)
    return stuff


def our_own_bvp_solve():
    # constants
    gamma = 5 / 3
    v_plus = 3 / 5
    v_star = gamma / (gamma + 2)
    e_minus = (gamma + 2) * (v_plus - v_star) / 2 / gamma / (gamma + 1)

    # ODE
    f = lambda x, y: np.array([y[0] * (y[0] - 1) + gamma * (y[1] - y[0] * e_minus),
                               y[0] * (-(y[0] - 1) ** 2 / 2 + y[1] - e_minus + gamma * e_minus * (y[0] - 1))])

    a = -20
    b = 20
    n = 40
    dim = 2

    bc = lambda ya, yb: [ya[0] - np.array([3 / 50, 1]), yb[0] - np.array([9 / 50, 3 / 5]), yb[1] - np.array([-.6, 1])]

    u0 = np.random.random((n + 1, dim))

    solution = root(lambda u: fun(u, a, b, dim, n, f, bc), u0, method='lm')

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