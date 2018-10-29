# This module contains the boundary value solver code, which we call in the Jupyter Notebook.
import numpy as np
from scipy.optimize import root
from matplotlib import pyplot as plt


def fun(cf, a, b, dim, deg, ode, bc):
    cf = cf.reshape((dim,deg+1))
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

    # ya_prime = p_der[:, 0]
    # yb_prime = p_der[:, -1]

    # output for fsolve
    stuff = [
        *np.ravel(bc(ya, yb)),  # boundary conditions
        *np.ravel((p_der - ode(x,p)))  # ODE conditions
    ]
    print(stuff)
    return stuff


def our_own_bvp_solve(f, a, b, n, u0, dim, bc, tol=1e-10):
    """Solves a boundary value problem using Chebyshev colocation. Returns a list of functions that form the solution to
    the problem."""

    y0 = np.polynomial.chebyshev.chebfit(np.linspace(a, b, len(u0)), u0, 20)
    print(y0)
    plt.plot()
    input()

    solution = root(lambda u: fun(u, a, b, dim, n, f, bc), y0, method='lm', tol=tol)
    if not solution.success:
        print('root finding failed')

    cf = solution.x
    cf = cf.reshape((dim, cf.size // dim))

    return [np.polynomial.chebyshev.Chebyshev(cf[i], (a, b)) for i in range(dim)]


if __name__ == '__main__':
    # # constants
    # gamma = 5 / 3
    # v_plus = 3 / 5
    # v_star = gamma / (gamma + 2)
    # e_minus = (gamma + 2) * (v_plus - v_star) / 2 / gamma / (gamma + 1)

    # # ODE
    # f = lambda x, y: np.array([y[0] * (y[0] - 1) + gamma * (y[1] - y[0] * e_minus),
    #                            y[0] * (-(y[0] - 1) ** 2 / 2 + y[1] - e_minus + gamma * e_minus * (y[0] - 1))])

    # a = 0
    # b = 20
    # n = 40
    # dim = 2

    # bc = lambda ya, yb: [ya[0] - np.array([3 / 50, 1]), yb[0] - np.array([9 / 50, 3 / 5]), yb[1] - np.array([0, 0])]

    f = lambda x, y: np.array([
        y[1],
        -np.exp(y[0])
    ])

    a = 0
    b = 1
    n = 20
    dim = 2

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    # f = lambda x, y: y

    # a = 0
    # b = 1
    # n = 10
    # dim = 1

    # def bc(ya, yb):
    #     return ya[0] - 1

    u0 = np.zeros((n + 1, dim))
    u0[0, 0] = 3
    u0[:, 1] = np.exp(u0[:, 0])

    solution = our_own_bvp_solve(f, a, b, n, u0, dim, bc)

    # plotting
    dom = np.linspace(a, b, 1000)
    plt.plot(dom, solution[0](dom),
            #  np.polynomial.chebyshev.Chebyshev(cf[1], (a, b))(dom),
             label='estimate')
    plt.legend()
    plt.show()
