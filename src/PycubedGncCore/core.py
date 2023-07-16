import autograd.numpy as np
from autograd import jacobian, grad
from collections import namedtuple

Jacobians = namedtuple('Jacobians', ['dx', 'du'])
Cost = namedtuple('Cost', ['stage', 'final'])
Hessians = namedtuple('Hessians', ['xx', 'xu', 'ux', 'uu'])
Gradients = namedtuple('Gradients', ['x', 'u'])


def calculate_dynamics_jacobians(f, x, u):
    return Jacobians(
        dx=jacobian(lambda: f(x, u), x),
        du=jacobian(lambda: f(x, u), u),
    )


def calculate_gradients(dj: Jacobians, p, q, r):
    A = dj.dfdx
    B = dj.dfdu

    return Gradients(
        x=q + A.T @ p,
        u=r + B.T @ p,
    )


def calculate_hessian_ilqr(df: Jacobians, cost: Hessians, P):
    A = df.dx
    B = df.du
    Gxu = cost.xu + A.T @ P @ B
    return Hessians(
        Gxx=cost.xx + A.T @ P @ A,
        Guu=cost.uu + B.T @ P @ B,
        Gxu = Gxu,
        Gux = Gxu.T,
    )


def backward_pass(N, f, cost: Cost, x, u, K):
    delta_J = 0.0
    p = [0 for _ in N]
    P = [0 for _ in N]
    p[N-1] = grad(cost.final, x[N-1])
    P[N-1] = jacobian(cost.final, x[N-1])
    d = [0 for _ in N]
    K = [0 for _ in N]

    for i in range(N-2, -1, -1):
        dj = calculate_dynamics_jacobians(f, x[i], u[i])
        q = grad(lambda: cost.stage(x, u[i]), x)
        r = grad(lambda: cost.stage(x[i], u), u)
        g = calculate_gradients(dj, p[i+1], q, r)

        G = calculate_hessian_ilqr(dj, G, P[i+1])

        # concider adding regularization here if DDP is being used

        d[i] = np.linalg.inv(G.uu) * g.u
        K[i] = np.linalg.inv(G.uu) * G.ux

        P[i] = G.xx + K[i].T @ G.uu @ K[i] - G.xu @ K[i] - K[i].T @ G.ux
        p[i] = g.x - K[i].T @ g.u + K[i].T @ G.ux * d[i] - G.xu * d[i]

        delta_J += np.dot(g.u, d[i])

    return d, K, delta_J


def forward_rollout(f, x, u, N, alpha, d, K):
    """
    Forward rollout of the system

    Arguments:
        f {function} -- system dynamics
        x {list} -- state trajectory
        u {list} -- control trajectory
        N {int} -- dynamics steps
        alpha {float} -- control penalty
        d {list} -- disturbance trajectory
        K {list} -- feedback gains
    """
    xp = [x[0]]
    up = []
    for i in range(1, N):
        up.append(
            u[i] - alpha * d[i] - K[i] * (xp[i] - x[i])
        )
        xp.append(
            f(xp[i], up[i])
        )

    return xp, up


def line_search(f, J, x, u, d, K, N, delta_J, g: Gradients, beta=0.1, c=0.5, max_iter=20):
    """
    Line search 

    Arguments:
        f {function} -- system dynamics
        J {function} -- cost function
        x {list} -- state trajectory
        u {list} -- control trajectory
        d {list} -- disturbance trajectory
        K {list} -- feedback gains
        N {int} -- dynamics steps
        alpha {float} -- control penalty
        beta {float} -- line search parameter
        c {float} -- line search parameter
        max_iter {int} -- maximum number of iterations
    """
    alpha = 1
    x, u = forward_rollout(f, x, u, N, alpha, d, K, g)
    initial_cost = J(x, u)
    while True:
        if max_iter == 0:
            print("Line search failed")
        max_iter -= 1
        x, u = forward_rollout(f, x, u, N, alpha, d, K)

        alpha *= c
        if J(x, u) <= initial_cost + beta * delta_J:
            return x, u


def make_cost_function(cost: Cost, N):
    """
    Make cost function

    Arguments:
        cost {Cost} -- cost function
        N {int} -- dynamics steps

    Returns:
        function -- cost function of state, and control
    """
    def J(x, u):
        total_cost = 0
        for i in range(N-1):
            total_cost += cost.stage(x[i], u[i])
        total_cost += cost.final(x[N-1])
        return total_cost

    return J


def core_solve(f, N, cost: Cost, x, u, max_iters=10, alpha=0.5, exit_tolerance=1e-6):
    iterations = 0
    d = [1 for i in range(N-1)]
    while max(abs(d)) > 1e-3:
        iterations += 1
        if iterations > max_iters:
            raise Exception("Max iterations reached")
        d, K, delta_J = backward_pass(N, f, cost, x, u, K)
        J = make_cost_function(J, cost, N)
        x, u, delta_J = line_search(f, J, x, u, d, K, N, alpha)
        if delta_J < exit_tolerance:
            break
    pass
