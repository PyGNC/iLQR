from .core import core_solve, Cost
from .rk4 import rk4

def interpolate(start, end, steps):
    return [start + i * (end - start) / steps for i in range(steps)]

def iLQR(dynamics, stage_cost, final_cost, u0, x0, xgoal, N, dt):
    def rk4_dynamics(x, u):
        return rk4(dynamics, x, u, dt)

    cost = Cost(
        stage=stage_cost,
        final=final_cost
    )

    x = interpolate(x0, xgoal, N)
    u = u0 * (N - 1)

    return core_solve(rk4_dynamics, N, cost, x, u)
