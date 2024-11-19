import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from examples.utils import *
from monviso import VI

M, n = 2, 1

# Create the problem variables
# This VI is solved for every x[0] = x[1] s.t. x[0]>=0
A = np.array([[0, 1], [-1, 0]])
F = lambda x: A @ x
x = cp.Variable(n * M)
q = np.array([1, -1])
S = [q @ x >= 0] # x[0] >= x[1]
L = np.linalg.norm(A, 2)

# Select the (feasible) point (-1, -1)
x_des = [1, 1]
J = lambda x: (x - x_des)

# Create the VI and the initial solution(s)
sso = VI(F, n=n * M, S=S)
x0 = np.random.rand(n * M)

# Solve the VI using the available algorithms
max_iter = 1000

algorithm = "fbf_hsdm"
params = {"x": x0, "J": J, "step_size": .5 / L, "alpha": .7}
sol_hsdm = sso.solution(
    algorithm,
    params,
    max_iter,
    eval_func= lambda x: np.linalg.norm(x-x_des),
    log_path=f"examples/logs/basic-selection/{algorithm}.log",
)

algorithm = "fbf"
params = {"x": x0, "step_size": .5 / L}
sol_fbf = sso.solution(
    algorithm,
    params,
    max_iter,
    eval_func=lambda x: np.linalg.norm(x-x_des),
    log_path=f"examples/logs/basic-selection/{algorithm}.log",
)

plot_results(
    "examples/logs/basic-selection",
    "examples/figs/basic-selection.pdf",
    r"$\|x-x^{\text{des}}\|$",
)
