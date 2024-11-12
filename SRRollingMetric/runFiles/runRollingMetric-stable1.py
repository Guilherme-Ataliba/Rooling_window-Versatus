import sys
sys.path.append("../src")

from CSOWP_SR import *
from ExpressionTree import *
from rollingWindow import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
import numpy as np

def t1(X):
    return 10*np.exp(-0.5*np.exp(-0.5*X + 2))
X1 = np.linspace(-5, 15, 140)
y1 = t1(X1)

def t2(X):
    return np.exp(-X)
X2 = np.linspace(1, 20, 140)
y2 = t2(X2)

air_passengers = pd.read_csv("data/AirPassengers.csv")
X3 = air_passengers.index
y3 = air_passengers["#Passengers"]

treated_seco = pd.read_csv("data/treated_seco.csv")
treated_seco

treated_seco["eventdate"] = pd.to_datetime(treated_seco["eventdate"])
treated_seco.set_index("eventdate", inplace=True, drop=True)
treated_seco = treated_seco.asfreq("7D")
treated_seco.rename(columns={"Unnamed: 0": "weeks"}, inplace=True)

X4 = treated_seco.weeks[::2]
y4 = treated_seco.value[::2]

data = [
    [X1, y1],
    [X2, y2],
    [X3, y3],
    [X4, y4]
]

class rollingSR():
    def __init__(self):
        self.SR = PySRRegressor(
            binary_operators=["+", "-", "*", "/"], 
            unary_operators=["exp", "expp(x) = exp(-x)", "cos", "sin"],
            temp_equation_file=True,
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            extra_sympy_mappings={"expp": lambda x: smp.exp(-x)},
            niterations=60,
            populations=20,
            population_size=60,
            warm_start=False,
            verbosity=False,
            progress=False
        )

    def fit(self, X, y):
        self.SR.fit(X, y)

    def get_solutions(self):
        return self.SR.sympy().simplify()


# Running


RoWi = SRRollingMetric(dir_path="Outputs/Stable1")
start_points1 = [-1, -30, -60, -90, -110]
n_points = 30

for start in start_points1:
    RoWi.fit(X1, y1, rollingSR, direction="left", start=X1[start], n_points=n_points, visualize=False, n_runs=3)
    RoWi.run()
    