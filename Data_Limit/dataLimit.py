import sys
sys.path.append("../src")

from CSOWP_SR import *
from ExpressionTree import *
from rollingWindow import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from pathos.multiprocessing import ProcessingPool as Pool
from copy import deepcopy

import os
import pickle


class trainRegions():

    def __init__(self, SR_model, dir_path=None):
        self.SR_model = SR_model
        self.dir_path = dir_path

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    def fit(self, X, y, nstart, ntimes=1, nend=None):
        self.X = X
        self.y = y
        self.nstart = nstart
        self.ntimes = ntimes

        if nend is None:
            self.nend = X.shape[0]
        else:
            self.nend = nend


    def run(self):

        for n in range(self.ntimes):

            for nPics in range(self.nstart, self.nend+1):
                rowi = rollingWindow()
                rowi.fit(self.X, self.y, self.SR_model, nPics = nPics)

                solutions = rowi.run()
                
                with open(self.dir_path + f"/solutions-{nPics}-{n}.pkl", "wb") as file:
                    pickle.dump(solutions, file)
    
    def _process_single_iteration(self, args):
        nPics, n, X, y, SR_model, dir_path = args
        
        class SR_model():
            def __init__(self):
                self.SR = PySRRegressor(
                    unary_operators=["exp"],
                    temp_equation_file=True,
                    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                    warm_start=False
                )

            def fit(self, X, y):
                self.SR.fit(X, y)

            def get_solutions(self):
                return self.SR.sympy().simplify()


        rowi = rollingWindow()
        rowi.fit(X, y, SR_model, nPics=nPics)
        solutions = rowi.run()

        with open(f"{dir_path}/solutions-{nPics}-{n}.pkl", "wb") as file:
            pickle.dump(solutions, file)
    
    def run_parallel(self, n_processes=8):
        task_args = [(nPics, n, self.X, self.y, deepcopy(self.SR_model), self.dir_path)
                 for n in range(self.ntimes)
                 for nPics in range(self.nstart, self.nend + 1)]

        with Pool(processes=n_processes) as pool:
            pool.map(self._process_single_iteration, task_args)

                

    
        