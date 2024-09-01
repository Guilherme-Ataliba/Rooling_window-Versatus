import pandas as pd
import numpy as np
from random import seed
import matplotlib.pyplot as plt
import sympy as smp
import warnings
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from typing import *
from inspect import isclass, signature
import os

from pathos.multiprocessing import ProcessingPool


class rollingWindow():

    def __init__(self, SEED=42, ignore_warnings=False, dir_path=None):
        self.ignore_warning = ignore_warnings
        self.SEED = SEED
        self.functions = None
        self.dir_path = dir_path

        if self.dir_path is not None:
            if not os.path.isdir(self.dir_path):
                os.makedirs(self.dir_path)
        


    def fit(self, X: np.ndarray, y: np.ndarray, SR_model, x_range:Tuple[float, float]=None, L: float = None, nPics: int = None, visualize=False):
        
        self.X = X
        self.y = y

        if not isclass(SR_model):
           raise TypeError("SR_model must be a class, not an instantiated object")
        if not callable(getattr(SR_model, "fit")):
           raise AttributeError("SR_model must have a fit method")
        if not callable(getattr(SR_model, "get_solutions")):
           raise AttributeError("SR_model must have a get_solutions method")

        self.SR_class = SR_model

        if len(X.shape) > 1:
            if X.shape[1] > 1:
                self.X0 = X[:, 0]
            else:
                self.X0 = X
        else:
          self.X0 = X

        if x_range is None:
            self.x_range = (self.X0.min(), self.X0.max())
        else:
            self.x_range = x_range

        self.a = self.x_range[0]
        self.b = self.x_range[1]
        self.L = L
        self.nPics = nPics
        
        if (L is None) and (nPics is None):
            raise ValueError("L and nPics can't be simultaneously None, one or both must be informed.")
        elif L is None:
            self.L = (self.b-self.a)/nPics
        elif nPics is None:
            self.nPics = (self.b-self.a)/L

        self.nPics = int(self.nPics)
        self.stepSize = (self.b - self.a)/self.nPics

        if visualize is True:
            self.visualize()


    def _filter(self, step):
        XStep = np.c_[self.X[(self.X0 >= step) & (self.X0 < step + self.L)]]
        yStep = np.c_[self.y[(self.X0 >= step) & (self.X0 < step + self.L)]]
        return XStep, yStep

    def _execute(self, step, SR_model):
        XStep, yStep = self._filter(step)
        SR_model.fit(XStep, yStep)
        solution = SR_model.get_solutions()
        return solution

    def run(self, n_processes=0, overwrite=False):
        if self.ignore_warning:
            warnings.filterwarnings("ignore")

        if self.dir_path is not None:
            if os.path.isfile(os.path.join(self.dir_path, "solutions.pickle")) and not overwrite:
                raise FileExistsError("solutions.pickle exists")
        
        np.random.seed(self.SEED)
        seed(self.SEED)

        end = self.b-self.stepSize
        step = self.a

        solutions = {}

        # Rolling Window

        # Run in Serial
        if n_processes == 0:
            SR_model = self.SR_class()

            # Rolling Window
            print(f"Training {step}/{end}")
            sol = self._execute(step, SR_model)
            solutions[step] = sol
            step += self.stepSize
            while step < end:
                print(f"Training {step}/{end}")
                sol = self._execute(step, SR_model)
                solutions[step] = sol
                step += self.stepSize
            print(f"Training {step}/{end}")
            sol = self._execute(step, SR_model)
            solutions[step] = sol

        # Run in Parallel
        else:
            stepList = [step]
            step += self.stepSize
            while step < end:
                stepList.append(step)
                step += self.stepSize
            stepList.append(step)

            if n_processes > len(stepList):
                SR_num = len(stepList)
            else:
                SR_num = n_processes
            SR_list = [self.SR_class() for _ in range(SR_num)]

            args = []
            c = 0
            for step in stepList:
                args.append( (step, SR_list[c]) )
                c = (c+1)%SR_num

            with ProcessingPool(processes=n_processes) as pool:
                results = pool.map(lambda arg: self._execute(arg[0], arg[1]), args)

            solutions = list(zip(stepList, results))
    
        # Return
        self.solutions = solutions
        
        if self.dir_path:
            with open(os.path.join(self.dir_path, "solutions.pickle") , "wb") as file:
                pickle.dump(solutions, file)


        return solutions
        
    def set_functions(self, functions):
      self.functions = functions
    
    def multi_plots(self, x_range, n_points=1000, axes=None):
        if self.functions is None:
            raise RuntimeError("You must call set_functions to inform the functions array to use")
        
        if axes is not None:
            axes = axes.flatten()
        else:
            axes = [plt.gca() for _ in range(len(self.functions))]

        X = np.linspace(x_range[0], x_range[1], n_points)
        
        
        for c, func in enumerate(self.functions):
            
            if func.__code__.co_argcount >= 1:
                y = func(X)
            else:
                y = np.array([func() for _ in range(len(X))])

            if type(y) is not np.ndarray:
                y = np.array([y for _ in X])

            axes[c].plot(X, y, label=c)
            axes[c].set_title(f"Solution - Interval {c+1}")
        plt.show()

    def plot_over(self, X=None):
      if self.functions is None:
        raise RuntimeError("You must call set_functions to inform the functions array to use")
      
      plt.plot(self.X0, self.y)

      end = self.b-self.stepSize

      for c, step in enumerate(np.arange(self.a, end+self.stepSize, self.stepSize)):
        X_step, _ = self._filter(step)

        func = self.functions[c]

        if func.__code__.co_argcount >= 1:
            
            # Getting the indexes for each feature the function used
            feature_indexes = signature(func).parameters.keys()
            feature_indexes = [int(feature[1:]) for feature in feature_indexes]

            features = [X_step[:, i] for i in feature_indexes]

            y = func(*features)
        else:
            y = np.array([func() for _ in range(len(X_step))])

        plt.plot(X_step, y)


    def visualize(self, bg_palette="inferno", bg_alpha=0.2,
                  bg_linecolor="black", bg_linewidth=1,
                  linecolor="black", linewidth=3,
                  linestyle="dashed"):
        ax = plt.gca()
        ax.plot(self.X, self.y, linewidth=linewidth, 
                c=linecolor, linestyle=linestyle)
        ax.set_xlim(self.X.min(), self.X.max())
        ax.set_ylim(self.y.min(), self.y.max())

        color_palette = sns.color_palette(bg_palette, self.nPics)

        Xmin = self.X.min()
        ymin = self.y.min()
        ymax = abs(self.y.min() - self.y.max())

        # Add a rectangle with a background color
        for c in range(self.nPics):
            x_start = Xmin + c*self.stepSize

            rect = patches.Rectangle((x_start, ymin), self.L, ymax,
                                      linewidth=bg_linewidth, edgecolor=bg_linecolor,
                                        facecolor=color_palette[c], alpha=bg_alpha)
            ax.add_patch(rect)

        # Display the plot
        plt.show()
























