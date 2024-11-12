import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as smp
import plotly 
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle
import edist.ted as ted

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import rollingWindow

import sys
sys.path.append("../src")

import utils
import ExpressionTree

import re
import warnings
import logging

sns.set_theme("notebook")


class AnalysisPipeline():
    def __init__(self, X, y, plot_interval):
        sns.set_theme("notebook")
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

        self.X = X
        self.y = y
        self.plot_interval = plot_interval
        self.plot_X = np.arange(*plot_interval)

        self.MSE = False
        self.matches = False
        self.std = False

        self.df = None
        self.matches_df = None

        self.LSTM_data = None
        self.LSTM_vector_data = None
        self.regressor = None


    def _create_partial_df(self, path):
        with open(path, "rb") as file:
            air_info = pickle.load(file)

        interval = list(air_info.keys())
        funcs = list(air_info.values())

        air_df = pd.DataFrame()
        air_df["interval"] = interval
        air_df["func"] = funcs


        x0 = smp.symbols("x0")
        air_lamb = []

        for i, row in air_df.iterrows():
            lamb = smp.lambdify([x0], row.func)   
            air_lamb.append(lamb)

        air_df["lamb"] = air_lamb

        starts, ends, iters = [], [], []
        pattern = re.compile(r"\d+")
        for row in air_df["interval"]:
            start, end, it = pattern.findall(row)
            
            starts.append(int(start))
            ends.append(int(end))
            iters.append(int(it))

        air_df["start"] = starts
        air_df["end"] = ends
        air_df["iteration"] = iters
        air_df.drop("interval", axis=1, inplace=True)

        sizes = []
        for row in air_df["func"]:
            size = len(utils.exprToTree(row))
            sizes.append(int(size))

        air_df["size"] = sizes

        return air_df

    def create_df(self, path, n_points=None):
        dfs = []
        for i in range(0, 2):
            path_new = rf"{path}-{i}.pkl"
            df = self._create_partial_df(path_new)
            dfs.append(df)

        df = pd.concat(dfs).sort_values("end").reset_index(drop=True)

        df["window_size"] = df["start"] - df["end"]

        if n_points is not None:
            self.df = df[df["window_size"] <= n_points].reset_index(drop=True)
        else:
            self.df = df

    def plot_data(self):
        if self.df is None:
            raise ValueError("You must first create the data using create_df")

        RoWi = rollingWindow.SRRollingMetric()

        class rollingSR():
            def __init__(self):
                pass

            def fit(self, X, y):
                pass

            def get_solutions(self):
                pass

        plt.figure(figsize=(12, 6))
        RoWi.fit(self.X, self.y, rollingSR, direction="left", start=self.X[self.df["start"][0]], 
                 n_points=int(self.df["window_size"].max())+1, visualize=True)
        plt.show()


    def plot_complexity(self, style="line", figsize=(12, 6), invert_axis=True):
        plt.figure(figsize=figsize)
        
        if style.lower() == "line":
            sns.lineplot(self.df, x="window_size", y="size")
        elif style.lower() == "boxplot":
            sns.boxplot(data=self.df, x="window_size", y="size")
        else:
            raise ValueError("Invalid style option")

        plt.title('Solution Complexity X Window Size')

        plt.show()

    def _calculate_MSE_single(self, a, b, func):
        X = self.X[a:b]
        y = self.y[a:b]
        y_pred = func(X)

        return np.mean((y - y_pred)**2)
    
    def _calculate_MSE(self):
        MSEs = []
        for i, row in self.df.iterrows():
            MSE = self._calculate_MSE_single(row.end, row.start, row.lamb)
            MSEs.append(MSE)

        self.df["MSE"] = MSEs

        self.MSE = True
        

    def plot_MSE(self, style = "line", y_log_scale=True, invert_axis=True):
        if not self.MSE:
            self._calculate_MSE()
        
        plt.figure(figsize=(14, 6))

        if style.lower() == "line":
            sns.lineplot(self.df, x="window_size", y="MSE")
            sns.scatterplot(self.df, x="window_size", y="MSE", alpha=0.5, c="black")
        elif style.lower() == "boxplot":
            sns.boxplot(self.df, x="window_size", y="MSE")
        else:
            raise ValueError("Invalid style option")

        if y_log_scale:
            plt.yscale("log")

        plt.xticks(np.arange(self.df.window_size.min(), self.df.window_size.max()+1))

        plt.title('MSE X Window Size')
        plt.show()

    def plot_solutions(self, filter_value=None, y_range=(-200, 600), size=(1200, 600)):
        if filter_value is not None:
            df = self.df[self.df["window_size"] >= filter_value].copy()
        else:
            df = self.df.copy()

        # Create a figure
        fig = go.Figure()

        #Calculate ys
        ys = []
        for i, row in df.iterrows():
            y = row.lamb(self.plot_X)

            try:
                len(y)
            except:
                y = [y for _ in self.plot_X]

            ys.append(y)

        # Add traces for each Y column
        c = 0
        for i, row in df.iterrows():
            fig.add_trace(go.Scatter(x=self.plot_X, y=ys[c], name=f"{c}"))
            c += 1

        # Create dropdown
        # Set the initial y-axis range
        fig.update_layout(yaxis=dict(range=y_range))  # Default y-axis range

        fig.update_layout(
            width=size[0],  # Set the width of the plot
            height=size[1],  # Set the height of the plot
        )

        # Set the title and display the figure
        fig.update_layout(title='All Solutions')
        fig.show()

    def progressively_plot_solutions(self, filter_value=None, y_range=(-200, 600), size=(1200, 600)):
        if filter_value is not None:
            df = self.df[self.df["window_size"] >= filter_value].copy()
        else:
            df = self.df.copy()
        
        df_invert = df.sort_values("end", ascending=False)

        # Create a figure
        fig = go.Figure()

        #Calculate ys
        ys = []
        for i, row in df_invert.iterrows():
            y = row.lamb(self.plot_X)

            try:
                len(y)
            except:
                y = [y for _ in self.plot_X]

            ys.append(y)

        # Add traces for each Y column
        c = 0
        for i, row in df_invert.iterrows():
            fig.add_trace(go.Scatter(x=self.plot_X, y=ys[c], name=f"{c}",
                                    visible=True if i == len(df_invert)-1 else False  # Initially show only the first line
                                    ))
            c += 1

        # Create steps for the slider, each controlling how many lines are shown
        num_lines = len(df_invert)
        steps = []
        for i in range(num_lines):
            step = dict(
                method="update",
                args=[{"visible": [j <= i for j in range(num_lines)]},  # Show lines up to the ith one
                    {"title": f"Showing {i+1} lines"}],  # Update plot title
                label=f"{i+1}"
            )
            steps.append(step)

        # Create dropdown
        # Set the initial y-axis range
        fig.update_layout(yaxis=dict(range=y_range))  # Default y-axis range

        fig.update_layout(
            width=size[0],  # Set the width of the plot
            height=size[1],  # Set the height of the plot
        )

        # Add slider to the layout
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Number of lines: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
        )


        # Set the title and display the figure
        # fig.update_layout(title='Interactive Data Selection')
        fig.update_layout(title='Progressively Show Solutions')

        
        fig.show()

    def _find_matches(self, filter_value):
        matches = []
        indexes = []
        range_array = np.arange(*self.plot_interval)

        if filter_value is not None:
            df = self.df[self.df["window_size"] >= filter_value].copy()
        else:
            df = self.df.copy()

        for c in range_array:
            ys = []
            for i, row in df.iterrows():
                y = row.lamb(c)
                ys.append(y)
            
            matches.append(ys)
            indexes.append(c)

        self.matches_df = pd.DataFrame({"indexes": indexes, "matches": matches})

        self.matches = True
    
    def _calculate_std(self, filter_value):
        self._find_matches(filter_value)

        stds = []
        for i, row in self.matches_df.iterrows():
            filtered = np.array(row.matches)[np.isfinite(row.matches)]
            filtered2 = filtered[filtered <= 1e10]
            std = np.nanstd(filtered2)
            stds.append(std)

        self.matches_df["std"] = stds
        
        self.std = True

    def plot_variation(self, style="std", figsize=(1200, 600), filter_value=None):        
        self._calculate_std(filter_value = filter_value)

        if style == "std":
            fig = px.line(self.matches_df, x="indexes", y="std", log_y=True,
                          width=figsize[0], height=figsize[1])

            fig.update_layout(
                yaxis=dict(
                    exponentformat="power",  # Show ticks as powers of 10
                )
            )

        elif style == "boxplot":
            exploded = self.matches_df.copy().explode("matches")
            exploded['matches'] = pd.to_numeric(exploded['matches'])
            exploded.reset_index(drop=True, inplace=True)

            
            fig = px.box(exploded, x="indexes", y="matches",
                        width=figsize[0], height=figsize[1])
            
        fig.update_layout(title='Data Variation Distribution')


        fig.show()



    # LSTMs
    def train_test_split(self, train_size=0.74, test_size=0.36, figsize=(14, 6)):
        scaler = StandardScaler()
        dado_escalado = scaler.fit_transform(np.c_[self.X, self.y])

        X3_scaled = dado_escalado[:, 0]
        y3_scaled = dado_escalado[:, 1]

        # Train & Test
        tamanho_treino = int(len(X3_scaled)*train_size)
        tamanho_teste = len(X3_scaled) - tamanho_treino

        X_train = X3_scaled[0:tamanho_treino]
        y_train = y3_scaled[0:tamanho_treino]

        X_test = X3_scaled[tamanho_treino:len(X3_scaled)]
        y_test = y3_scaled[tamanho_treino:len(X3_scaled)]

        # Test one and Two
        tamanho_teste = int(len(X_test)*test_size)

        X_test_one = X_test[0:tamanho_teste]
        y_test_one = y_test[0:tamanho_teste]

        X_test_two = X_test[tamanho_teste:len(X_test)]
        y_test_two = y_test[tamanho_teste:len(X_test)]

        self.LSTM_data = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "X_test_one": X_test_one, "y_test_one": y_test_one,
            "X_test_two": X_test_two, "y_test_two": y_test_two
        }

        plt.figure(figsize=figsize)
        plt.title("Data Split and Standardized for LSTM")
        sns.lineplot(x=X_train, y=y_train, label="Train Data")
        sns.lineplot(x=X_test_one, y=y_test_one, label="Test Data")
        sns.lineplot(x=X_test_two, y=y_test_two, label="Validation Data")
        plt.show()

    def _treat_data_for_LSTM(self):
        if self.LSTM_data is None:
            raise ValueError("You must first call train test split")

        def separa_dados(vetor, n_passos):
            X_novo, y_novo = [], []

            for i in range(n_passos, vetor.shape[0]):
                X_novo.append(list(vetor.loc[i-n_passos:i-1]))
                y_novo.append(vetor.loc[i])
            
            X_novo, y_novo = np.array(X_novo), np.array(y_novo)
            return X_novo, y_novo

        
        vetor = pd.DataFrame(self.LSTM_data["y_train"])[0]
        X_treino_novo, y_treino_novo = separa_dados(vetor, 1)

        vetor2 = pd.DataFrame(self.LSTM_data["y_test_one"])[0]
        X_teste_novo, y_test_novo = separa_dados(vetor2, 1)

        vetor3 = pd.DataFrame(self.LSTM_data["y_test_two"])[0]
        X_teste_novo_two, y_teste_novo_two = separa_dados(vetor3, 1)

        self.LSTM_vector_data = {
            "X_train_new": X_treino_novo, "y_train_new": y_treino_novo,
            "X_test_new": X_teste_novo, "y_test_new": y_test_novo,
            "X_test_two_new": X_teste_novo_two, "y_test_two_new": y_teste_novo_two
        }

    def _train_LSTM(self):
        if self.LSTM_vector_data is None:
            self._treat_data_for_LSTM()

        regressor = Sequential([
            Dense(8, input_dim=1, kernel_initializer="Ones",
                activation="linear", use_bias=True),
            Dense(64, kernel_initializer="random_uniform",
                activation="sigmoid", use_bias=True),
            Dense(1, kernel_initializer="random_uniform",
                activation="linear", use_bias=True)
        ])

        regressor.compile(loss="mean_squared_error", optimizer="adam")

        regressor.summary()
        
        regressor.fit(self.LSTM_vector_data["X_train_new"], self.LSTM_vector_data["y_train_new"], epochs=100)

        self.regressor = regressor

    def plot_LSTM_results(self, figsize=(14, 6)):
        if self.regressor is None:
            self._train_LSTM()

        y_predict_novo = self.regressor.predict(self.LSTM_vector_data["X_train_new"])
        y_predict_test_novo = self.regressor.predict(self.LSTM_vector_data["X_test_new"])
        y_predict_test_novo_two = self.regressor.predict(self.LSTM_vector_data["X_test_two_new"])

        print(len(self.LSTM_data["X_train"]), len(y_predict_novo.reshape(-1)))

        plt.figure(figsize=figsize)
        plt.title("LSTM Train Results")
        sns.lineplot(x=self.LSTM_data["X_train"], y=self.LSTM_data["y_train"], label="Train Data")
        sns.lineplot(x=self.LSTM_data["X_test"], y=self.LSTM_data["y_test"], label="Test Data")
        sns.lineplot(x=self.LSTM_data["X_train"][0:-1], y=y_predict_novo.reshape(-1), label="Train Prediction")
        sns.lineplot(x=self.LSTM_data["X_test_one"][0:-1], y=y_predict_test_novo.reshape(-1), label="Test Prediction")
        sns.lineplot(x=self.LSTM_data["X_test_two"][0:-1], y=y_predict_test_novo_two.reshape(-1), label="Test Validation")

        