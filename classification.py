import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_model import DataGenerate
from csv import *

# Generate Dataset

with open("D:\ML Datasets\data.csv","+a") as file:
    x,y = DataGenerate().x,DataGenerate().y
    for i in range(len(x)):
        file.write(f"{x[i]},{y[i]}\n")


# Classification model

class Classification:
    def __init__(self) -> None:
        df = pd.read_csv(r"data.csv",names=["x","y"])
        self.x=np.array(df["x"])
        self.y=np.array(df["y"])
        self.w = np.array([5,-5])
        self.sign = {}
    
    def classification(self):
        for i in range(len(self.x)):
            score = 1 if np.dot(self.w.T,[self.x[i],self.y[i]])>=0 else -1
            self.sign[(self.x[i],self.y[i])] = score
        return self.sign
    
    def data_visualisation(self):
        fig,ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        for i in range(len(self.classification())):
            plt.plot(self.x[i],self.y[i],".",color="red") if self.classification()[(self.x[i],self.y[i])]==1 else plt.plot(self.x[i],self.y[i],"+",color="blue")
        plt.show()


