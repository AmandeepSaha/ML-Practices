import numpy as np
import matplotlib.pyplot as plt

class DataGenerate:
    def __init__(self):
        self.rng = np.random
        self.x_values = []
        self.y_values = []
        self.x = np.array([self.rng.randint(0,100) for i in range(100)])
        self.y = np.array([self.rng.randint(0,100) for i in range(100)])
        
    def x(self):
        return self.x

    def y(self):
        return self.y
    
    def provided_model(self,m=-np.pi/3,c=90,x=np.linspace(0,100,20000)):
        self.m = m
        self.c = c
        return x,m*x + c


class DataVisualization:
    def __init__(self):
        self.data = DataGenerate()
        self.x = self.data.x
        self.y = self.data.y
        self.model = self.data.provided_model()
    
    
    def data_model_visualization(self):
        fig,ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.plot(self.model[0],self.model[1],color="red")
        plt.plot(self.x,self.y,".")
        plt.show()


class LinearClassificationModel:
    def __init__(self):
        self.data = DataGenerate()
        self.x = self.data.x
        self.y = self.data.y
        self.model = self.data.provided_model()
        
    def erroes(self):
        fig,ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        self.err = self.y - (-(np.pi/3)*self.x + 90)
        for i in range(len(self.err)):
            if self.err[i]>0:
                plt.plot(self.x[i],self.y[i],"+",color="red")
            elif self.err[i]<0:
                plt.plot(self.x[i],self.y[i],".",color="blue")
        plt.show()

# LinearClassificationModel().erroes()