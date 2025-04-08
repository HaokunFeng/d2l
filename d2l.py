import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import time
import numpy as np
import torch

class d2l:
    """Deep Learning Utilities"""
    
    @staticmethod
    def use_svg_display():
        """Use SVG for matplotlib."""
        backend_inline.set_matplotlib_formats('svg')

    @staticmethod
    def set_figsize(figsize=(3.5, 2.5)):
        """Set the figure size."""
        d2l.use_svg_display()
        plt.rcParams['figure.figsize'] = figsize

    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for the plot."""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    @staticmethod
    def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, 
             xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), 
             figsize=(3.5, 2.5), axes=None):
        """Plot the data."""
        if legend is None:
            legend = []
        
        d2l.set_figsize(figsize)
        axes = axes if axes else plt.gca()

        def has_one_axis(X):
            return (hasattr(X, "ndim") and X.ndim == 1 or 
                    isinstance(X, list) and not hasattr(X[0], "__len__"))
        
        if has_one_axis(X):
            X = [X]
        if Y is None:
            X, Y = [[]] * len(X), X
        elif has_one_axis(Y):
            Y = [Y]
        if len(X) != len(Y):
            X = X * len(Y)
        axes.cla()
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
        d2l.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
    @staticmethod
    def synthetic_data(w, b, num_examples):
        """Generate y = Xw + b + noise."""
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b
        y  += torch.normal(0, 0.01, y.shape)
        return X, y.reshape((-1, 1))
    
    @staticmethod
    def linreg(X, w, b):
        """The linear regression model."""
        return torch.matmul(X, w) + b
    
    @staticmethod
    def squared_loss(y_hat, y):
        """Squared loss."""
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    
    @staticmethod
    def sgd(params, lr, batch_size):
        """Minibatch stochastic gradient descent."""
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    class Timer:
        """Timer for measuring time taken by a block of code."""
        
        def __init__(self):
            self.time = []
            self.start()
        
        def start(self):
            """Start the timer."""
            self.tik = time.time()
        
        def stop(self):
            """Stop the timer and record the time taken."""
            self.time.append(time.time() - self.tik)
            return self.time[-1]

        def avg(self):
            return sum(self.time) / len(self.time) if self.time else 0
        
        def sum(self):
            return sum(self.time)
        
        def cumsum(self):
            return np.array(self.time).cumsum().tolist()