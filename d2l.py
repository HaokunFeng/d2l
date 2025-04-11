import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import time
import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display

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
    
    @staticmethod
    def get_fashion_mnist_labels(labels):
        """Return text labels for the Fashion-MNIST dataset."""
        text_labels = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        return [text_labels[int(i)] for i in labels]
    
    @staticmethod
    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
        """Display a list of images."""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                ax.imshow(img.numpy())
            else:
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        return axes
    
    @staticmethod
    def get_dataloader_workers():
        """Return the number of workers for data loading."""
        return 4

    @staticmethod
    def load_data_fashion_mnist(batch_size, resize=None):
        """Download the Fashion-MNIST dataset and load it into memory."""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../data", train=True, transform=trans, download=True
        )
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../data", train=False, transform=trans, download=True
        )
        return (
            data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers())
        )
    
    @staticmethod
    def load_array(data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    @staticmethod
    def accuracy(y_hat, y):
        """Compute the accuracy."""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    @staticmethod
    def evaluate_accuracy(net, data_iter):
        if isinstance(net, torch.nn.Module):
            net.eval()
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(d2l.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    class Accumulator:
        """Accumulate the sum and count."""
        
        def __init__(self, n):
            self.data = [0.0] * n
        
        def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)]
        
        def reset(self):
            self.data = [0.0] * len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    @staticmethod
    def train_epoch_ch3(net, train_iter, loss, updater):
        """Train a model for one epoch."""
        if isinstance(net, torch.nn.Module):
            net.train()
        metric = d2l.Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(X.shape[0])
            metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]
    
    @staticmethod
    def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
        """Train a model."""
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = d2l.evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc
    
    @staticmethod
    def predict_ch3(net, test_iter, n=6):
        """Predict on the test set."""
        for X, y in test_iter:
            break
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    
    @staticmethod
    def evaluate_loss(net, data_iter, loss):
        """Evaluate the loss on the dataset."""
        metric = d2l.Accumulator(2)
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
        return metric[0] / metric[1]
    
    class Animator:
        """Plotting class for training."""
        
        def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                     ylim=None, xscale='linear', yscale='linear',
                     fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
            if legend is None:
                legend = []
            d2l.use_svg_display()
            self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
            if nrows * ncols == 1:
                self.axes = [self.axes, ]
            self.config_axes = lambda: d2l.set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
            )
            self.X, self.Y, self.fmts = None, None, fmts
        
        def add(self, x, y):
            """Add data to the plot."""
            if not hasattr(y, "__len__"):
                y = [y]
            n = len(y)
            if not hasattr(x, "__len__"):
                x = [x] * n
            if not self.X:
                self.X = [[] for _ in range(n)]
            if not self.Y:
                self.Y = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(x, y)):
                if a is not None and b is not None:
                    self.X[i].append(a)
                    self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.display(self.fig)
            display.clear_output(wait=True)
