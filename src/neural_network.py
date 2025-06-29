import torch
import numpy as np

from dataclasses import dataclass
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
        
        
## Neural Network with built-in scaling and modular number of layers and neurons

class FFNN(nn.Module):
    """
    A class representing a fully-connected feedforward neural network model.

    Args:
        NNscale (dict): A dictionary containing the mean and standard deviation values for scaling the input/output data.
        activation (str): The activation function to use. Supported options: 'relu', 'sigmoid', 'tanh'. Default: 'tanh'.
        n_layers (int): The number of hidden layers in the neural network. Default: 1.
        n_neurons (int or list of int): The number of neurons in each hidden layer. Default: 30.
        device (torch.device): The device on which to perform computations. Default: torch.device('cpu').
        dtype (torch.dtype): The data type to use for the model. Default: torch.float32.

    Attributes:
        device (torch.device): The device on which the model is running.
        dtype (torch.dtype): The data type used for the model.
        xmean (torch.tensor): The mean of the input data used for scaling.
        xstd (torch.tensor): The standard deviation of the input data used for scaling.
        ymean (torch.tensor): The mean of the output data used for scaling.
        ystd (torch.tensor): The standard deviation of the output data used for scaling.
        nin (int): The number of input nodes.
        nout (int): The number of output nodes.
        activation (str): The activation function used by the model.
        n_layers (int): The number of hidden layers in the neural network.
        n_neurons (list of int): The number of neurons in each hidden layer.
        actfun (nn.Module): The PyTorch activation function module used by the model.
        fclist (nn.ModuleList): The list of PyTorch sequential modules representing the hidden layers of the model.
        out (nn.Linear): The output layer of the model.

    Methods:
        forward(x): Computes the forward pass of the neural network model.

    """
    def __init__(
            self,
            NNscale,
            activation='tanh',
            n_layers: int | None=1,
            n_neurons: int | list[int]=30,
            device=torch.device('cpu'),
            dtype=torch.float32
        ):
        super(FFNN, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.xmean = torch.tensor(NNscale['xmean'],device=self.device,dtype=self.dtype).view(1,-1)
        self.xstd = torch.tensor(NNscale['xstd'],device=self.device,dtype=self.dtype).view(1,-1)
        self.ymean = torch.tensor(NNscale['ymean'],device=self.device,dtype=self.dtype).view(1,-1)
        self.ystd = torch.tensor(NNscale['ystd'],device=self.device,dtype=self.dtype).view(1,-1)
        
        self.nin = NNscale['xmean'].shape[0]
        self.nout = NNscale['ymean'].shape[0]
        
        self.activation = activation
        self.n_layers = n_layers
        if isinstance(n_neurons,int):
            self.n_neurons = [self.nin]+([n_neurons]*n_layers)
        elif hasattr(n_neurons, "__len__") and len(n_neurons)==n_layers and all(isinstance(x,(int,np.generic)) for x in n_neurons):
            self.n_neurons = [self.nin]+list(n_neurons)
        else:
            raise Exception(
                f'Parameter n_neurons ill defined: expected int or array of length {n_layers}, got {n_neurons}!'
            )
        
        # store in the NN model flags for which components are modeled - states and cost functions
        activations = {'relu':nn.ReLU(),'sigmoid':nn.Sigmoid(),'tanh':nn.Tanh()}
        try:
            self.actfun = activations[self.activation]
        except Exception as e:
            raise Exception(f'Unknown activation function! Supported activations: [relu,sigmoid,tanh]; received: {self.activation}.')

        self.fclist = torch.nn.ModuleList(
            [nn.Sequential(
                nn.Linear(
                    self.n_neurons[i], 
                    self.n_neurons[i+1], 
                    device=self.device, 
                    dtype=self.dtype
                ),
                self.actfun
            ) for i in range(self.n_layers)]
        )
        self.out = nn.Linear(self.n_neurons[-1], self.nout,device=self.device,dtype=self.dtype)

        self.init_state_dict = self.state_dict()


    @classmethod
    def from_state_dict(cls, state_dict, NNscale, activation='tanh', n_layers=1, n_neurons=30, device=torch.device('cpu'), dtype=torch.float32):
        model = cls(NNscale, activation, n_layers, n_neurons, device, dtype)
        model.load_state_dict(state_dict)
        model.init_state_dict = state_dict
        return model


    def forward(self, x):
        x = (x - self.xmean)/self.xstd
        for i in range(self.n_layers):
            x = self.fclist[i](x)
        x = self.out(x)
        return x   
    




def train_NN(
        model: FFNN, 
        trainloader: DataLoader, 
        criterion: nn.modules.loss, 
        n_epochs: int, 
        noise: float, 
        lr: float, 
        weight_decay: float, 
        show_tqdm: bool
    ):
        """
        Train the neural network on the defined training set.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            A data loader containing the training set.
        criterion : torch.nn.modules.loss
            A loss function to evaluate the training error.
        n_epochs : int
            The number of epochs to train for.
        noise : float
            The standard deviation of the Gaussian noise to add to the inputs during training.
        lr : float
            The learning rate for the optimizer.
        weight_decay : float
            The weight decay for the optimizer.
        show_tqdm : bool
            Whether to show a tqdm progress bar during training.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        for epoch in trange(int(n_epochs),desc = f'NN training: [{model.nin}] -> [{model.nout}]',mininterval=0.5,disable = not show_tqdm):
            for sample in trainloader:
                x_batch, y_batch = sample
                x_batch *= (1+torch.randn_like(x_batch) * noise)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, (y_batch-model.ymean)/model.ystd)
                loss.backward()
                optimizer.step()

        return model



@dataclass
class Train_Param:
    batch_size: int 
    shuffle: bool
    n_epochs: int
    noise: float
    lr: float
    weight_decay: float
    criterion: nn.modules.loss