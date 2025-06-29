import numpy as np

from typing import Literal
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm_notebook
from copy import deepcopy

from ..neural_network import FFNN, train_NN
from .prun_utils import calc_prun_amount, restore_prun_mask, calc_actual_amount
from .prun_dataclasses import Prun_Train_Param



def retrain_model(
        model: FFNN,
        pt_params: Prun_Train_Param,
        trainloader: DataLoader
    ):
    """
    Retrains the given neural network model after applying pruning, optionally restoring initial parameters.

    Parameters
    ----------
    ``model`` : FFNN
        The feedforward neural network model to be retrained.
    ``pt_params`` : Prun_Train_Param
        Parameters for pruning and retraining, including training criteria, epochs, noise, 
        learning rate, and weight decay.
    ``trainloader`` : DataLoader
        DataLoader providing the training data.

    Returns
    -------
    ``model`` : FFNN
        The retrained neural network model.
    """
    if pt_params.prun_param.use_init_params:
        init_model = FFNN.from_state_dict(
            model.init_state_dict, pt_params.scale, model.activation,
            model.n_layers, model.n_neurons[1:], model.device, model.dtype
        )
        model = restore_prun_mask(model, init_model)

    model = train_NN(model, trainloader, 
                     pt_params.train_param.criterion(), 
                     pt_params.train_param.n_epochs, 
                     pt_params.train_param.noise, 
                     pt_params.train_param.lr, 
                     pt_params.train_param.weight_decay,
                     show_tqdm=False)
    return model



def prun_and_retrain(
        model: FFNN, 
        pt_params: Prun_Train_Param,
        trainloader: DataLoader
):
    """
    Prunes the neural network model and then retrains it using the provided parameters and data.

    Parameters
    ----------
    ``model`` : FFNN
        The feedforward neural network model to be pruned and retrained.
    ``pt_params`` : Prun_Train_Param
        Parameters for pruning and retraining, including methods and training criteria.
    ``trainloader`` : DataLoader
        DataLoader providing the training data.

    Returns
    -------
    ``model`` : FFNN
        The pruned and retrained neural network model.
    """
    model = pt_params.prun_param.base_pruning(model)
    model = retrain_model(model, pt_params, trainloader)
    return model



def iter_prun_nodes(
        model: FFNN, 
        pt_params: Prun_Train_Param,
        dataset: Dataset = None,
        show_tqdm: bool = False
    ):
    """
    Iteratively prunes the neural network model and optionally retrains it at each step.

    Parameters
    ----------
    ``model`` : FFNN
        The feedforward neural network model to be pruned iteratively.
    ``pt_params`` : Prun_Train_Param
        Parameters for pruning and retraining, including pruning method, amount, and retraining criteria.
    ``dataset`` : Dataset, optional
        Dataset for retraining the model after each pruning step.
         Default = None.
    ``show_tqdm`` : bool, optional
        Flag to display the tqdm progress bar.
         Default = False.

    Returns
    -------
    ``model`` : FFNN
        The pruned (and optionally retrained) neural network model.
    """

    temp_pt_param = deepcopy(pt_params)

    if dataset is None and temp_pt_param.prun_param.retrain:
        raise ValueError(f'Dataset is None -> provide a dataset')
    
    if dataset is not None:
        dataloader = DataLoader(dataset, batch_size=temp_pt_param.train_param.batch_size, shuffle=temp_pt_param.train_param.shuffle)

    total_amount = temp_pt_param.prun_param.amount
    curr_amount = 0.0

    for i in tqdm_notebook(range(temp_pt_param.prun_param.prun_iterations), desc='Train iteration', disable=not show_tqdm):
        temp_amount = calc_prun_amount(i, temp_pt_param.prun_param.prun_iterations, total_amount, temp_pt_param.prun_param.pruning_schedule)
        if total_amount >= 1.0:
            temp_amount = int(np.rint(temp_amount))
        
        temp_amount = temp_amount - curr_amount

        temp_pt_param.prun_param.amount = temp_amount

        # print(total_amount, temp_pt_param.prun_param.amount)
        if temp_pt_param.prun_param.retrain:
            model = prun_and_retrain(model, temp_pt_param, dataloader)
        else:
            model = temp_pt_param.prun_param.base_pruning(model)
        curr_amount += temp_amount

    temp_pt_param.prun_param.amount = calc_actual_amount(model)
    return model


