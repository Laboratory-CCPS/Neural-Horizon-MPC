import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

from typing import Literal

from ..neural_network import FFNN
from ..torch_utils import count_linear_modules, count_parameters, count_non_zero_parameters



def calc_actual_amount(model: nn.Module):
    """Counts the actual amount of pruned parameters"""
    return 1 - count_non_zero_parameters(model) / count_parameters(model)



def _get_new_features(model: nn.Module):
    """
    Retrieves the number of input and output features for each nn.Linear layer in the model, considering the non-zero weights and biases.

    Parameters
    ----------
    ``model`` : nn.Module
        The neural network model from which to extract the features.

    Returns
    -------
    ``features`` : list[int]
        A list of integers where each integer represents the number of non-zero input or output features of `nn.Linear` layers.
    ``feature_indeces`` : list[list[bool]]
        A list of boolean lists indicating which features are non-zero for each `nn.Linear` layer.
    """
    features = []
    feature_indeces = []
    num_linear_modules = count_linear_modules(model)
    
    first_layer= True
    counter_linear_module = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            counter_linear_module += 1
            
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy()
            if first_layer:
                first_layer = False
                not_zero = [True for _ in range(module.in_features)]
            else:
                w_col_nonzero = np.any((weight != 0), axis=0)
                not_zero = list(map(any, zip(w_row_nonzero, w_col_nonzero, b_row_nonzero)))

            b_row_nonzero = (bias != 0)
            w_row_nonzero = np.any((weight != 0), axis=1)

            features.append(_count_nonzero(not_zero))
            feature_indeces.append(not_zero)

            if counter_linear_module == num_linear_modules:
                features.append(module.out_features)
                feature_indeces.append([True for _ in range(module.out_features)])
    return features, feature_indeces


def _get_old_features(model: nn.Module):
    """
    Retrieves the number of input and output features for each nn.Linear layer in the model.

    Parameters
    ----------
    ``model`` : nn.Module
        The neural network model from which to extract the features.

    Returns
    -------
    ``features`` : list[int]
        A list of integers where each integer represents the number of input or output features of `nn.Linear` layers.
    """
    features = []
    counter_linear_module = 0
    first_layer = True

    for module in model.modules():
        if isinstance(module, nn.Linear):
            counter_linear_module += 1

            if first_layer:
                first_layer = False
                features.append(module.in_features)
            
            features.append(module.out_features)

    return features


def _count_nonzero(is_zero: list[list[bool]]):
    """
    Counts the number of non-zero elements in a list of lists where each inner list contains boolean values.

    Parameters
    ----------
    ``is_zero`` : list[list[bool]]
        A list of lists where each inner list contains boolean values representing zero or non-zero states.

    Returns
    -------
    ``count`` : int
        The number of `True` values (non-zero) in the provided list of lists.
    """
    count = 0
    for is_zero_val in is_zero:
        if is_zero_val:
            count += 1
    return count


def extract_FFNN_NNscale(model: FFNN):
    """
    Parameters
    ----------
    ``model`` : FFNN
        The FFNN model from which the normalization scale parameters will be extracted.

    Returns
    -------
    ``scale_params`` : dict
        A dictionary containing the following normalization scale parameters:
        - ``xmean``: Mean of the input features.
        - ``xstd``: Standard deviation of the input features.
        - ``ymean``: Mean of the output features.
        - ``ystd``: Standard deviation of the output features.
    """
    return {
        'xmean': model.xmean.view(-1, 1).detach().cpu().numpy(),
        'xstd': model.xstd.view(-1, 1).detach().cpu().numpy(),
        'ymean': model.ymean.view(-1, 1).detach().cpu().numpy(),
        'ystd': model.ystd.view(-1, 1).detach().cpu().numpy(),
    }



def remove_nodes(model: FFNN):
    """
    Removes inactive nodes (=0) from a feedforward neural network (FFNN) model and 
    creates a new model with the updated architecture.

    Parameters
    ----------
    ``model`` : FFNN
        The original FFNN model from which inactive nodes will be removed.

    Returns
    -------
    ``new_model`` : FFNN
        The FFNN model with the inactive nodes removed.
    """
    old_features = _get_old_features(model)
    new_features, feature_indeces = _get_new_features(model)
    if new_features == old_features:
        return model

    NNscale = extract_FFNN_NNscale(model)
    new_model = FFNN(NNscale, model.activation, model.n_layers, new_features[1:-1], model.device, model.dtype)

    idx = 0
    
    for old_module, new_module in zip(model.modules(), new_model.modules()):
        if isinstance(old_module, nn.Linear) and isinstance(new_module, nn.Linear):
            out_indices = np.array([feature_indeces[idx+1]], dtype=bool)
            in_indices = np.array([feature_indeces[idx]], dtype=bool)
            idx += 1

            indices = torch.from_numpy(in_indices & out_indices.T)
            with torch.no_grad():
                new_module.weight = nn.parameter.Parameter(
                    torch.reshape(
                        old_module.weight[indices], (new_module.out_features, new_module.in_features)
                        )
                    )
                new_module.bias = nn.Parameter(old_module.bias[out_indices.flatten()])
    return new_model



def remove_pruning_reparamitrizations(
    model: nn.Module, 
    make_sparse: bool = False
):
    """
    Removes pruning reparameterizations from a model, applying the masks to the weights and biases.

    Parameters
    ----------
    ``model`` : nn.Module
        The pruned model from which the pruning reparameterizations will be removed.
    ``make_sparse`` : bool, optional
        Indicates whether the weights and biases should be converted to a sparse format.
        *Default = False*

    Returns
    -------
    ``model`` : nn.Module
        The model with the pruning masks applied to the weights and biases.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
                if make_sparse:
                    module.weight.to_sparse()
            except:
                pass
            try:
                prune.remove(module, "bias")
                if make_sparse:
                    module.bias.to_sparse()
            except:
                pass
    else:
        return model


    
def calc_prun_amount(iter: int, iterations: int, amount, schedule: Literal['exponential', 'binomial', 'linear']):
    """
    Calculates the pruning amount based on the iteration, total iterations, and the specified pruning schedule.

    Parameters
    ----------
    ``iter`` : int
        The current pruning iteration.
    ``iterations`` : int
        The total number of pruning iterations.
    ``amount`` : float
        The total amount of pruning to be applied by the end of all iterations.
    ``schedule`` : Literal['exponential', 'binomial', 'linear']
        The schedule type that defines how the pruning amount is distributed across iterations.

    Returns
    -------
    ``pruning_amount`` : float
        The calculated pruning amount for the current iteration.
    """
    match schedule:
        case 'exponential':
            multipl = 1 / (1-np.exp(-1))
            return amount*multipl*(1-np.exp(- (iter + 1) / iterations))
        case 'binomial':
            return np.abs(1 - (1-amount)**((iter + 1) / iterations))
        case 'linear':
            return amount*((iter + 1) / iterations)
        case _:
            raise ValueError(f'schedule string "{schedule}" is not known/implemented! Need to be ')



def restore_prun_mask(pruned_model: FFNN, new_model: FFNN):
    """
    Restores the pruning mask from a pruned model to a new model by applying the same masks.

    Parameters
    ----------
    ``pruned_model`` : FFNN
        The model with pruning masks applied.
    ``new_model`` : FFNN
        The new model where the pruning masks will be restored.

    Returns
    -------
    ``new_model`` : FFNN
        The new model with the restored pruning masks.
    """
    for (_, pruned_module), (_, new_module) in zip(pruned_model.named_modules(), new_model.named_modules()):
        if isinstance(pruned_module, nn.Linear) and isinstance(new_module, nn.Linear):
            # weights
            if hasattr(pruned_module, 'weight_mask'):
                weight_mask = pruned_module.weight_mask.data.detach().clone()
                prune.custom_from_mask(new_module, name='weight', mask=weight_mask)

            # bias
            if hasattr(pruned_module, 'bias_mask'):
                bias_mask = pruned_module.bias_mask.data.detach().clone()
                prune.custom_from_mask(new_module, name='bias', mask=bias_mask)

        elif (isinstance(pruned_module, nn.Linear) and not isinstance(new_module, nn.Linear)) \
              or (not isinstance(pruned_module, nn.Linear) and isinstance(new_module, nn.Linear)):
            raise ModuleException(f'Modules are not the same!\n{pruned_module}\n{new_module}')
    return new_model



class ModuleException(Exception):
    """
    Exception raised for errors related to PyTorch modules.
    """
    def __init__(self, message='Error with PyTorch module'):
        self.message = message
        super().__init__(self.message)