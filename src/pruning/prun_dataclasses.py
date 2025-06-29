import numpy as np

from dataclasses import dataclass
from typing import Callable, Literal
from abc import ABC, abstractmethod

from .structured_pruning import nodes_ln_structured
from .unstructured_pruning import nodes_l1_unstructured
from ..neural_network import FFNN, Train_Param



@dataclass
class Prun_Param(ABC):
    """
    Settings for pruning a network.

    Attributes
    ----------
    ``prun_iterations`` : int
        The iterations how 
    ``amount`` : int
        The amount of the nodes that should be pruned.
    ``method`` : Callable[[FFNN], FFNN]
        A function that prunes the entire network. 
    ``use_init_params`` : bool
        A bool to determine if it uses the initial network parameters 
        after pruning for retraining.
    ``pruning_schedule`` : Literal['exponential', 'binomial', 'linear'], optional
        The pruning schedule that determines the amount pruned each pruning iteration.
        *Default = 'exponential'*
    ``param_name`` : Literal['weight', 'bias'], optional
        The parameters that should be pruned
        *Default = 'weight'*
    ``n`` : int | float | inf | -inf | 'fro' | 'nuc', optional
        The norm to determine the relevance of weights.
        *Default = 1*
    ``dim`` : int, optional
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node.
        *Default = 1*
    ``retrain`` : bool, optional
        A bool to determine if the network should be retrained after pruning.
        *Defualt = True*
    """
    prun_iterations: int
    amount: int | float
    method: Callable[[FFNN], FFNN]
    use_init_params: bool
    pruning_schedule: Literal['exponential', 'binomial', 'linear'] = 'exponential'
    param_name: Literal['weight', 'bias'] = 'weight'
    n: float = 1
    dim: int = 1
    retrain: bool = True

    @abstractmethod
    def base_pruning(self, model):
        pass



@dataclass
class Node_Prun_LTH(Prun_Param):
    """
    Settings for iterative Lottery Ticket Hypothesis (LTH) strucktured node pruning for a neural network.

    Attributes
    ----------
    ``prun_iterations`` : int
        The iterations how 
    ``amount`` : int
        The amount of the nodes that should be pruned.
    ``method`` : Callable[[FFNN], FFNN]
        A function that prunes the entire network. 
        *Default = nodes_ln_structured*
    ``use_init_params`` : bool
        A bool to determine if it uses the initial network parameters 
        after pruning for retraining.
        *Default = True*
    ``pruning_schedule`` : Literal['exponential', 'binomial', 'linear'], optional
        The pruning schedule that determines the amount pruned each pruning iteration.
        *Default = 'exponential'*
    ``param_name`` : Literal['weight', 'bias'], optional
        The parameters that should be pruned.
        *Default = 'weight'*
    ``n`` : int | float | inf | -inf | 'fro' | 'nuc', optional
        The norm to determine the relevance of weights.
        *Default = 1*
    ``dim`` : int, optional
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node.
        *Default = 1*
    ``retrain`` : bool, optional
        A bool to determine if the network should be retrained after pruning.
        *Defualt = True*
    """

    def __init__(self, prun_iterations: int, amount: int | float, n: int=1, dim: int=1):
        """
        Settings for pruning the nodes of a network with an exponentialy decreasing pruning 
        schedule for the iterations. It uses the LTH scheme, where it retrains the pruned 
        initial parameters of the network. Which parameters to prune is determined based 
        on the already trained network.

        Parameters
        ----------
        ``prun_iterations`` : int
            The iterations how 
        ``amount`` : int
            The amount of the nodes that should be pruned.
        ``n`` : int | float | inf | -inf | 'fro' | 'nuc', optional
            The norm to determine the relevance of weights.
            *Default = 1*
        ``dim`` : int, optional
            The dimension of the structured pruning. 
            0 -> all weights that enter one node. 
            1 -> all weights that exit one node.
            *Default = 1*
        """
        super().__init__(
            prun_iterations=prun_iterations,
            amount=amount,
            method=nodes_ln_structured, 
            use_init_params=True,
            pruning_schedule='exponential',
            param_name='weight',
            n=n,
            dim=dim,
            retrain=True
        )

    def base_pruning(self, model):
        return self.method(model, self.param_name, self.amount, self.n, self.dim)



@dataclass
class Local_Unstructured_Prun_LTH(Prun_Param):
    """
    Settings for iterative Lottery Ticket Hypothesis (LTH) unstructured l1 pruning for a neural network.

    Attributes
    ----------
    ``prun_iterations`` : int
        The iterations how 
    ``amount`` : int
        The amount of the nodes that should be pruned.
    ``method`` : Callable[[FFNN], FFNN]
        A function that prunes the entire network. 
        *Default = nodes_ln_structured*
    ``use_init_params`` : bool
        A bool to determine if it uses the initial network parameters 
        after pruning for retraining.
        *Default = True*
    ``pruning_schedule`` : Literal['exponential', 'binomial', 'linear'], optional
        The pruning schedule that determines the amount pruned each pruning iteration.
        *Default = 'exponential'*
    ``param_name`` : Literal['weight', 'bias'], optional
        The parameters that should be pruned.
        *Default = 'weight'*
    ``n`` : int | float | inf | -inf | 'fro' | 'nuc', optional
        The norm to determine the relevance of weights.
        *Default = 1*
    ``dim`` : int, optional
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node.
        *Default = 1*
    ``retrain`` : bool, optional
        A bool to determine if the network should be retrained after pruning.
        *Defualt = True*
    """

    def __init__(self, prun_iterations: int, amount: int | float):
        """
        Settings for pruning the nodes of a network with an exponentialy decreasing pruning 
        schedule for the iterations. It uses the LTH scheme, where it retrains the pruned 
        initial parameters of the network. Which parameters to prune is determined based 
        on the already trained network.

        Parameters
        ----------
        ``prun_iterations`` : int
            The iterations how 
        ``amount`` : int | float
            The amount of the nodes that should be pruned.
        """
        super().__init__(
            prun_iterations=prun_iterations,
            amount=amount,
            method=nodes_l1_unstructured, 
            use_init_params=True,
            pruning_schedule='exponential',
            param_name='weight',
            retrain=True
        )

    def base_pruning(self, model):
        return self.method(model, self.param_name, self.amount)



@dataclass
class Node_Prun_Finetune(Prun_Param):
    """
    Settings for iterative structured node pruning and finetuning of a neural network.

    Attributes
    ----------
    ``prun_iterations`` : int
        The iterations how 
    ``amount`` : int
        The amount of the nodes that should be pruned.
    ``method`` : Callable[[FFNN], FFNN]
        A function that prunes the entire network. 
        *Default = nodes_ln_structured*
    ``use_init_params`` : bool
        A bool to determine if it uses the initial network parameters 
        after pruning for retraining.
        *Default = False*
    ``pruning_schedule`` : Literal['exponential', 'binomial', 'linear'], optional
        The pruning schedule that determines the amount pruned each pruning iteration.
        *Default = 'exponential'*
    ``param_name`` : Literal['weight', 'bias'], optional
        The parameters that should be pruned.
        *Default = 'weight'*
    ``n`` : int | float | inf | -inf | 'fro' | 'nuc', optional
        The norm to determine the relevance of weights.
        *Default = 1*
    ``dim`` : int, optional
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node.
        *Default = 1*
    ``retrain`` : bool, optional
        A bool to determine if the network should be retrained after pruning.
        *Defualt = True*
    """

    def __init__(self, prun_iterations: int, amount: int | float, n: int=1, dim: int=1):
        """
        Settings for pruning the nodes of a network with an exponentialy decreasing pruning 
        schedule for the iterations. It uses the Finetuning scheme, where it retrains the pruned 
        network.

        Parameters
        ----------
        ``prun_iterations`` : int
            The iterations how 
        ``amount`` : int
            The amount of the nodes that should be pruned.
        ``n`` : int | float | inf | -inf | 'fro' | 'nuc', optional
            The norm to determine the relevance of weights.
            *Default = 1*
        ``dim`` : int, optional
            The dimension of the structured pruning. 
            0 -> all weights that enter one node. 
            1 -> all weights that exit one node.
            *Default = 1*
        """
        super().__init__(
            prun_iterations=prun_iterations,  # Set default iterations
            amount=amount,
            method=nodes_ln_structured, 
            use_init_params=False,
            pruning_schedule='exponential',
            param_name='weight',
            n=n,
            dim=dim,
            retrain=True
        )

    def base_pruning(self, model):
        return self.method(model, self.param_name, self.amount, self.n, self.dim)



@dataclass
class Prun_Train_Param:
    train_param: Train_Param
    prun_param: Prun_Param
    scale: dict[np.ndarray]