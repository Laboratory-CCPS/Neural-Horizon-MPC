import numpy as np
import json
import os
import warnings
import time

from dataclasses import dataclass, field, fields
from typing import Literal



@dataclass
class MPC_Param:
    """
    Data class to hold the parameters for a model predictive control (MPC) problem with a pendulum on a cart.

    Attributes
    ----------
    ``Q``: np.ndarray, optional
        State stage cost, default is np.diag([1, 1, 1e-5, 1e-5]).
    ``R``: np.ndarray, optional
        Input stage cost, default is np.diag([1e-5]).
    ``ubnd``: float, optional
        Bound on absolute value of input, default is 80.
    ``xbnd``: np.ndarray, optional
        Bound on absolute value of state, default is np.array([[2],[6*np.pi],[10],[10]]).
    ``N``: int, optional
        Length of the overall horizon (MPC horizon + Neural horizon), default is 30.
    ``N_MPC``: int, optional
        Length of the MPC horizon, default is 8.
    ``xinit``: np.ndarray, optional
        Starting point for the closed-loop trajectory, default is np.array([[0],[np.pi],[0],[0]]).
    ``Ts``: float, optional
        Sampling rate, default is 0.02.
    ``T_sim``: float, optional
        Simulation horizon (in seconds), default is 3.
    ``xlabel``: List[str]
        List of state names
    ``ulabel``: str
        Input name

    ### NN configuration parameters
    ``get_states``: bool, optional
        Whether to estimate the state with the neural network, default is True.
    ``get_state_bounds``: bool, optional
        Whether to bounds the states estimated by the neural network, default is True.
    ``get_Jx``: bool, optional
        Whether to estimate the state stage costs with the neural network, default is False.
    ``get_Ju``: bool, optional
        Whether to estimate the input stage costs with the neural network, default is False.

    ### Model parameters
    ``nx``: int, optional
        Number of states, default is 4.
    ``nu``: int, optional
        Number of inputs, default is 1.
    ``M``: float, optional
        Cart weight in kilograms, default is 1.
    ``m``: float, optional
        Pendulum weight in kilograms, default is 0.1.
    ``g``: float, optional
        Gravity constant in m/s^2, default is 9.81.
    ``l``: float, optional
        Pendulum length in meters, default is 0.8.

    Notes
    -----
    The `Q_NN` attribute is set to the same value as `Q` for simplicity.
    The `P` attribute is set to the same value as `Q` for simplicity.
    The `N_NN` attribute is set to `N - N_MPC`.
    The `N_sim` attribute is set to `T_sim / Ts`.
    """
    Q:     np.ndarray = field(default_factory=lambda: np.diag([1, 1, 1e-5, 1e-5]))          # state stage cost xT*Q*x
    R:     np.ndarray = field(default_factory=lambda: np.diag([1e-5]))                      # input stage cost uT*R*u
    ubnd:  float      = 80                                                                  # bound on abs(u)
    xbnd:  np.ndarray = field(default_factory=lambda: np.array([[2],[6*np.pi],[10],[10]]))  # bound on abs(x)
    N:     int        = 30                                                                  # length of the overall horizon (MPC horizon + Neural horizon)
    N_MPC: int        = 8                                                                   # length of MPC horizon
    xinit: np.ndarray = field(default_factory=lambda: np.array([[0],[np.pi],[0],[0]]))      # starting point for the closed-loop trajectory
    Ts:    float      = 0.02                                                                # sampling rate
    T_sim: float      = 3                                                                   # simulation horizon (seconds)
    
    # NN configuration parameters
    get_states:       bool = True
    get_state_bounds: bool = True
    get_Jx:           bool = False
    get_Ju:           bool = False
    
    # Model parameters
    nx: int  = 4    # number of states
    nu: int  = 1    # number of inputs
    M: float = 1    # cart weight [kg]
    m: float = 0.1  # pendulum weight [kg]
    g: float = 9.81 # gravity constant [m/s^2]
    l: float = 0.8  # pendulum length [m]
    xlabel: list      = field(default_factory=lambda: ['px', 'theta','v','omega']) # list of state names
    ulabel: str       = 'u'                         # input name
    Q_NN:  np.ndarray = field(init=False) # Q for Neural horizon, set same as Q for simplicity
    P:     np.ndarray = field(init=False) # Terminal constraint, set same as Q for simplicity
    N_NN:  int        = field(init=False) # length of the Neural horizon = N - N_MPC
    N_sim: int        = field(init=False) # number of simulation steps   = T_sim/Ts
    
    def __post_init__(self) -> None:
        # self.xlabel = ['px', 'theta','v','omega']
        self.Q_NN = self.Q
        self.P = self.Q
        self.N_NN = self.N - self.N_MPC
        self.N_sim = int(self.T_sim/self.Ts)

    def __eq__(self, other):
        if isinstance(other, MPC_Param):
            equal=True
            for field in fields(self):
                self_val,other_val = getattr(self, field.name),getattr(other, field.name) 
                if isinstance(self_val, np.ndarray):
                    equal = equal and np.array_equal(self_val, other_val)
                else:
                    equal = equal and (self_val == other_val)
            return equal
        return False

    def diff(self,other):
        diff = []
        if isinstance(other, MPC_Param):
            for field in fields(self):
                self_val,other_val = getattr(self, field.name),getattr(other, field.name) 
                if isinstance(self_val, np.ndarray):
                    if not np.array_equal(self_val, other_val):
                        diff +=[field.name]
                elif (self_val != other_val):
                    diff +=[field.name]
        return diff
    
    def save(self,filename=None,filedir='Results//Parameters'):
        pjson = {}
        for f in fields(self):
            v = getattr(self, f.name)
            pjson[f.name]=v.tolist() if type(v) is np.ndarray else v

        pjson_object = json.dumps(pjson, indent=4)
        if filename is None:
            filename=f'MPC_Param_{time.strftime("%d_%b_%Y_%H_%M")}.json'
        os.makedirs(filedir,exist_ok=True)
        with open(os.path.join(filedir,filename),'w') as file:
            file.write(pjson_object)

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            warnings.warn(f'File {filepath} does not exist!',UserWarning)
            return None
        
        with open(filepath,'r') as file:
            pjson = json.load(file)

        new_param = cls()
        for f in fields(new_param):
            if f.name in pjson and f.type is np.ndarray:
                setattr(new_param, f.name, np.array(pjson[f.name]))
            elif f.name in pjson:
                setattr(new_param, f.name, pjson[f.name])
            else:
                warnings.warn(
                    f'\nKey: {f.name} not in loaded params. Loaded param class may be different from in code used params class.', 
                    UserWarning
                )

        return new_param
    

@dataclass
class AMPC_Param(MPC_Param):
    """
    Data class to hold the parameters for a model predictive control (MPC) problem with a pendulum on a cart.

    Attributes
    ----------
    ``Q``: np.ndarray, optional
        State stage cost, default is np.diag([1, 1, 1e-5, 1e-5]).
    ``R``: np.ndarray, optional
        Input stage cost, default is np.diag([1e-5]).
    ``ubnd``: float, optional
        Bound on absolute value of input, default is 80.
    ``xbnd``: np.ndarray, optional
        Bound on absolute value of state, default is np.array([[2],[6*np.pi],[10],[10]]).
    ``N``: int, optional
        Length of the overall horizon (MPC horizon + Neural horizon), default is 30.
    ``N_MPC``: int, optional
        Length of the MPC horizon, default is 8.
    ``xinit``: np.ndarray, optional
        Starting point for the closed-loop trajectory, default is np.array([[0],[np.pi],[0],[0]]).
    ``Ts``: float, optional
        Sampling rate, default is 0.02.
    ``T_sim``: float, optional
        Simulation horizon (in seconds), default is 3.
    ``xlabel``: List[str]
        List of state names
    ``ulabel``: str
        Input name

    ### NN configuration parameters
    ``get_states``: bool, optional
        Whether to estimate the state with the neural network, default is True.
    ``get_state_bounds``: bool, optional
        Whether to bounds the states estimated by the neural network, default is True.
    ``get_Jx``: bool, optional
        Whether to estimate the state stage costs with the neural network, default is False.
    ``get_Ju``: bool, optional
        Whether to estimate the input stage costs with the neural network, default is False.

    ### Model parameters
    ``nx``: int, optional
        Number of states, default is 4.
    ``nu``: int, optional
        Number of inputs, default is 1.
    ``M``: float, optional
        Cart weight in kilograms, default is 1.
    ``m``: float, optional
        Pendulum weight in kilograms, default is 0.1.
    ``g``: float, optional
        Gravity constant in m/s^2, default is 9.81.
    ``l``: float, optional
        Pendulum length in meters, default is 0.8.

    ### extra parameters
    ``version`` : int, default = 0
        Version of the same AMPC setting.
    ``param_name`` : str
        Parameter name for savings '{N_MPC}M_{N_NN}N_{version}v'
    
    Notes
    -----
    The `Q_NN` attribute is set to the same value as `Q` for simplicity.
    The `P` attribute is set to the same value as `Q` for simplicity.
    The `N_NN` attribute is set to `N - N_MPC`.
    The `N_sim` attribute is set to `T_sim / Ts`.
    """
    version : int = 0
    param_name : str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.param_name = f'{self.N_MPC}M_{self.N_NN}N_{self.version}v'
    
    def __eq__(self, other):
        super().__eq__(other)


@dataclass
class NH_AMPC_Param(MPC_Param):
    """
    Data class to hold the parameters for a model predictive control (MPC) problem with a pendulum on a cart.

    MPC Parameters
    --------------
    ``Q``: np.ndarray, optional,
        State stage cost, default is np.diag([1, 1, 1e-5, 1e-5]).
    ``R``: np.ndarray, optional,
        Input stage cost, default is np.diag([1e-5]).
    ``ubnd``: float, optional,
        Bound on absolute value of input, default is 80.
    ``xbnd``: np.ndarray, optional,
        Bound on absolute value of state, default is np.array([[2],[6*np.pi],[10],[10]]).
    ``N``: int, optional,
        Length of the overall horizon (MPC horizon + Neural horizon), default is 30.
    ``N_MPC``: int, optional,
        Length of the MPC horizon, default is 8.
    ``xinit``: np.ndarray, optional,
        Starting point for the closed-loop trajectory, default is np.array([[0],[np.pi],[0],[0]]).
    ``Ts``: float, optional,
        Sampling rate, default is 0.02.
    ``T_sim``: float, optional,
        Simulation horizon (in seconds), default is 3.
    ``xlabel``: List[str],
        List of state names
    ``ulabel``: str
        Input name

    NN configuration Parameters
    ---------------------------
    ``get_states``: bool, optional,
        Whether to estimate the state with the neural network, default is True.
    ``get_state_bounds``: bool, optional,
        Whether to bounds the states estimated by the neural network, default is True.
    ``get_Jx``: bool, optional,
        Whether to estimate the state stage costs with the neural network, default is False.
    ``get_Ju``: bool, optional,
        Whether to estimate the input stage costs with the neural network, default is False.

    Model Parameters
    ----------------
    ``nx``: int, optional,
        Number of states, default is 4.
    ``nu``: int, optional,
        Number of inputs, default is 1.
    ``M``: float, optional,
        Cart weight in kilograms, default is 1.
    ``m``: float, optional,
        Pendulum weight in kilograms, default is 0.1.
    ``g``: float, optional,
        Gravity constant in m/s^2, default is 9.81.
    ``l``: float, optional,
        Pendulum length in meters, default is 0.8.

    Dataset Parameters
    ------------------
    ``N_DS`` : int, default = 0,
        Horizon of the dataset.
    ``TRAIN_V_DS`` : int, optional,
        Dataset version for training the NN. 
    ``TEST_V_DS`` : int, optional,
        Dataset version for NN testing. 
    ``DS_begin`` : Literal['begin', 'fixed', ''], default = '',
        Which dataset index as input of the network. 
         'begin' - always use the first index as the input to the NN. 
          'fixed' - always use the 'DS_feature' index as input to the NN.
           '' - uses the horizon P.N_MPC where the neural horizon starts.
    ``DS_feature`` : int, default = 8,
        Feature where to always start. Only used if 'DS_begin'='fixed'.
    ``DS_samples`` : int, default = 0,
        Number of trajectory samples in the dataset.  
    ``DS_opts_name`` : str, default = '',
        Dataset acados solver name for the postinit dataset name. 
    
    NN Parameters
    -------------
    ``V_NN`` : int, optional,
        Version of the NN.
    ``N_hidden`` : int, optional,
        Hidden neurons of the NN.
    ``N_hidden_end`` : int, optional,
        Hidden neurons of the NN after pruning.

    Name Strings
    ------------
    ``param_name`` : str,
        Parameter name for savings.
    ``train_DS_name`` : str,
        Dataset file name used for NN training.
    ``test_DS_name`` : str,
        Dataset file name used for NN testing.
    ``NN_name`` : str,
        NN file name.
    ``Pruned_NN_name`` : str,
        Pruned NN file name.

    Notes
    -----
    The `Q_NN` attribute is set to the same value as `Q` for simplicity.

    The `P` attribute is set to the same value as `Q` for simplicity.

    The `N_NN` attribute is set to `N - N_MPC`.

    The `N_sim` attribute is set to `T_sim / Ts`.
    """
    # Dataset options
    N_DS : int = 0
    TRAIN_V_DS : int = None
    TEST_V_DS : int = None
    DS_begin : Literal['begin', 'fixed', ''] = ''
    DS_feature : int = 8
    DS_samples : int = 0
    DS_opts_name : str = ''
    
    # NN options
    V_NN : int = None
    N_hidden : int = None
    N_hidden_end : int = None

    # names
    param_name : str = field(init=False)
    train_DS_name : str = field(init=False)
    test_DS_name : str = field(init=False)
    NN_name : str = field(init=False)
    Pruned_NN_name : str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._set_new_param_name()

        # train dataset name
        self.train_DS_name = get_dataset_name(self.N_DS, self.DS_samples, self.DS_opts_name, self.TRAIN_V_DS)

        # test dataset name
        self.test_DS_name = get_dataset_name(self.N_DS, self.DS_samples, self.DS_opts_name, self.TEST_V_DS)

        # network name
        self._set_new_NN_name()

        # pruned network name
        if self.N_hidden_end == self.N_hidden:
            raise ValueError(
                'Something is wrong with the N_hidden_end or N_hidden' + \
                f' -> {self.N_hidden} - {self.N_hidden_end}'
            )
        self._set_new_pruned_NN_name()


    def __eq__(self, other):
        super().__eq__(other)


    def _set_new_NN_name(self):
        self.NN_name = get_network_name(
            f'{self.DS_feature}M_{self.N_NN}N_{self.N_DS}ND_{self.TRAIN_V_DS}VD' if 'fixed' == self.DS_begin else self.param_name, 
            self.DS_samples, 
            self.DS_opts_name, 
            self.N_hidden, 
            self.V_NN
        )

    def _set_new_param_name(self):
        ds_begin = '' if '' == self.DS_begin else f'_{self.DS_begin}'
        ds_features =  f'_{self.DS_feature}M' if 'fixed' == self.DS_begin else ''
        self.param_name = f'{self.N_MPC}M_{self.N_NN}N_{self.N_DS}ND_{self.TRAIN_V_DS}VD{ds_begin}{ds_features}'

    
    def _set_new_pruned_NN_name(self):
        if self.N_hidden_end is not None:
            self.Pruned_NN_name = create_pruned_nn_name(self.NN_name, self.N_hidden_end)
        else:
            self.Pruned_NN_name = None


    def change_N_hidden(self, N_hidden: int):
        self.N_hidden = N_hidden
        if self.N_hidden_end == self.N_hidden:
            warnings.warn(
                'param N_hidden_end set to -> None because N_hidden is set to the same as N_hidden_end was before.', 
                UserWarning
            )
            self.N_hidden_end = None
        
        self._set_new_NN_name()
        self._set_new_pruned_NN_name()




def get_dataset_name(
    dataset_horizon: int,
    num_samples: int,
    acados_name: str,
    dataset_version: int | None,
):
    """
    Creates a standard dataset name out of the given parameters. 
    """
    _v_ds = '' if dataset_version is None else f'_{dataset_version}v'
    return f'MPC_data_{dataset_horizon}steps_{num_samples}datapoints_{acados_name}{_v_ds}.csv'


def get_network_name(
        param_name: str,
        dataset_samples: int,
        acados_name: str,
        n_hidden: int | None, 
        network_version: int | None
    ):
    """
    Creates a standard neural network name out of the given parameters. 
    """
    _v_nn = '' if network_version is None else f'_{network_version}v'
    _n_hidden = '' if n_hidden is None else f'_{n_hidden}Nhid'
    return f'NN_acados_{param_name}_{dataset_samples}Dp_{acados_name}{_n_hidden}{_v_nn}.ph'


def create_pruned_nn_name(NN_name: str, end_hidden_size: int):
    """
    Creates a standard pruned neural network name out of the given parameters. 
    """
    setp_strs = NN_name.split('.')

    if len(setp_strs) > 2:
        raise ValueError(f'NN_name has more than one \".\" inside! {NN_name}')
    
    return f'{setp_strs[0]}_prun_{end_hidden_size}Nhid.{setp_strs[1]}'