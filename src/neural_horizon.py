import torch,os,warnings
import casadi as cs
import numpy as np
import pandas as pd

from torch import nn
from tqdm.notebook import trange
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from .parameters import MPC_Param, NH_AMPC_Param
from .neural_network import FFNN, train_NN, Train_Param

from .pruning.prun_dataclasses import Prun_Param, Prun_Train_Param
from .pruning.iter_pruning import iter_prun_nodes
from .pruning.prun_utils import remove_nodes, remove_pruning_reparamitrizations




class NN_for_casadi():
    """
    A class for building and training a neural network for use with CasADi.

    Attributes
    ----------
    P : MPC_Param dataclass object
        Parameters of the model and the Neural Network
    device : torch.device
        The device on which to run the model.
    dtype : torch.dtype
        The data type of the model.
    labels : list of str
        The names of the columns in the data containing the labels.
    features : list of str
        The names of the columns in the data containing the features.
    states : list of str
        The names of the states of the model.
    inputs : list of str
        The names of the columns in the data containing the inputs.
    scale : dict
        A dictionary containing the means and standard deviations of the features and labels.
        The keys are 'xmean', 'xstd', 'ymean', and 'ystd'.
    trainset : torch.utils.data.TensorDataset
        The dataset used to train the model.
    NN : FFNN
        The neural network model.
    NN_casadi : casadi.Function
        The neural network model converted to a CasADi function.
    """
    def __init__(self, df_file, P=MPC_Param(), features=None, labels=None, device=torch.device('cpu'), dtype=torch.float32):
        """
        Constructor for the `NN_for_casadi` class.

        Parameters
        ----------
        df_file : str
            File path of the csv file containing the dataset.
        P : MPC_Param, optional
            An instance of the `MPC_Param` class containing parameters for the MPC problem, by default MPC_Param().
        features : list of str, optional
            A list of strings containing the names of the columns to be used as features in the neural network, constructed by default from the states.
        labels : list of str, optional
            A list of strings containing the names of the columns to be used as labels in the neural network, constructed by default from the states.
        states : list of str, optional
            A list of strings containing the names of the columns that represent the states of the system, by default ['px', 'theta','v','omega'].
        inputs : list of str, optional
            A list of strings containing the names of the columns that represent the inputs to the system, by default ['u'].
        device : torch.device, optional
            The device on which the neural network will be trained, by default torch.device('cpu').
        dtype : torch.dtype, optional
            The data type for the neural network, by default torch.float32.
        """        
        self.P = P
        self.df_file = df_file
        
        # components of the NN
        self.NN = None
        self.scale = None
        self.NN_casadi = None
        self.trainset = None
        self.train_param = None
        self.prun_param = None
        self.device = torch.device(device)
        if self.device.type=='cpu' and dtype!=torch.float32:
            self.dtype=torch.float32
            warnings.warn(f'Only float32 precision supported for device type cpu, provided dtype={dtype} ignored.',UserWarning)
        else:
            self.dtype = dtype
        self.features = features
        self.labels = labels
    


    def gen_training_data(self,do_scaling=True):
        """
        Generate training data for the neural network and scaling parameters.

        Parameters
        ----------
        do_scaling : bool, optional
            Whether to perform scaling on the data, by default True.

        Returns
        -------
        torch.utils.data.TensorDataset
            A tensor dataset containing the training data.
        """
        
        if self.features is None:
            # by default features are states at N_MPC time point
            self.features = [f'{x}_p{self.P.N_MPC}' for x in self.P.xlabel]

        if self.labels is None:
            self.labels = []
            if self.P.get_states:
                # by default state labels span the time frame from [N_MPC+1, N+1]
                self.labels += [f'{x}_p{i}'  for i in range(self.P.N_MPC+1,self.P.N+1) for x in self.P.xlabel]
            if self.P.get_Jx:
                self.labels +=['Jx']
            if self.P.get_Ju:
                self.labels +=['Ju']
                
        if self.P.get_Jx or self.P.get_Ju:
            stage_ind = sorted(set([int(x.split('_')[1][1:]) for x in self.labels if '_p' in x]))
            # For the case when only Ju/Jx are expected
            if len(stage_ind)==0:
                stage_ind = sorted(set(range(self.P.N_MPC+1,self.P.N+1)))
            df = self.calculate_cost_functions(stage_ind=stage_ind)
        else:
            df = pd.read_csv(self.df_file)

        tx, ty, self.scale = get_traindata_and_scaling(df, self.features, self.labels, do_scaling)
        return torch.utils.data.TensorDataset(torch.tensor(tx,device=self.device,dtype=self.dtype), torch.tensor(ty,device=self.device,dtype=self.dtype))



    def calculate_cost_functions(self,df:pd.DataFrame=None,stage_ind:list=None) -> pd.DataFrame:
        """
        Compute the stage costs for the given states and inputs, based on the provided cost matrices.

        Args:
            df (pandas.DataFrame): A DataFrame containing the system states and inputs, with column names
                corresponding to the state and input names followed by the timestep index (e.g., 'px_p0', 'theta_p1', 'u_p3').
            stage_ind (list of int): A list of time indices for which to calculate the stage cost. Default is range(N_MPC+1,N+1).

        Returns:
            pandas.DataFrame: A DataFrame containing the original states and inputs, as well as two new columns:
                'Jx': the stage cost for the states at each timestep, and
                'Ju': the stage cost for the inputs at each timestep.
        """
        
        if df is None:
            df = pd.read_csv(self.df_file)
        
        # state cost
        P = self.P
        Q,R,PP,N_MPC,N = P.Q_NN,P.R,P.P,P.N_MPC,P.N
        states,inputs = P.xlabel,P.ulabel
        if stage_ind is None:
            stage_ind = list(range(N_MPC+1,N+1))

        Jx = pd.Series(0,index=df.index)
        for i in stage_ind[:-1]: # for states stage costs go from 1...N ( 1...N-1 w/ Q, and at N w/ P)
            Jx+=NN_for_casadi.get_stage_cost(df[[f'{x}_p{i}' for x in states]],Q)
        Jx+=NN_for_casadi.get_stage_cost(df[[f'{x}_p{stage_ind[-1]}' for x in states]],PP)
        df['Jx']=Jx

        # input cost
        Ju = pd.Series(0,index=df.index)
        for i in stage_ind[:-1]:
            Ju+=NN_for_casadi.get_stage_cost(df[[f'{u}_p{i}' for u in inputs]],R)
        df['Ju']=Ju

        return df
    


    def NNprun(self, prun_params: Prun_Param, show_tqdm: bool = False):
        """
        Prunes the pytorch network from the given pruning settings and 
        stores it in ``self.NN``.
        
        Parameters
        ----------
        ``prun_params`` : Prun_Param
            A dataclass that contains the pruning settings.

        ``show_tqdm`` : bool, optional
            A bool determining if the retraining process after pruning should be shown. 
             Default = False
        """
        if self.trainset is None:
            self.trainset = self.gen_training_data()

        self.prun_param = prun_params

        if self.train_param is None:
            raise ValueError(f'Class {self.__class__.__name__} has no train_param!')
        
        pt_param = Prun_Train_Param(self.train_param, self.prun_param, self.scale)
        pruned_model = iter_prun_nodes(self.NN, pt_param, self.trainset, show_tqdm=show_tqdm)
        self.NN = pruned_model



    def remove_nodes(self):
        """
        Sets the pruned parameters to zero, remove the prune mask and then remove the 
        parameters or nodes of the pytorch model. After that it stores it in self.NN.
        
        !!Attention!! 
            Only use after NNprun, otherwise no effect.
        """
        self.NN = remove_pruning_reparamitrizations(self.NN)
        self.NN = remove_nodes(self.NN)



    def NNprunCasadi(self, prun_params: Prun_Param, show_tqdm: bool = False):
        """
        Prunes the pytorch network from the given pruning settings. 
        Sets the pruned parameters to zero, remove the prune mask and 
        then remove the parameters or nodes of the pytorch model. 
        After that it calls the function ``transform_pytorch_to_casadi``

        Parameters
        ----------
        ``prun_params`` : Prun_Param
            A dataclass that contains the pruning settings.

        ``show_tqdm`` : bool, optional
            A bool determining if the retraining process after pruning should be shown. 
             Default = False
        """
        self.NNprun(prun_params, show_tqdm)
        self.remove_nodes()
        self.transform_pytorch_to_casadi()



    @staticmethod
    def get_stage_cost(x,Q):
        Qdf = pd.DataFrame(Q,columns = x.columns,index =x.columns)
        return (x @ Qdf).mul(x).sum(axis='columns')
    


    def evaluate_NN(self,df_file=None):
        """
        Evaluate the neural network's accracy using R2-score.

        Parameters
        ----------
        df_file : str, optional
            The file path of the csv file containing the dataset to evaluate the neural network on, by default uses train set.
        """
        if df_file is None:
            train_x,train_y = self.trainset.tensors
        else:
            NN_val = NN_for_casadi(df_file,self.P,device=self.device,dtype=self.dtype,labels=self.labels,features=self.features)
            train_x,train_y = NN_val.gen_training_data().tensors

        y_scaled = (train_y.float().cpu().numpy()-self.scale['ymean'])/self.scale['ystd']
        with torch.no_grad():
            est_y = self.NN(train_x).float().cpu().numpy()

        r2s = r2_score(y_scaled,est_y)
        rel_err = np.abs((y_scaled-est_y)/self.scale['ystd'])

        print(f'''NN evaluation:
NN: [{self.NN.nin}]->[{self.NN.nout}], {self.NN.n_layers} layer(s), {self.NN.n_neurons} neuron(s) per layer
R2-score: {r2s:0.4f}
Relative error: {100*rel_err.mean():0.2f}% mean, {100*rel_err.std():0.2f}% standard deviation''')
        return r2s, rel_err
    


    def transform_pytorch_to_casadi(self,name_input='featues',name_output='predictions'):
        """
        Transform the PyTorch neural network to a CasADi function.

        Parameters
        ----------
        name_input : str, optional
            The name of the input variable for the CasADi function, by default 'features'.
        name_output : str, optional
            The name of the output variable for the CasADi function, by default 'predictions'.

        Returns
        -------
        casadi.Function
            A CasADi function representing the neural network.
        """
        ## Counter for layers and dicts for storage of weights, biases, and activation
        layer_counter,net_weights,net_biases,net_activations = 1,{},{},{}   
        ## Get bias and weights in order of layers
        for p in self.NN.named_parameters():
            if 'weight' in p[0]:
                net_weights[str(layer_counter)] = p[1].float().cpu().detach().numpy()
            if 'bias' in p[0]:
                net_biases[str(layer_counter)] = p[1].float().cpu().detach().numpy()
                layer_counter += 1
                
        ## Define common activation functions
        def apply_act_fun(act,x):
            if act == 'relu':
                return cs.fmax(cs.SX.zeros(x.shape[0]),x)
            elif act == 'sigmoid':
                return 1/(1+cs.exp(-x))
            elif act == 'tanh':
                return cs.tanh(x)
            else:
                raise ValueError(f'Unknown activation function! Supported activations: [relu,sigmoid,tanh]; received: {act}.')

        ## Reconstruct network with activation
        # first layer is special since input shape is defined by feature size
        input_var = cs.SX.sym('input', net_weights['1'].shape[1])
        scaled_input_var = (input_var - self.NN.xmean.float().cpu().numpy().T)/self.NN.xstd.float().cpu().numpy().T # apply scaling from the model
        output_var = cs.mtimes(net_weights['1'], scaled_input_var) + net_biases['1']
        
        output_var = apply_act_fun(self.NN.activation,output_var)
        # loop over layers and apply activation except for last one
        for l in range(2, layer_counter):
            output_var = cs.mtimes(net_weights[str(l)], output_var) + net_biases[str(l)]
            if l < layer_counter - 1:
                output_var = apply_act_fun(self.NN.activation,output_var)
        
        # unscale the outputs
        output_var = output_var*self.NN.ystd.float().cpu().numpy().T + self.NN.ymean.float().cpu().numpy().T
        
        self.NN_casadi = cs.Function('nn_casadi_function', [input_var], [output_var], [name_input],[name_output])
        


    def NNcompile(self,batch_size=512,shuffle=True,activation='tanh',n_layers=3,n_neurons=32,n_epochs=500,noise=0,lr=1e-3,weight_decay=0,show_tqdm=True):
        """Compiles the data for training, runs it, and converts the result to CasADi.
        
        Parameters
        ----------
        batch_size : int, optional
            The batch size for training the neural network (default is 512).
        shuffle : bool, optional
            Whether to randomize the order of the training data (default is True).
        activation : str, optional
            The activation function to use for the neural network. Must be one of 'tanh', 'relu', or 'sigmoid' (default is 'tanh').
        n_layers : int, optional
            The number of hidden layers for the neural network (default is 3).
        n_neurons : int or list, optional
            The number of neurons in each hidden layer. If an integer is provided, the same number of neurons will be used for each hidden layer (default is 32).
        n_epochs : int, optional
            The number of epochs to train the neural network (default is 500).
        noise : float, optional
            The amount of relative noise to add to the training data (default is 0).
        lr : float, optional
            The learning rate for training the neural network (default is 1e-3).
        weight_decay : float, optional
            The weight decay for training the neural network (default is 0).
        show_tqdm : bool, optional
            Whether to show the progress bar during training (default is True).
        """
        if self.train_param is not None:
            warnings.warn('Train parameters reset by given NNcompile parameters', UserWarning)
        
        self.train_param = Train_Param(batch_size, shuffle, n_epochs, noise, lr, weight_decay, nn.MSELoss)
        self.trainset = self.gen_training_data()
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.train_param.batch_size, shuffle=self.train_param.shuffle)
        self.NN = FFNN(self.scale,activation=activation,n_layers=n_layers,n_neurons=n_neurons,device=self.device,dtype=self.dtype)
        self.NN = train_NN(self.NN, 
                           trainloader = trainloader, 
                           criterion = self.train_param.criterion(), 
                           n_epochs = self.train_param.n_epochs, 
                           noise = self.train_param.noise, 
                           lr = self.train_param.lr, 
                           weight_decay = self.train_param.weight_decay, 
                           show_tqdm = show_tqdm)
        self.transform_pytorch_to_casadi()
        


    def NNsave(self, file = None, filedir: os.PathLike = 'Results//Trained_Networks'):
        """Saves the current neural network to file.
        
        Parameters
        ----------
        file : str, optional
            The name of the file to save the neural network to. If not provided, the filename will be generated based on the neural network
            architecture and other parameters (default is None).
        filedir : str, optional
            The directory to save the file to (default is 'Results//Trained_Networks').
            
        Returns
        -------
        str or None
            The filepath of the saved neural network, or None if the neural network has not been compiled yet.
        """
        
        if self.NN is None:
            # NN wasn't initialized yet
            warnings.warn(f'Neural Network not initialized yet. Call NNcompile method to generate and train the Neural Network first.',UserWarning)
            return None
        
        # construct the file path
        if file is None:
            file = f'NN_{self.NN.nin}_to_{self.NN.nout}_{self.NN.activation}_{self.NN.n_layers}x{self.NN.n_neurons[1:]}_{self.NN.device.type}_{str(self.NN.dtype).split(".")[-1]}.ph'
        os.makedirs(filedir,exist_ok=True)
        filepath = os.path.join(os.path.abspath(filedir),file)    
        
        torch.save({
            'NNscale': self.scale,
            'features': self.features,
            'labels': self.labels,
            'MPC_Param': self.P,
            'df_file':self.df_file,
            'Train_Param': self.train_param, 
            'Prun_Param': self.prun_param,
            'activation': self.NN.activation,
            'n_layers': self.NN.n_layers,
            'n_neurons': self.NN.n_neurons[1:],
            'model_state_dict': self.NN.state_dict(),
            'inital_model_state_dict': self.NN.init_state_dict,
            }, filepath)
        return filepath
        


    @staticmethod
    def NNload(filepath,**kwargs):
        """Loads a trained neural network from file.
        
        Parameters
        ----------
        filepath : str
            The filepath of the file containing the trained neural network and all related parameters.
        kwargs: dict
            key-value pairs that pass through to __init__ of the new class instance. When provided, these
            will supercede the parameters stored in the filepath
        """
        # load a trained Neural Network from file
        loaded_NN = torch.load(filepath, map_location=kwargs.get('device', torch.device('cpu')))

        # parameters priority: kwargs overwrite the parameters from the loaded NN:
        for par,parname in zip(['df_file','MPC_Param','features','labels'],['df_file','P','features','labels']):
            # print(kwargs.get(parname), loaded_NN[par])
            if kwargs.get(parname) is not None and hasattr(loaded_NN, par) and kwargs.get(parname)!=loaded_NN[par]:
                if isinstance(kwargs.get(parname), MPC_Param):
                    diff = kwargs[parname].diff(loaded_NN[par])
                    diff_string = '\n'.join([f'{f}: {loaded_NN[par].__dict__[f]} -> {kwargs[parname].__dict__[f]}' for f in diff])
                    warnings.warn(f'Stored parameter values overwritten by explicit values in the MPC parameters dataclass:\n{diff_string}.',UserWarning)
                else:
                    warnings.warn(f'Stored parameter value {par}={loaded_NN[par]} is overwritten by explicit value {kwargs.get(parname)}.',UserWarning)
            kwargs[parname] = kwargs.get(parname,loaded_NN[par])
        
        NN_out = NN_for_casadi(**kwargs)

        # this function will override every component that might already be filled in
        NN_out.scale = loaded_NN['NNscale']

        NN_out.train_param = loaded_NN['Train_Param'] if 'Train_Param' in loaded_NN else None
        NN_out.prun_param = loaded_NN['Prun_Param'] if 'Prun_Param' in loaded_NN else None

        NN_out.NN = FFNN(NN_out.scale, activation=loaded_NN['activation'], n_layers=loaded_NN['n_layers'], n_neurons=loaded_NN['n_neurons'], device=NN_out.device, dtype=NN_out.dtype)
        NN_out.NN.load_state_dict(loaded_NN['model_state_dict'])

        NN_out.NN.init_state_dict = loaded_NN['inital_model_state_dict'] if 'inital_model_state_dict' in loaded_NN else None
        NN_out.transform_pytorch_to_casadi()
        
        # print model stats
        features_str = str(NN_out.features) if len(NN_out.features)<5 else f'[{NN_out.features[0]}, {NN_out.features[1]}, ..., {NN_out.features[-2]}, {NN_out.features[-1]}] ({len(NN_out.features)} elements)'
        labels_str   = str(NN_out.labels)   if len(NN_out.labels)<5   else f'[{NN_out.labels[0]}, {NN_out.labels[1]}, ..., {NN_out.labels[-2]}, {NN_out.labels[-1]}] ({len(NN_out.labels)} elements)'
        print(f'''Model loaded from file "{os.path.abspath(filepath)}".
Model hyperparameters:
Feature names: {features_str}
Label names: {labels_str}
Activation function: {NN_out.NN.activation}
Number of hidden layers: {NN_out.NN.n_layers} with {NN_out.NN.n_neurons[1:]} neurons
Size of outer layers: {NN_out.NN.nin} input neurons, {NN_out.NN.nout} output neurons''')
            
        return NN_out
    
    @classmethod
    def loadNN_from_Param(
        nh_mpc_params: NH_AMPC_Param, 
        nn_dir: str, 
        mpc_dataset_dir: str, 
        device: torch.device, 
        dtype: torch.dtype,
    ):
        mpc_trainset_file = os.path.join(mpc_dataset_dir, nh_mpc_params.train_DS_name)
        trained_nn_file = os.path.join(
            nn_dir, 
            nh_mpc_params.Pruned_NN_name if nh_mpc_params.N_hidden_end is not None else nh_mpc_params.NN_name
        )
        return NN_for_casadi.NNload(trained_nn_file, df_file=mpc_trainset_file, P=nh_mpc_params, device=device, dtype=dtype)
    


def load_NN(
        nh_mpc_params: NH_AMPC_Param, 
        nn_dir: str, 
        mpc_dataset_dir: str, 
        device: torch.device, 
        dtype: torch.dtype,
        force_load_unpruned: bool = False
    ) -> NN_for_casadi:
    """

    Parameters
    ----------

    Returns
    -------
    """
    mpc_trainset_file = os.path.join(mpc_dataset_dir, nh_mpc_params.train_DS_name)
    trained_nn_file = os.path.join(
        nn_dir, 
        nh_mpc_params.NN_name if nh_mpc_params.N_hidden_end is None or force_load_unpruned else nh_mpc_params.Pruned_NN_name
    )
    return NN_for_casadi.NNload(trained_nn_file, df_file=mpc_trainset_file, P=nh_mpc_params, device=device, dtype=dtype)




def get_traindata_and_scaling(df: pd.DataFrame, features: list, labels: list, do_scaling: bool = True):
    """

    Parameters
    ----------

    Returns
    -------
    """
    wrong_labels = [x for x in labels if x not in df.columns]
    if len(wrong_labels) > 0:
        raise Exception(f'Expected labes not found in the dataframe: {wrong_labels}')
    
    tx, ty = df[features].to_numpy(), df[labels].to_numpy()
    nx, ny = len(features), len(labels)
    scale = {'xmean': np.ones(nx), 'xstd': np.ones(nx), 'ymean': np.ones(ny), 'ystd': np.ones(ny)}

    if do_scaling:
        # scaling to be used within NN, so that inputs and outputs remain the same
        sx = StandardScaler()
        sx.fit(tx) # fit scaler to features
        scale['xmean'], scale['xstd'] = sx.mean_, np.sqrt(sx.var_)

        sy = StandardScaler()
        sy.fit(ty)
        scale['ymean'], scale['ystd'] = sy.mean_, np.sqrt(sy.var_)
    return tx, ty, scale