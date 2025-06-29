import numpy as np
import pandas as pd
import casadi as cs
from tqdm.auto import trange
import time,os
from .mpc_classes import MPC_class
from .parameters import MPC_Param

class MPC_dataset_gen():
    """
    Generates training dataset of model predictive control (MPC) predictions.
    
    Methods:
    --------
    __init__(self, P=MPC_Param(), n_samples=1000, save_iter=1000, chance_reset_sample=0.25):
        Initializes an MPC_dataset_gen object.
        
        Parameters:
        -----------
        P : MPC_Param dataclass object, optional (default=MPC_Param())
            An object containing the MPC parameters.
        n_samples : int, optional (default=1000)
            The number of samples to generate.
        save_iter : int, optional (default=1000)
            Determines how often to save intermediate results.
        chance_reset_sample : float, optional (default=0.25)
            The probability of resetting the initial state to a random value.
            
    get_new_init_state(self, scale=np.array([[.75],[.25],[.25],[.25]]), bias=np.array([[0],[np.pi],[0],[0]])):
        Generates a new initial state for the MPC problem.
        The new starting point is uniformly sampled from
            [-P.xbnd*scale,P.xbnd*scale] + bias
        
        Parameters:
        -----------
        scale : numpy array, optional (default=np.array([[.75],[.25],[.25],[.25]]))
            Range of the domain where initial point is sampled as the proportion of state bounds (P.xbnd).
        bias : numpy array, optional (default=np.array([[0],[np.pi],[0],[0]]))
            Bias term for the new starting point sampling.
        
        Returns:
        --------
        x_init : numpy array
            The new initial state.
        x_guess : numpy array
            An array of initial state trajectory guess for the OCP solver.
        u_guess : numpy array
            An array of initial input trajectory guess for the OCP solver.
            
    save_df(self, df=None, filename=None, filedir='Results//MPC_data_gen', float_format='%.4f'):
        Saves the generated data to a CSV file.
        
        Parameters:
        -----------
        df : pandas DataFrame, optional (default=None)
            The dataframe to save.
        filename : str, optional (default=None)
            The filename to use.
        filedir : str, optional (default='Results//MPC_data_gen')
            The directory to save the file in.
        float_format : str, optional (default='%.4f')
            The format string for floating point numbers.
        
        Returns:
        --------
        filename : str
            The location of the saved file.
    
    generate_data(self, show_tqdm=True):
        Generates the dataset.
        
        Parameters:
        -----------
        show_tqdm : bool, optional (default=True)
            A flag to show a progress bar.
        
        Returns:
        --------
        filename : str
            The location of the CSV file containing generated data.
    """
    def __init__(self,P=MPC_Param(),n_samples=1000,save_iter=1000,chance_reset_sample = 0.25):
        self.P = P
        self.MPC = MPC_class(self.P)
        self.n_samples = n_samples
        self.save_iter = save_iter
        self.chance_reset_sample = chance_reset_sample
        
        self.df = None
    
    def get_new_init_state(self, scale = np.array([[.75],[.25],[.25],[.25]]), bias = np.array([[0],[np.pi],[0],[0]])):
        bnd = self.P.xbnd*scale
        x_init = (2*bnd*np.random.random((self.P.nx,1)) - bnd + bias)
        x_guess = np.zeros((x_init,1,self.P.N_MPC+1))
        u_guess = np.zeros((self.P.nu,self.P.N_MPC))
        return x_init,x_guess,u_guess
    
    def save_df(self,df=None,filename=None,filedir='Results//MPC_data_gen',float_format='%.4f'):
        if len(filedir)>0:
            os.makedirs(filedir,exist_ok=True)
        
        if df is None:
            df = self.df
        
        if filename is None:
            k = df.shape[0]
            filename = os.path.join(filedir,f'MPC_{self.P.N_MPC}steps_{k}datapoints.csv')
        
        df.to_csv(filename, index=False, float_format=float_format)
        self.P.save(filename=f'{os.path.basename(filename)[:-3]}json',filedir=filedir)
        
        return filename
        
    def generate_data(self,show_tqdm=True):
        ds_dict = {f'{x}_p{i}':np.empty(self.n_samples) for i in range(self.P.N_MPC) for x in self.P.xlabel+[self.P.ulabel]}|{f'{x}_p{self.P.N_MPC}':np.empty(self.n_samples) for x in self.P.xlabel}

        # initialize x[0]:
        x_init,x_guess,u_guess = self.get_new_init_state()
        x_broken = np.empty((0,self.P.nx))
        dir_name = f'Temp//Datagen_log_{time.strftime("%d_%b_%Y_%H_%M")}'

        for i in trange(self.n_samples,disable = not show_tqdm):
            time.sleep(0.001)
            if np.random.random()<self.chance_reset_sample:
                x_init,x_guess,u_guess = self.get_new_init_state()

            soltry=True
            while soltry:
                try:
                    X,U,_ = self.MPC.solve(x0=x_init,x_init=x_guess,u_init=u_guess)
                    soltry=False
                except:
                    x_broken = np.vstack((x_broken,x_init))
                    x_init,x_guess,u_guess = self.get_new_init_state()

            x_init = X[:,1]
            x_guess = np.hstack((X[:,1:],X[:,-1:]))
            u_guess = np.hstack((U[1:],U[-1:]))

            for j in range(self.P.N_MPC+1):
                for k,l in enumerate(self.P.xlabel):
                    ds_dict[f'{l}_p{j}'][i] = X[k,j]
                if j<self.P.N_MPC:
                    ds_dict[f'{self.P.ulabel}_p{j}'][i] = U[j]    

            if not i%self.save_iter and i>0:
                # save intermediate results
                self.save_df(df=pd.DataFrame(ds_dict).iloc[:i+1],filedir=dir_name)

        # make dataframe
        self.df = pd.DataFrame(ds_dict)
        filename = self.save_df()
        
        print(f'Data generation complete.\nGenerated {self.df.shape[0]} data points for the baseline MPC with {self.P.N_MPC} steps.\nResult stored under   {filename}')
        
        return filename