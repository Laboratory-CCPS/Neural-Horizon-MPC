import numpy as np
import pandas as pd
import casadi as cs
import time
import os
import logging

from typing import Optional, Literal, Any
from acados_template.acados_model import AcadosModel
from tqdm.auto import trange


from .mpc_classes_acados import AMPC_class
from .parameters import AMPC_Param, NH_AMPC_Param
from .errors import OutOfBoundsError



class AMPC_dataset_gen():
    """
    Methods
    -------  
    get_new_init_state(self, scale=np.array([[.75],[.25],[.25],[.25]]), bias=np.array([[0],[np.pi],[0],[0]])) -> numpy.ndarray:
        Generates a new initial state for the MPC problem.  
    get_new_guesses(self, x_curr: np.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        Generates new state and input guesses.
    save_df(df=None, filename=None, filedir='Results//MPC_data_gen', float_format='%.4f') -> PathLike:
        Saves the generated data to a CSV file.
    generate_data(show_tqdm=True, filename=None, filedir=os.path.abspath(os.path.join('Results', 'MPC_data_gen'))) -> PathLike:
        Generates the dataset.
    """
    def __init__(self, 
            model: AcadosModel,
            P: AMPC_Param | NH_AMPC_Param, 
            solver_options: Optional[dict[str, Any]] = None,
            n_samples = 1000, 
            save_iter = 1000, 
            chance_reset_sample = 0.25,
            x_init: Optional[np.ndarray] = None,
            init_scale = np.array([.75, .25, .25, .25]), 
            init_bias = np.array([0, np.pi, 0, 0])
        ):
        """
        Generates training dataset of model predictive control (MPC) predictions.
        
        Parameters
        ----------
        ``P`` : MPC_Param dataclass object, optional 
            An object containing the MPC parameters. Default = MPC_Param()
        ``n_samples`` : int, optional 
            The number of samples to generate. Default = 1000
        ``save_iter`` : int, optional 
            Determines how often to save intermediate results. Default = 1000
        ``chance_reset_sample`` : float, optional 
            The probability of resetting the initial state to a random value. Default = 0.25
        ``init_scale`` : numpy.ndarray, optional
            Range of the domain where initial point is sampled as the proportion of state bounds (P.xbnd).
            Default = np.array([[.75],[.25],[.25],[.25]])
        ``init_bias`` : numpy.ndarray, optional
            Bias term for the new starting point sampling.
            Default = np.array([[0],[np.pi],[0],[0]])
        """
        if solver_options is None:
            solver_options = {}

        self.P = P
        self.model = model
        self.MPC = AMPC_class(model, self.P, solver_options=solver_options, ignore_status_errors={0, 5})
        self.n_samples = n_samples
        self.save_iter = save_iter
        self.chance_reset_sample = chance_reset_sample
        self.x_inits = x_init
        self.init_scale = init_scale
        self.init_bias = init_bias
        
        self.df = None

        self.store_iterate_filename = os.path.join('Temp' , 'ocp_solver_iterate.json')
        os.makedirs(os.path.dirname(self.store_iterate_filename), exist_ok=True)
        self.MPC.ocp_solver.store_iterate(self.store_iterate_filename, overwrite=True)
    

    def get_new_init_state(self):
        """
        Generates a new initial state for the MPC problem.
        The new starting point is uniformly sampled from
            [-P.xbnd*scale,P.xbnd*scale] + bias

        
        Returns
        -------
        ``x_curr`` : numpy.ndarray
            The new initial state.
        """
        bnd = self.P.xbnd.reshape((-1, ))*self.init_scale
        x_curr = (2*bnd*np.random.random((self.P.nx,)) - bnd + self.init_bias)
        return x_curr
    

    def get_new_guesses(self, x_curr: np.ndarray):
        """
        Generates new state and input guesses.
        State guess is a repeated ``x_curr`` and input guess is a zero array. 
        
        Parameters
        ----------
       ``x_curr`` : np.ndarray
            
        Returns
        -------
        ``x_guess`` : numpy.ndarray
            An array of initial state trajectory guess for the OCP solver.
        ``u_guess`` : numpy.ndarray
            An array of initial input trajectory guess for the OCP solver.
        """
        x_guess = np.repeat(x_curr.reshape((4, 1)), repeats=self.P.N_MPC, axis=1)
        u_guess = np.zeros((self.P.nu, self.P.N_MPC))
        return x_guess, u_guess
    

    def save_df(self, df=None, filename=None, filedir=os.path.abspath(os.path.join('Results', 'MPC_data_gen')), float_format='%.4f'):
        """
        Saves the generated data to a CSV file.
        
        Parameters
        ----------
        ``df`` : pandas DataFrame, optional (default=None)
            The dataframe to save.
        ``filename`` : str, optional (default=None)
            The filename to use.
        ``filedir`` : str, optional (default='Results//MPC_data_gen')
            The directory to save the file in.
        ``float_format`` : str, optional (default='%.4f')
            The format string for floating point numbers.
        
        Returns
        -------
        ``filename`` : str
            The location of the saved file.
        """
        if len(filedir)>0:
            os.makedirs(filedir, exist_ok=True)
        
        if df is None:
            df = self.df
        
        if filename is None:
            k = df.shape[0]
            filename = f'MPC_{self.P.N_MPC}steps_{k}datapoints.csv'
        filename = os.path.join(filedir, filename)
        
        df.to_csv(filename, index=False, float_format=float_format)
        self.P.save(filename=f'{os.path.basename(filename)[:-3]}json', filedir=filedir)
        
        return filename
    

    def generate_data(self, show_tqdm=True, filename=None, filedir=os.path.abspath(os.path.join('Results', 'MPC_data_gen'))):
        """
        Generates the dataset. Saves temporary data in the 'Temp' folder.
        
        Parameters
        ----------
        ``show_tqdm`` : bool, optional
            A flag to show a progress bar. Default = True
        ``filename`` : str, optional
            The filename where the data should be stored.
        ``filedir`` : PathLike, optional
            The directory where the data should be stored.
        
        Returns
        -------
        ``filename`` : str
            The location of the CSV file containing generated data.
        """
        ds_dict = {f'{x}_p{i}':np.empty(self.n_samples) 
                    for i in range(self.P.N_MPC) 
                    for x in self.P.xlabel+[self.P.ulabel]} | {f'{x}_p{self.P.N_MPC}':np.empty(self.n_samples) 
                    for x in self.P.xlabel}

        # initialize x[0]:
        x_reff = np.zeros((self.MPC.P.nx,))
        x_curr = self.get_new_init_state()
        x_guess, u_guess = self.get_new_guesses(x_curr)
        # x_broken = np.empty((0, self.P.nx))
        dir_name = f'Temp//Datagen_log_{time.strftime("%d_%b_%Y_%H_%M")}'

        for i in trange(self.n_samples, disable = not show_tqdm, desc=f'MPC horizon of {self.P.N_MPC}', unit='Samples'):
            

            time.sleep(0.001)
            if self.x_inits is None and np.random.random() < self.chance_reset_sample:
                x_curr = self.get_new_init_state()
                x_guess, u_guess = self.get_new_guesses(x_curr)
            elif self.x_inits is not None:
                x_curr = self.x_inits[i, :]
                x_guess, u_guess = self.get_new_guesses(x_curr)

            soltry=True
            while soltry:
                try:
                    self.MPC.reset_solver()
                    solve_results = self.MPC.solve(x0=x_curr, x_guess=x_guess, u_guess=u_guess)

                    in_ubnd = np.all(np.abs(solve_results.simU_traj) <= self.P.ubnd)
                    in_xbnd = np.all(np.abs(solve_results.simX_traj) <= self.P.xbnd)
                    if not (in_ubnd and in_xbnd):
                        raise OutOfBoundsError((solve_results.simU_traj, solve_results.simX_traj), (self.P.ubnd, self.P.xbnd))
                    
                    soltry = False
                except Exception as e:
                    time.sleep(0.001)
                    if self.MPC.current_status == 4 and ('nlp_solver_type' not in self.MPC.acados_options or self.MPC.acados_options['nlp_solver_type'] == 'SQP'):
                        self.MPC.recreate_solver()
                    x_curr = self.get_new_init_state()
                    x_guess, u_guess = self.get_new_guesses(x_curr)
                    logging.debug(f'{str(e)} :-> {x_curr}')
                
            x_curr = solve_results.simX_traj[:, 1]
            x_guess = np.hstack((solve_results.simX_traj[:, 1:], solve_results.simX_traj[:, -1:]))
            u_guess = np.vstack((solve_results.simU_traj[1:], solve_results.simU_traj[-1:]))

            for j in range(self.P.N_MPC+1):
                for k,l in enumerate(self.P.xlabel):
                    ds_dict[f'{l}_p{j}'][i] = solve_results.simX_traj[k, j]
                if j<self.P.N_MPC:
                    ds_dict[f'{self.P.ulabel}_p{j}'][i] = solve_results.simU_traj[:, j]    

            if not i%self.save_iter and i>0:
                # save intermediate results
                self.save_df(df=pd.DataFrame(ds_dict).iloc[:i+1], filedir=dir_name)

        # make dataframe
        self.df = pd.DataFrame(ds_dict)
        filename = self.save_df(filename=filename, filedir=filedir)

        # print('Broken initial datapoints: \n{}'.format(x_broken))
        
        print(f'Data generation complete.\nGenerated {self.df.shape[0]} data points for the baseline MPC with {self.P.N_MPC} steps.\nResult stored under   {filename}')
        
        return filename