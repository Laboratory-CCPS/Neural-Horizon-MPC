import os
import time
import warnings
import casadi as cs
import numpy as np
import scipy.linalg
import gc

from typing import Optional, Literal
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
from tqdm.auto import trange
from copy import deepcopy

from .parameters import MPC_Param, NH_AMPC_Param, AMPC_Param
from .neural_horizon import NN_for_casadi
from .mpc_dataclass import AMPC_data, SolveStepResults
from .decorators import enforce_call_order
from .errors import SingletonError



# ===================================================================================================
# BASIC ACADOS MPC
# ===================================================================================================
class Base_AMPC_class:
    """
    A class as a basic acados MPC implementation based on the give parameters.
    Only one instance can be uses at a time and the instance has to be deleted 
    after use with 'cleanup()' and also deleted.
    """
    _instance_exists = False

    def __new__(cls, *args, **kwargs):
        """
        Ensures multiple instances are NOT called together, due to unexpected behavior of acados.
        """
        if not Base_AMPC_class._instance_exists:
            Base_AMPC_class._instance_exists = True
            return super(Base_AMPC_class, cls).__new__(cls)
        else:
            raise SingletonError(cls)
        
    def __init__(
            self, 
            model: AcadosModel,
            P: AMPC_Param | NH_AMPC_Param,  
            solver_options: Optional[dict] = None,
            acados_name: str = '',
            solver_verbose = False,
            ignore_status_errors: Optional[set[Literal[1, 2, 3, 4, 5]]] = None,
            horizon_name: str = '',
    ):
        """
        Implements a acados MPC using a horizon of length ``P.N_MPC``, while using the default or 
        given acados options, for the solver.

        !!ATTENTION!! function ``cleanup()`` must be called after usage.

        Parameters
        ----------
        ``model`` : AcadosModel
            An acados model, either only continuous or with discrete implementation.
        ``P`` : MPC_Param
            The MPC_Param dataclass containing the MPC problem parameters.
        ``solver_options`` : dict, optional 
            A dictionary with acados options for ocp and ocp solver as well as cost type.
             See page -> https://docs.acados.org/python_interface/index.html 
              Default = None
        ``acados_name`` : str, optional
            A string defining the acados name used for ocp_solver.json and tqdm. 
             Default = ''
        ``solver_verbose`` : bool, optional
            A bool defining if during solving and broke solver the statistics are printet.
             Default = False,
        ``ignore_status_errors`` : set of [1, 2, 3, 4, 5], optional
            A set defining if during solving it does not raises an error 
             when {1, 2, 3, 4, 5} are the status returns of acados.
              Default = True
        ``horizon_name`` : str, optional
            A string used for ocp_solver.json and tqdm.
             Default = ''
        """
        self.acados_options = {} if solver_options is None else deepcopy(solver_options) # deepcopy because of pops later.

        # Params and options
        self.P = P
        self.cost_type = self.acados_options.pop('cost_type', 'LINEAR_LS')
        self.acados_name = acados_name
        self.solver_verbose = solver_verbose
        self.ignore_status_errors = set() if ignore_status_errors is None else ignore_status_errors

        self.horizon_name = horizon_name

        self.max_rti_iters = self.acados_options.pop('max_rti_iters', 10) 
        self.rti_tol = self.acados_options.pop('rti_tol', 1e-4) 
        self.use_iter_rti_impl = self.acados_options.pop('use_iter_rti_impl', False)
        self.use_initial_guesses = self.acados_options.pop('use_initial_guesses', False)

        self.solver_status_meanings = {
            0 : 'ACADOS_SUCCESS',
            1 : 'ACADOS_FAILURE',
            2 : 'ACADOS_MAXITER',
            3 : 'ACADOS_MINSTEP',
            4 : 'ACADOS_QP_FAILURE',
            5 : 'ACADOS_READY',
        }
        self.current_status = 0

        if not self.ignore_status_errors.issubset(self.solver_status_meanings.keys()):
            raise ValueError(f'Parameter must be a set in {set(self.solver_status_meanings.keys())} -> got {self.ignore_status_errors}')
        
        # Model
        self.model = model

        # Dims
        self.N_MPC = self.P.N_MPC
        self.nx = self.P.nx
        self.nu = self.P.nu
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        # Base OCP
        self.create_base_ocp()


    @enforce_call_order('set_ocp')
    def create_base_ocp(self):
        """
        Sets the basic acados OCP filled with the model and the horizon.
        """
        self.ocp = AcadosOcp()

        # Model
        self.ocp.model = self.model

        # OCP Dims
        self.ocp.dims.N = self.N_MPC 


    @enforce_call_order('set_solver_options')
    def set_solver_options(self):
        """
        Sets all given acados options by using the key as attribute of ocp.solver_options 
        and the value as value.
        """
        # Prediction horizon
        self.ocp.solver_options.tf = self.P.Ts*self.N_MPC

        # Solver options
        for key, value in self.acados_options.items():
            try:
                setattr(self.ocp.solver_options, key, value)
            except Exception as error:
                print('options {} cannot be set!\nException occurred: {} - {}'. format(key, type(error).__name__, error))

    
    @enforce_call_order('set_constraints')
    def set_ocp_constraints(self):
        """
        Sets the OCP constraints given by an instance MPC_param.
        """
        # U 
        self.ocp.constraints.lbu = np.array([-self.P.ubnd])
        self.ocp.constraints.ubu = np.array([self.P.ubnd])
        self.ocp.constraints.idxbu = np.array([0])

        # X
        self.ocp.constraints.lbx = -self.P.xbnd.reshape((-1,))
        self.ocp.constraints.ubx = self.P.xbnd.reshape((-1,))
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        # X_e
        self.ocp.constraints.lbx_e = -self.P.xbnd.reshape((-1,))
        self.ocp.constraints.ubx_e = self.P.xbnd.reshape((-1,))
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

        # X0
        self.ocp.constraints.x0 = self.P.xinit.reshape((-1,))


    @enforce_call_order('set_cost')
    def set_cost(self):
        """
        Sets the stage and terminal cost to the give type of cost.
        """
        if self.cost_type == 'LINEAR_LS':
            self.set_LLS_stage_cost()
            self.set_LLS_terminal_cost()
        elif self.cost_type == 'EXTERNAL':
            self.set_EXT_stage_cost()
            self.set_EXT_terminal_cost()


    @enforce_call_order('set_stage_cost')
    def set_LLS_stage_cost(self):
        """
        Sets the stage cost to a basic linear least sqares cost.
        """
        # State X
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx,:self.nx] = np.eye(self.nx)
        
        # U
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[4,0] = 1.0

        # Cost
        self.ocp.cost.W = scipy.linalg.block_diag(self.P.Q, self.P.R)
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.yref  = np.zeros((self.ny, ))

        
    @enforce_call_order('set_terminal_cost')
    def set_LLS_terminal_cost(self):
        """
        Sets the terminal cost to a basic linear least sqares cost.
        """
        # Terminal X 
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # Terminal Cost
        self.ocp.cost.W_e = self.P.Q
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.yref_e = np.zeros((self.ny_e, ))


    @enforce_call_order('set_stage_cost')
    def set_EXT_stage_cost(self):
        """
        Sets the stage cost to an external casadi cost

        0.5 * cs.vertcat(x, u).T @ cost_W @ cs.vertcat(x, u)

        where ``cost_W`` is blockdiagonal matrix of P.Q and P.R
        """
        # States, input and weight
        x, u  = self.ocp.model.x, self.ocp.model.u
        cost_W = scipy.linalg.block_diag(self.P.Q, self.P.R)

        # Cost
        self.ocp.cost.cost_type = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost = .5*cs.vertcat(x, u).T @ cost_W @ cs.vertcat(x, u)

    
    @enforce_call_order('set_terminal_cost')
    def set_EXT_terminal_cost(self):
        """
        Sets the terminal cost to an external casadi cost

        0.5 * x.T @ self.P.Q @ x
        """
        # States
        x  = self.ocp.model.x
        
        # Terminal Cost
        self.ocp.cost.cost_type_e = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_e = .5*x.T @ self.P.Q @ x


    def set_acados_name(self, nh_str=None, join_str='_'):
        """
        Sets the ``MPC_name``, by combining all acados option values that are strings
        and connect them with ``join_str``. Also add a the ``nh_str`` at the start, if not None.

        Keyword Parameters
        ------------------
        ``nh_str`` : str, optional
            A string that is placed at the start of the ``ocp_solver_string``, when not None.
            Default = None
        ``join_str`` : str, optional
            A string that connects all the acados option values. Default = '_'
        """
        name = [v for v in self.acados_options.values() if type(v) is str]
        name = join_str.join(name)

        if nh_str is not None:
            name = join_str.join((nh_str, name))

        self.acados_name = name


    @enforce_call_order('set_ocp_string')
    def set_ocp_string(self):
        """
        Sets the ``ocp_solver_string``, by combining 'acados_ocp_' with the MPC_name 
        as well as the extension.
        """
        solver_dir = os.path.abspath('temp_solver_jsons')
        if not os.path.exists(solver_dir):
            os.mkdir(solver_dir)

        self.solver_json_path = os.path.join(
            solver_dir,
            f'ampc_solver_{self.acados_name}_{self.horizon_name}.json'
        )
        

    @enforce_call_order('set_ocp_solver')
    def create_ocp_solver(self):
        """
        Creates an acados OCP solver out of the OCP. 
        Stores the solver settings in a json named by the ``ocp_solver_string``.
        """
        assert not hasattr(self, 'ocp_solver'), 'creating multiple OCP SOLVERS leads to problems!'
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.solver_json_path, verbose=self.solver_verbose)

    
    @enforce_call_order('deleted')
    def cleanup(self):
        """
        Deletes first the ``ocp_solver``, then the ``ocp`` and then the ``model`` of this instance.
        Also resets the option to create a new instance of this class and collect garbage.
        """
        try:
            os.remove(self.solver_json_path)
            del self.ocp_solver
            del self.ocp
            del self.model
        except:
            pass
        finally:
            Base_AMPC_class._instance_exists = False
            gc.collect() 


    def reset_solver(self):
        self.ocp_solver.reset(reset_qp_solver_mem=1)

    
    def recreate_solver(self):
        del self.ocp_solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.solver_json_path, verbose=self.solver_verbose)
        

    def solve(
            self, 
            x0: np.ndarray,
            x_guess: Optional[np.ndarray] = None, 
            u_guess: Optional[np.ndarray] = None
    ) -> SolveStepResults:
        """
        Solves the OCP with acados. Print statistics, if solver returns status that is not 0. <br>
        status:
        - 0 = ACADOS_SUCCESS
        - 1 = ACADOS_FAILURE
        - 2 = ACADOS_MAXITER
        - 3 = ACADOS_MINSTEP
        - 4 = ACADOS_QP_FAILURE
        - 5 = ACADOS_READY
        
        Parameters
        ----------
        ``x0`` : numpy.ndarray
            An array that is the initial value for the open loop MPC.
        ``x_guess`` : numpy.ndarray
            An array that is the initial guess for all states the open loop MPC should predict.
        ``u_guess`` : numpy.ndarray
            An array that is the initial guess for all inputs the open loop MPC should predict.

        Returns
        -------
        ``solver_results`` : SolveStepResults
            A dataclass with solver times, solver iterations, x- and y-trajectories
        """
        solver_results = SolveStepResults.from_params(self.P)

        # initial guesses
        if x_guess is not None and self.use_initial_guesses:
            for i_x in range(x_guess.shape[1]):
                self.ocp_solver.set(i_x, "x", x_guess[:, i_x])
        if u_guess is not None and self.use_initial_guesses:
            for i_u in range(u_guess.shape[1]):
                self.ocp_solver.set(i_u, "u", u_guess[:, i_u])


        # SQP
        if self.ocp.solver_options.nlp_solver_type == 'SQP':
            # initial bounds
            self.ocp_solver.set(0, "lbx", x0)
            self.ocp_solver.set(0, "ubx", x0)

            # solve SQP
            start_time = time.perf_counter()
            status = self.ocp_solver.solve()
            solver_results.soltime_p = time.perf_counter() - start_time

            # check status
            self._solver_status_error(status)

            # save results
            solver_results.soltime_a = self.ocp_solver.get_stats('time_tot')
            solver_results.soliters = np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)
            
            
        # ACADOS RTI
        elif self.ocp.solver_options.nlp_solver_type == 'SQP_RTI' and not self.use_iter_rti_impl:
            # preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            start_time = time.perf_counter()
            status = self.ocp_solver.solve()
            prep_time = time.perf_counter() - start_time

            # save preperation results
            solver_results.prep_time = self.ocp_solver.get_stats('time_tot')
            solver_results.prep_iters = np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)

            # initial bounds
            self.ocp_solver.set(0, "lbx", x0)
            self.ocp_solver.set(0, "ubx", x0)

            # feedback phase
            self.ocp_solver.options_set('rti_phase', 2)
            start_time = time.perf_counter()
            status = self.ocp_solver.solve()
            solver_results.soltime_p = time.perf_counter() - start_time + prep_time

            # check status
            self._solver_status_error(status)

            # save feedback results
            solver_results.fb_time = self.ocp_solver.get_stats('time_tot')
            solver_results.fb_iters = np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)
            solver_results.add_prep_and_fb()


        # TO CONVERGENCE RTI
        elif self.ocp.solver_options.nlp_solver_type == 'SQP_RTI' \
            and self.use_iter_rti_impl and 4 == self.ocp.solver_options.as_rti_level:

            # initial bounds
            self.ocp_solver.set(0, "lbx", x0)
            self.ocp_solver.set(0, "ubx", x0)

            for i in range(self.max_rti_iters):
                # solve problem
                start_time = time.perf_counter()
                status = self.ocp_solver.solve()
                solver_results.soltime_p += time.perf_counter() - start_time

                # check status
                self._solver_status_error(status)

                # save results
                solver_results.soliters += np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)
                solver_results.soltime_a += self.ocp_solver.get_stats('time_tot')

                # check convergence
                start_time = time.perf_counter()
                residuals = self.ocp_solver.get_residuals()
                if max(residuals) < self.rti_tol:
                    break
                solver_results.soltime_p += time.perf_counter() - start_time

        
        # get open loop traj
        for i in range(self.N_MPC):
            solver_results.simX_traj[:, i] = self.ocp_solver.get(i, "x")
            solver_results.simU_traj[:, i] = self.ocp_solver.get(i, "u")
        solver_results.simX_traj[:, self.N_MPC] = self.ocp_solver.get(self.N_MPC, "x")

        return solver_results
    

    def _solver_status_error(self, status):
        self.current_status = status
        # solver failed print
        if self.solver_verbose and status != 0:
            self.ocp_solver.print_statistics()
        if status not in self.ignore_status_errors and status != 0:
            raise Exception('Solver returned status {} -> {}'.format(status, self.solver_status_meanings[status]))
    



# ===================================================================================================
# ACADOS MPC
# ===================================================================================================
class AMPC_class(Base_AMPC_class):
    """
    Class that uses ``Base_MPC_acados`` and instantiates a basic acados MPC by setting cost, 
    constraints, solver options, solver string and solver in this order.
    """
    def __init__(
            self, 
            model: AcadosModel,
            P: AMPC_Param, 
            solver_options: Optional[dict] = None, 
            horizon_name: str = 'AMPC',
            acados_name: str = '',
            solver_verbose = False,
            ignore_status_errors: Optional[set[Literal[1, 2, 3, 4, 5]]] = None,
        ):
        """
        Implements a acados MPC using a horizon of length ``P.N_MPC``, while using the default or 
        given acados options, for the solver. Calls all the functions to create an ocp solver.

        !!ATTENTION!! function ``cleanup()`` must be called after usage.

        Parameters
        ----------
        ``model`` : AcadosModel
            An acados model, either only continuous or with discrete implementation.
        ``P`` : MPC_Param
            The MPC_Param dataclass containing the MPC problem parameters.
        ``solver_options`` : dict, optional 
            A dictionary with acados options for ocp and ocp solver as well as cost type.
             See page -> https://docs.acados.org/python_interface/index.html 
              Default = None
        ``acados_name`` : str, optional
            A string defining the acados name used for ocp_solver.json and tqdm. 
             Default = ''
        ``solver_verbose`` : bool, optional
            A bool defining if during solving and broke solver the statistics are printet.
             Default = False,
        ``ignore_status_errors`` : set of [1, 2, 3, 4, 5], optional
            A set defining if during solving it does not raises an error 
             when {1, 2, 3, 4, 5} are the status returns of acados.
              Default = True
        ``horizon_name`` : str, optional
            A string used for ocp_solver.json and tqdm.
             Default = ''
        """
        if solver_options is None:
            solver_options = {}

        super().__init__(
            model,
            P, 
            solver_options = solver_options, 
            horizon_name = horizon_name, 
            acados_name = acados_name, 
            solver_verbose = solver_verbose, 
            ignore_status_errors = ignore_status_errors,
        )
        # Cost
        self.set_cost()

        # Constraints
        self.set_ocp_constraints()

        # OCP solver options
        self.set_solver_options()

        # OCP string
        self.set_ocp_string()

        # OCP solver
        self.create_ocp_solver()

        


# ===================================================================================================
# ACADOS NH-MPC
# ===================================================================================================
class Base_NH_AMPC_class(Base_AMPC_class):
    """
    Class that uses ``Base_MPC_acados`` and instantiates a neural horizon acados MPC.
    """
    def __init__(
            self, 
            model: AcadosModel,
            NNmodel: NN_for_casadi, 
            solver_options: Optional[dict] = None, 
            horizon_name: str = 'NH_AMPC', 
            acados_name: str = '', 
            solver_verbose: bool = False,
        ):
        """
        Implements a NH-AMPC using a horizon of length ``P.N_MPC`` for M and ``P.N_NN`` for N, 
        while using the default or given acados options, for the solver. 

        !!ATTENTION!! function ``cleanup()`` must be called after usage.

        Parameters
        ----------
        ``model`` : AcadosModel
            An acados model, either only continuous or with discrete implementation.
        ``NNmodel`` : NN_for_casadi
            A class containing the trained neural network model with the method
             NN_casadi(x0): a function that returns the optimal control sequence 
              for the horizon starting at x0, and the total cost of this sequence
               in the form of a list of CasADi symbolic variables.
        ``solver_options`` : dict, optional 
            A dictionary with acados options for ocp and ocp solver as well as cost type.
             See page -> https://docs.acados.org/python_interface/index.html 
              Default = None
        ``acados_name`` : str, optional
            A string defining the acados name used for ocp_solver.json and tqdm. 
             Default = ''
        ``solver_verbose`` : bool, optional
            A bool defining if during solving and broke solver the statistics are printet.
             Default = False,
        ``ignore_status_errors`` : set of [1, 2, 3, 4, 5], optional
            A set defining if during solving it does not raises an error 
             when {1, 2, 3, 4, 5} are the status returns of acados.
              Default = True
        ``horizon_name`` : str, optional
            A string used for ocp_solver.json and tqdm.
             Default = ''
        """
        if solver_options is None:
            solver_options = {}

        if 'cost_type' in solver_options:
            solver_options = deepcopy(solver_options) # cause of pop
            solver_options.pop('cost_type')
            if 'LINEAR_LS' == solver_options['cost_tpye']: 
                print('Used Nonlinear cost type for calculation!')
                
        super().__init__(
            model,
            NNmodel.P, 
            solver_options = solver_options, 
            horizon_name = horizon_name, 
            acados_name = acados_name, 
            solver_verbose = solver_verbose, 
            ignore_status_errors = {0, 2, 3, 5},
        )
        self.NNmodel = NNmodel


    @enforce_call_order('set_terminal_cost')
    def set_NLLS_terminal_cost_NN(self):
        """
        Sets an nonlinear least sqares terminal cost for the NH-MPC in acados. 
        The ``cost_y_expr_e`` is a vertical stacked x array:

        [x_m[0] x_m[1] x_m[2] x_m[3] x_(m+1)[0] x_(m+1)[1] x_(m+1)[2] x_(m+1)[3] ... x_n[0] x_n[1] x_n[2] x_n[3]].T

        The weight is a block of [Q, *[Q_NN]*N_NN]
        """
        # States
        x = self.ocp.model.x
        x_NN = self.NNmodel.NN_casadi(x)

        # set cost typ to external
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # Cost
        self.ocp.model.cost_y_expr_e = cs.vertcat(x, x_NN)
        self.ocp.cost.W_e = scipy.linalg.block_diag(self.P.Q, *[self.P.Q_NN]*self.P.N_NN)
        self.ocp.cost.yref_e = np.zeros((self.ny_e + x_NN.shape[0], ))



class NH_AMPC_class(Base_NH_AMPC_class):
    """
    Class that uses ``Base_NH_AMPC_class`` and instantiates a neural horizon acados MPC.
    """
    def __init__(
            self, 
            model: AcadosModel,
            NNmodel: NN_for_casadi, 
            solver_options: Optional[dict] = None, 
            horizon_name: str = 'NH_AMPC', 
            acados_name: str = '', 
            solver_verbose: bool = False,
        ):
        """
        Implements a NH-AMPC using a horizon of length ``P.N_MPC`` for M and ``P.N_NN`` for N, 
        while using the default or given acados options, for the solver.
        Calls all the functions to create an NH-AMPC solver. 

        !!ATTENTION!! function ``cleanup()`` must be called after usage.

        Parameters
        ----------
        ``model`` : AcadosModel
            An acados model, either only continuous or with discrete implementation.
        ``NNmodel`` : NN_for_casadi
            A class containing the trained neural network model with the method
             NN_casadi(x0): a function that returns the optimal control sequence 
              for the horizon starting at x0, and the total cost of this sequence
               in the form of a list of CasADi symbolic variables.
        ``solver_options`` : dict, optional 
            A dictionary with acados options for ocp and ocp solver as well as cost type.
             See page -> https://docs.acados.org/python_interface/index.html 
              Default = None
        ``acados_name`` : str, optional
            A string defining the acados name used for ocp_solver.json and tqdm. 
             Default = ''
        ``solver_verbose`` : bool, optional
            A bool defining if during solving and broke solver the statistics are printet.
             Default = False,
        ``ignore_status_errors`` : set of [1, 2, 3, 4, 5], optional
            A set defining if during solving it does not raises an error 
             when {1, 2, 3, 4, 5} are the status returns of acados.
              Default = True
        ``horizon_name`` : str, optional
            A string used for ocp_solver.json and tqdm.
             Default = ''
        """
        if solver_options is None:
            solver_options = {}

        super().__init__(
            model,
            NNmodel, 
            solver_options = solver_options, 
            horizon_name = horizon_name, 
            acados_name = acados_name, 
            solver_verbose = solver_verbose, 
        )
        
        # Cost
        self.set_LLS_stage_cost()
        self.set_NLLS_terminal_cost_NN()

        # Constraints
        self.set_ocp_constraints()

        # OCP solver options
        self.set_solver_options()

        # OCP string
        self.set_ocp_string()

        # OCP solver
        self.create_ocp_solver()




# ===================================================================================================
# TRAJECTORY
# ===================================================================================================
def get_AMPC_trajectory(
            controller: AMPC_class | NH_AMPC_class,
            W: np.ndarray=None,
            xinit: np.ndarray=None,
            show_tqdm: bool=True,
            verbose = True
    ) -> AMPC_data:
    """
    Generates a trajectory based on the MPC ``controller``.

    Parameters
    ----------
    ``controller`` : MPC_acados | MPC_NN_acados 
        The acados MPC controller class.
    ``W`` : ndarray, optional
        Variance for the additive state disturbance, can be a scalar or array of size X. Default = None
    ``xinit`` : ndarray, optional
        Initial state of the system. Defaults = None.
    ``show_tqdm`` : bool, optional
        Whether to show a tqdm progress bar. Defaults = True.

    Returns
    -------
    ``MPC_results`` : AMPC_data
        A dataclass with the closed and open loop trajectories, solving times, parameters and acados options. 
    """
    P = controller.P
    MPC_results = AMPC_data(P=P, acados_name=controller.acados_name, acados_options=controller.acados_options)
    x_curr = P.xinit.reshape((-1,)) if xinit is None else xinit

    x_guess = np.repeat(x_curr.reshape((4, 1)), repeats=controller.N_MPC, axis=1)
    u_guess = np.zeros((1, controller.N_MPC))

    for i in trange(P.N_sim, desc = controller.horizon_name, mininterval=0.5, disable = not show_tqdm):
        try:
            solver_results = controller.solve(x_curr, x_guess, u_guess)
        except Exception as e:
            msg = e.args[0]
            warnings.warn(f'MPC solver failed on step {i}, reason: {msg}',UserWarning)
            break
        time.sleep(0.001)

        # bring the theta within the [-pi,+pi] range
        if solver_results.simX_traj[1, 0] > 1.25*np.pi:
            solver_results.simX_traj[1, :] -= 2*np.pi
        elif solver_results.simX_traj[1, 0] < -1.25*np.pi:
            solver_results.simX_traj[1, 0] += 2*np.pi
            
        ## SAVE RESULTS
        # timings
        MPC_results.Time[i] = solver_results.soltime_p
        MPC_results.Acados_Time[i] = solver_results.soltime_a
        MPC_results.Prep_Time[i] = solver_results.prep_time
        MPC_results.Fb_Time[i] = solver_results.fb_time

        # iterations
        MPC_results.Iterations[i] = solver_results.soliters
        MPC_results.Prep_Iterations[i] = solver_results.prep_iters
        MPC_results.Fb_Iterations[i] = solver_results.fb_iters

        # trajectories
        MPC_results.X[:,i] = x_curr
        MPC_results.U[:,i] = solver_results.simU_traj[:, 0]
        MPC_results.X_traj[i,:,:] = solver_results.simX_traj
        MPC_results.U_traj[i,:,:] = solver_results.simU_traj

        x_curr = solver_results.simX_traj[:, 1]
        x_guess = np.hstack((solver_results.simX_traj[:, 1:], solver_results.simX_traj[:, -1:]))
        u_guess = np.vstack((solver_results.simU_traj[1:], solver_results.simU_traj[-1:]))

        # add disturbance
        if W is not None:
            try:
                x_curr += (np.random.randn(P.nx)*W)
            except ValueError as err:
                print(f'Disturbance variance W passed incorrectly, expected scalar or array of size ({P.nx},), got {W}')
                print(err)
                break

    # add predicted state trajectories 
    if isinstance(controller, NH_AMPC_class):
        MPC_results.X_traj = add_NN_Xtrajs(MPC_results.X_traj, P, controller.NNmodel)

    # calculate cost
    T_traj = P.Ts*(i+1)
    X_cost = MPC_results.X[:,:i]
    U_cost = MPC_results.U[:,:i]
    MPC_results.Cost = (np.sum(X_cost*P.Q.dot(X_cost)) + np.sum(U_cost*P.R.dot(U_cost)))

    # freeze dataclass
    MPC_results.freeze()

    if verbose:
        print(f'Trajectory cost calculation: {i} steps taken, traj. time {T_traj:0.2f} sec, cost = {MPC_results.Cost:0.7f}')

    return MPC_results



def add_NN_Xtrajs(x_trajs: np.ndarray, P: AMPC_Param | NH_AMPC_Param, NN: NN_for_casadi):
    """
    Adds the ocp trajectories calculated by the NN, depending on the terminal MPC state. 
    
    Parameters
    ----------
    ``x_trajs`` : np.ndarray
        Trajectory of the OCP calculated by the MPC.
    ``P`` : MPC_Param
        MPC_Param dataclass object containing the MPC controller parameters.
    ``NN`` : NN_for_casadi
        A class containing the trained neural network model with the method
         NN_casadi(x0): a function that returns the optimal control sequence 
          for the horizon starting at x0, and the total cost of this sequence
           in the form of a list of CasADi symbolic variables.

    Returns
    -------
    ``x_trajs`` : numpy.ndarray
        Updated trajectory of the OCP calculated by the MPC.
    """
    x_trajs = np.concatenate((x_trajs, np.full((P.N_sim, P.nx, P.N_NN), np.nan)), axis=2)
    for k in range(P.N_sim):
        # terminal X from MPC on Nth sim step
        x_terminal = x_trajs[k, :, P.N_MPC]

        # X of NN
        x_NN = NN.NN_casadi(x_terminal)

        # combined X
        x_trajs[k, :, P.N_MPC+1:] = np.reshape(x_NN, (-1, P.nx)).T
    return x_trajs