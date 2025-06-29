import casadi as cs
import numpy as np
import time,warnings
from tqdm.auto import trange
from .inverted_pendulum import F_model_RK4, nonlinear_pend_dynamics
from .parameters import MPC_Param
from .neural_horizon import NN_for_casadi
from .mpc_dataclass import MPC_data

class MPC_class:
    """
    class MPC_class:
    Implements a basic Model Predictive Control (MPC) algorithm using a horizon of length P.N_MPC.

    Args:
        P (MPC_Param): The MPC_Param dataclass containing the MPC problem parameters.

    Attributes:
        P (MPC_Param): The MPC_Param object containing the MPC problem parameters.
        opti (casadi.Opti): The CasADi optimization problem.
        X (casadi.MX): The MPC state trajectory variable.
        U (casadi.MX): The MPC input trajectory variable.
        x0 (casadi.MX): The initial state parameter.
        F (function): The dynamics function of the system.
    
    Methods:
        solve(x0, x_init=None, u_init=None) -> Tuple[np.ndarray, np.ndarray, float]:
            Solves the Optimal Control Problem (OCP) from the specified initial state and trajectory guesses.
            Returns the optimal state trajectory, input trajectory, and optimization time.
    """
    # basic MPC class with horizon = P.N_MPC
    def __init__(self,P:MPC_Param):
        self.P = P
        self.opti = cs.Opti()
        N,nx,nu = self.P.N_MPC,self.P.nx,self.P.nu
        self.X = self.opti.variable(nx,N+1)
        self.U = self.opti.variable(nu,N)
        self.x0 = self.opti.parameter(nx,1)
        self.F = F_model_RK4(nonlinear_pend_dynamics(self.P),self.P)
        
        # costs and constraints
        self.J,g = 0,[self.X[:,0] == self.x0]
        for k in range(N):
            self.J += self.X[:,k+1].T @ self.P.Q @ self.X[:,k+1] + self.U[:,k] @ self.P.R @ self.U[:,k]
            g.append(self.X[:,k+1] == self.F(self.X[:,k],self.U[:,k]))
        
        self.opti.minimize(self.J)
        self.opti.subject_to(g)
        self.opti.subject_to(self.opti.bounded(-self.P.ubnd,self.U,self.P.ubnd))
        self.opti.subject_to(self.opti.bounded(-self.P.xbnd,self.X[:,1:],self.P.xbnd))
        self.opti.solver('ipopt',{'ipopt.print_level': 0, 'print_time': 0})
        
    def solve(self,x0,x_init=None,u_init=None):
        self.opti.set_value(self.x0,x0)
        if x_init is not None:
            self.opti.set_initial(self.X,x_init)
        if u_init is not None:
            self.opti.set_initial(self.U,u_init)
        start = time.time()
        sol = self.opti.solve()
        soltime = time.time() - start
        return sol.value(self.X),sol.value(self.U),soltime

class MPC_NN_class(MPC_class):
    """
    class MPC_NN_class(MPC_class):
    A model predictive control class with Neural horizon - it uses a neural network
    to approximate the optimal trajectory over a longer horizon than the baseline MPC.

    Inherits from the MPC_class, which constructs the optimization problem for
    the baseline MPC problem. Extends the Neural horizon from the end of MPC horizon.

    Parameters
    ----------
    NNmodel : class NNmodel
        A class containing the trained neural network model with the method
            NN_casadi(x0): a function that returns the optimal control sequence 
                for the horizon starting at x0, and the total cost of this sequence
                in the form of a list of CasADi symbolic variables.

    Attributes
    ----------
    NNmodel : class NNmodel
        The neural network model used in this MPC_NN_class object.

    Methods
    -------
    solve(x0, x_init=None, u_init=None)
        Solves the MPC problem starting at x0 using the given initial guesses for
        the state trajectory x and input trajectory u, and returns the resulting 
        state and input trajectories, as well as the time it took to solve the problem.

    Notes
    -----
    This MPC class uses a neural network model to estimate the cost and dynamics
    over a longer horizon than the explicit MPC model. The second horizon is defined
    by two parameters N_MPC and N_NN, where N_MPC is the length of the MPC horizon,
    and N_NN is the length of the Neural horizon that extends from the end of
    the MPC horizon. The neural network is trained to predict the optimal state 
    trajectory and cost for this longer horizon based on the state prediction x(N_MPC).

    The neural network model used must have a method `NN_casadi` that takes in a 
    vector of initial states and returns the optimal state trajectory and/or the total
    cost of the trajectory predicted by the neural network. This method should return
    a list of CasADi symbolic variables that define the control sequence and cost.
    The obtained optimal trajectory and/or costs are added as stage costs and state
    constraints for the Neural horizon before solving the modified MPC problem.
    """
    def __init__(self,NNmodel:NN_for_casadi):
        MPC_class.__init__(self,NNmodel.P)
        N1,N2,nx,nu = self.P.N_MPC,self.P.N_NN,self.P.nx,self.P.nu
        self.NNmodel = NNmodel
        
        # costs and constraints of the second (Neural) horizon
        # whole horizon estimate from NN: x(N1) |--> [{x1(N1+1),x2(N1+1),...xnx(N1+1),x1(N1+2),...xnx(N2)},{Jx},{Ju}]
        J2, NN_out = 0, self.NNmodel.NN_casadi(self.X[:,N1])
        # construct cost of second horizon
        if self.P.get_Ju:
            J2 += NN_out[-1]
            NN_out = NN_out[:-1] 
        if self.P.get_Jx:
            J2 += NN_out[-1]
            NN_out = NN_out[:-1]
        if self.P.get_states:
            x_NN = NN_out.reshape((nx,-1)) # turn into matrix of [x(N1+1),x(N1+2),...x(N2)]
            # assert correct horizon length:
            if x_NN.shape[1]!=N2:
                warnings.warn(f'Dimension mismatch! Was expecting {N2} timesteps, however the NN provides {x_NN.shape[1]} timesteps. Smaller horizon length is used.',UserWarning)
                N2 = min(x_NN.shape[1],N2)
            # add state bounds for NN predicted states if requested
            if self.P.get_state_bounds:
                self.opti.subject_to(self.opti.bounded(-self.P.xbnd,x_NN[:,:N2],self.P.xbnd))
            if not self.P.get_Jx:
                for k in range(N2):
                    J2 += x_NN[:,k].T @ self.P.Q_NN @ x_NN[:,k]
        
        # At this point we only need to add the Neural horizon stage costs to the cost function
        self.opti.minimize(self.J+J2)
    
def get_MPC_trajectory(controller:MPC_class,Fmodel:cs.Function,cname:str=None,W:np.array=None,xinit:np.ndarray=None,show_tqdm:bool=True) -> MPC_data:
    """
    Generates a trajectory based on the MPC controller.

    Args:
        controller (MPC_class): The MPC controller class.
        Fmodel (callable): Model dynamics function that propagates the state with the given inputs.
        cname (str, optional): A string that represents the controller label. Defaults to None.
        W (ndarray, optional): Variance for the additive state disturbance, can be a scalar or array of size X
        xinit (ndarray, optional): Initial state of the system. Defaults to None.
        show_tqdm (bool, optional): Whether to show a tqdm progress bar. Defaults to True.

    Returns:
        dict: A dictionary with the following keys:
            - 'X': A numpy array of size (nx, N_sim), which represents the state of the system for all time steps.
            - 'U': A numpy array of size (nu, N_sim), which represents the control inputs for all time steps.
            - 'Time': A numpy array of size (N_sim,), which represents the computation time for each time step.
            - 'X_traj': A numpy array of size (N_sim, nx, N_MPC+1), which represents the predicted states for all time steps.
            - 'U_traj': A numpy array of size (N_sim, nu, N_MPC), which represents the predicted control inputs for all time steps.
            - 'P': MPC_Param dataclass object containing the MPC controller parameters.
    """
    P = controller.P
    MPC_results = MPC_data(P)
    
    x = P.xinit if xinit is None else xinit
    x_guess = cs.repmat(x,1,controller.X.shape[1])
    u_guess = cs.repmat(0,controller.U.shape[0],controller.U.shape[1])
    
    for i in trange(P.N_sim,desc = cname,mininterval=0.5,disable = not show_tqdm):
        try:
            X,U,soltime = controller.solve(x0=x,x_init=x_guess,u_init=u_guess)
        except Exception as e:
            msg = e.args[0].split("'")[-2]
            warnings.warn(f'MPC solver failed on step {i}, reason: {msg}',UserWarning)
            break
        time.sleep(0.001)
        
        # bring the theta within the [-pi,+pi] range
        if X[1,0]>1.25*np.pi:
            X[1,:]-=2*np.pi
        elif X[1,0]<-1.25*np.pi:
            X[1,:]+=2*np.pi
            
        # log iteration
        MPC_results.Time[i] = soltime
        MPC_results.X[:,i] = X[:,0]
        MPC_results.U[:,i] = U[0]
        MPC_results.X_traj[i,:,:] = X
        MPC_results.U_traj[i,:,:] = U
        
        # propagate model
        x = Fmodel(X[:,0],U[0]).full()
        x_guess = np.hstack((X[:,1:],X[:,-1:]))
        u_guess = np.hstack((U[1:],U[-1:]))
        
        # add disturbance
        if W is not None:
            try:
                x += (np.random.randn(P.nx)*W).reshape(P.nx,-1)
            except ValueError as err:
                print(f'Disturbance variance W passed incorrectly, expected scalar or array of size ({P.nx},), got {W}')
                print(err)
                break
        
    # Calculate trajectory cost as (sum(xT Q x + uT R u))/T_traj
    T_traj = P.Ts*(i+1)
    Xcost = MPC_results.X[:,:i]
    Ucost = MPC_results.U[:,:i]
    MPC_results.Cost = (np.sum(Xcost*P.Q.dot(Xcost)) + np.sum(Ucost*P.R.dot(Ucost)))

    # freeze dataclass
    MPC_results.freeze()

    # debug print
    print(f'Trajectory cost calculation: {i} steps taken, traj. time {T_traj:0.2f} sec, cost = {MPC_results.Cost:0.2f}')
        
    return MPC_results