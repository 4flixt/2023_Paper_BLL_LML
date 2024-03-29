import numpy as np
import pandas as pd
import pdb


class System:
    """ A class for simulation discrete-time state-space systems.
    
    """
    def __init__(self, rhs_func, meas_func, x0, u0, dt=1, t_now=0, sig_y=0, sig_x=0):
        try:
            x_next = rhs_func(x0, u0)        
        except:
            raise ValueError("Invalid rhs_func. The rhs_func must take two arguments: x and u.")
        try:
            y = meas_func(x0, u0)        
        except:
            raise ValueError("Invalid meas_func. The meas_func function must take two arguments: x and u.")
        
        if x0.shape != x_next.shape:
            raise ValueError("x0 and x_next must have the same shape.")

        self.rhs_func = rhs_func
        self.meas_func = meas_func 

        self.n_x = x0.shape[0]
        self.n_u = u0.shape[0]
        self.n_y =  y.shape[0]

        if isinstance(sig_y, (float, int)):
            self.sig_y = sig_y*np.ones((self.n_y,1))
        elif isinstance(sig_y, np.ndarray):
            self.sig_y = sig_y.reshape((self.n_y,1)) 
        if isinstance(sig_x, (float, int)):
            self.sig_x = sig_x*np.ones((self.n_x,1))
        elif isinstance(sig_x, np.ndarray):
            self.sig_x = sig_x.reshape((self.n_x,1)) 

        self.dt = dt
        self.reset(x0=x0, t_now=t_now)

    def make_step(self,u):
        """
        Run a simulation step by passing the current input.
        Returns the current measurement y.
        """
        if u.shape != (self.n_u,1):
            raise ValueError("Input must be a column vector of size ({}x1).".format(self.n_u))

        # Store initial x and time
        self._x.append(self.x0)
        self._time.append(self.t_now)

        # Update measurement
        y = self.meas_func(self.x0, u)
        v = self.sig_y*np.random.randn(self.n_y,1)
        y = y + v

        # Update state and time
        self.x0 = self.rhs_func(self.x0, u)
        w = self.sig_x*np.random.randn(self.n_x, 1)
        self.x0 += w

        self.t_now += self.dt

        # Store input and measurement
        self._u.append(u)
        self._y.append(y)

        return y

    def reset(self, x0=None, t_now = 0):
        """Initialize system and clear history.
        """

        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros((self.n_x,1))

        self._x = []
        self._u = []
        self._y = []
        self.t_now = t_now
        self._time = []


    def narx_io(self, l, delta_y_out = False, return_type = 'numpy'):
        """ Generates NARX input and output data from stored data.
        The NARX input is structured as [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k)].
        The NARX output is structured as [y(k+1)].

        Args:
            l (int): Number of past inputs and outputs to include in the NARX input.

        Returns:
            tuple: (NARX input, NARX output)

        Raises:
            ValueError: If the number of stored inputs and outputs is smaller than l.
            TypeError: If l is not an integer.

        """
        if not isinstance(l, int):
            raise TypeError('l must be an integer.')
        if self.time.size < l:
            raise ValueError("Not enough data to generate NARX input and output. Run make_step at least {} times.".format(l))

        narx_in = []
        for k in range(l):
            narx_in.append(self.y[k:-l+k])
        for k in range(l):
            narx_in.append(self.u[k:-l+k])

        if delta_y_out:
            narx_out = self.y[l:]-self.y[l-1:-1]
        else:
            narx_out = self.y[l:]

        if return_type == 'numpy':
            narx_in = np.concatenate(narx_in, axis=1)

        elif return_type == 'pandas':
            df_y = pd.concat(
                {f'y_{k}' :pd.DataFrame(narx_in[k]) for k in range(l)},
                axis= 1
            )
            df_u = pd.concat(
                {f'u_{i}' :pd.DataFrame(narx_in[l+k]) for i in range(l)},
                axis= 1
            )
            narx_in = pd.concat((df_y, df_u),axis=1)
            narx_out = pd.DataFrame(narx_out)

        elif return_type == 'list':
            pass 
        else:
            raise AttributeError(f'Return type {return_type} is invalid. Must choose from [list, numpy, pandas]')

        return narx_in, narx_out

    @property
    def x(self):
        return np.concatenate(self._x,axis=1).T

    @x.setter
    def x(self, *args):
        raise Exception('Cannot set x directly.')

    @property
    def u(self):
        return np.concatenate(self._u,axis=1).T

    @u.setter
    def u(self, *args):
        raise Exception('Cannot set u directly.')

    @property
    def y(self):
        return np.concatenate(self._y,axis=1).T

    @y.setter
    def y(self, *args):
        raise Exception('Cannot set y directly.')

    @property
    def time(self):
        return np.array(self._time)

    @time.setter
    def time(self, *args):
        raise Exception('Cannot set time directly.')
    
class LTISystem(System):
    """
    Helper class to simulate linear discrete time dynamical systems in state-space form.
    Initiate with system matrices: A,B,C,D such that:

    x_next = A@x + B@u + w_x
    y      = C@x + D@u + w_y

    Passing D is optional.
    Passing x0 is optional (will results to all zero per default).
    w_x and w_y are zero mean Gaussian noise with standard deviation sig_x and sig_y respectively.

    - Run a simulation step with :py:meth:`make_step` method.
    - Reset (clear history) with :py:meth:`reset` method.
    - Get the current state with :py:attr:`x` attribute (similar for inputs :py:attr:`u` and measurements :py:attr:`y`).

    Args: 
        A (np.ndarray): System matrix of shape (n_x, n_x).
        B (np.ndarray): Input matrix of shape (n_x, n_u).
        C (np.ndarray): Output matrix of shape (n_y, n_x).
        D (np.ndarray): Feedthrough matrix of shape (n_y, n_u). Defaults to None.
        x0 (np.ndarray): Initial state of shape (n_x, 1). Defaults to None.
        dt (float): Time step. Defaults to 1.
        t_now (float): Current time. Defaults to 0.
        sig_y (float): Standard deviation of measurement noise. Defaults to 0.
        sig_x (float): Standard deviation of process noise. Defaults to 0.
    """
    def __init__(self,A,B,C, D=None, x0=None, u0=None, dt=1, t_now=0, sig_y = 0, sig_x=0):
        A = A
        B = B
        C = C

        if x0 is None: 
            x0 = np.zeros((A.shape[0],1))
        if u0 is None: 
            u0 = np.zeros((B.shape[1],1))

        if D is None:
            D = np.zeros((C.shape[0], B.shape[1]))

        def meas_func(x,u):
            return C@x + D@u

        def rhs_func(x,u):
            return A@x + B@u

        super().__init__(rhs_func, meas_func, x0=x0, u0=u0, dt=dt, t_now=t_now, sig_y=sig_y, sig_x=sig_x) 


class NARX(System):
    """
    In a NARX system, the state is represented by:
    [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
    and for the input we have u(k).

    """

    def __init__(self, narx_func, l, y0, u0, dt=1, t_now=0):
        self.narx_l = l
        self.n_y = y0.shape[0]
        self.n_u = u0.shape[0]

        x0 = np.concatenate(l*[y0] + (l-1)*[u0], axis=0)

        # Splitting index to deconstruct NARX input.
        self.narx_splitter = np.cumsum(l*[self.n_y] + (l-1)*[self.n_u])[:-1]

        def rhs_func(narx_state,u):
            # Split narx state into 
            # [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
            narx_state_split = np.split(narx_state, self.narx_splitter, axis=0)
            y_list = narx_state_split[:self.narx_l]
            u_list = narx_state_split[self.narx_l:]
            # Create NARX input by adding u(k) to the end.
            narx_input = np.concatenate((*y_list, *u_list, u), axis=0)
            # Evaluate NARX function
            y_next = narx_func(narx_input)

            # Update NARX state
            y_list.pop(0)
            y_list.append(y_next)

            u_list.pop(0)
            u_list.append(u)

            narx_state_next = np.concatenate((*y_list, *u_list), axis=0)
            # Return updated NARX state
            return narx_state_next

        def meas_func(narx_state, u):
            # Split narx state into 
            # [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
            narx_state_split = np.split(narx_state, self.narx_splitter, axis=0)
            # Extract y(k)
            y = narx_state_split[self.narx_l-1]

            return y            
            
        super().__init__(rhs_func, meas_func, x0=x0, u0=u0, dt=dt, t_now=t_now, sig_y=0, sig_x=0) 


        def get_AB_ARX(self, W, narx_out_dy = False):
            """
            For a (N)ARX model with history length l, state:
            [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]
            and input u(k), we can write the state equation as:
            
            x(k+1) = A_ARX x(k) + B_ARX u(k)

            assuming that the NARX transition function is linear with:

            y(k+1) = W_x [y(k-l), y(k-l+1), ..., y(k), u(k-l), u(k-l+1), ..., u(k-1)]^T + W_u u(k)

            and W = [W_x, W_u] is the NARX weight matrix, this function returns A_ARX and B_ARX.

            If, the NARX transition function returns the deltay instead of y, we can write the state equation,
            please set narx_out_dy = True.

            """

            if not isinstance(W, np.ndarray):
                raise TypeError(f"W must be a numpy array, but is {type(W)}.")
            if W.ndim != 2:
                raise ValueError(f"W must be a 2D array, but is {W.ndim}D.")

            shape_w_should = (self.n_y, self.n_y*self.narx_l + self.n_u*(self.narx_l-1))
            shape_w_is = W.shape
            if shape_w_should != shape_w_is:
                raise ValueError(f"W has wrong shape. Shape must be {shape_w_should} but is {shape_w_is}.")

            # Split W into W_x and W_u
            W_y, W_u = np.split(W, [self.n_y*self.narx_l], axis=1)

            # Consider the case where the NARX transition function returns the deltay instead of y.
            if narx_out_dy:
                W_y[-self.n_y:, -self.n_y:] = np.eye(self.n_y)

            A_ARX = np.concatenate(
                [np.zeros(((self.narx_l-1)*self.n_y, self.n_y)), np.eye((self.narx_l-1)*self.narx_l)],
                axis=1,
            )
            A_ARX = np.concatenate(
                [A_ARX, W_y],
                axis=0,
            )
            B_ARX = W_u

            return A_ARX, B_ARX


def random_u(u0, switch_prob=0.5, max_amp=np.pi):
    # Hold the current value with 80% chance or switch to new random value.
    u_next = (0.5-np.random.rand(u0.shape[0],1))*max_amp # New candidate value.
    switch = np.random.rand() >= (1-switch_prob) # switching? 0 or 1.
    u0 = (1-switch)*u0 + switch*u_next # Old or new value.
    return u0



def test():
    A = np.array([[1,1],[0,1]])
    B = np.array([[0],[1]])
    C = np.array([[1,0]])

    sys = LTISystem(A,B,C, dt=0.1, sig_x=0.1, sig_y=0.1)

def test_arx():
    """
    Test ARX/ NARX system class. 

    1. Create simple LTI system
    2. Generate NARX i/o data from LTI system
    3. Linear regression to estimate ARX system parameter
    4. Create function with linear ARX regression parameters representing the ARX/NARX model
    5. Create NARX system with ARX function
    6. Compare NARX system output to LTI system output
    
    """ 

    # 1. Create simple LTI system
    sys_dc = sio.loadmat('sys_dc.mat')
    A = sys_dc['A_dc']
    B = sys_dc['B_dc']
    C = sys_dc['C']
    D = sys_dc['D']

    sig_y = np.array([1e-2, 1e-2, 1e-2])
    sig_x = 1e-4

    sys = LTISystem(A,B,C,D, sig_y=sig_y, sig_x=sig_x)

    # 2. Generate NARX i/o data from LTI system
    u0 = np.zeros((2,1))
    sym_steps = 500
    l = 4
    max_amp = np.pi
    cooldown = 0.2


    for k in range(sym_steps,):
        if k<sym_steps*(1-cooldown):
            u0 = random_u(u0, switch_prob=0.4, max_amp=max_amp)
        else:
            u0 = np.zeros((2,1))
        sys.make_step(u0)

    narx_data = sys.narx_io(l=l, delta_y_out=False, return_type = 'numpy')

    # 3. Linear regression to estimate ARX system parameter
    W = np.linalg.inv(narx_data[0].T@narx_data[0])@narx_data[0].T@narx_data[1]

    # 4. Create function with linear ARX regression parameters representing the ARX/NARX model
    def narx_func(narx_input):
        return W.T@narx_input

    # 5. Create NARX system with ARX function
    narx_sys = NARX(narx_func, l, y0 = np.zeros((3,1)), u0 = np.zeros((2,1)))

    # 6. Compare NARX system output to LTI system output
    sys.reset()
    for k in range(100):
        if k<sym_steps*(1-cooldown):
            u0 = random_u(u0, switch_prob=0.4, max_amp=max_amp)
        else:
            u0 = np.zeros((2,1))
        narx_sys.make_step(u0)
        sys.make_step(u0)
    

    fig, ax = plt.subplots(3,1, sharex=True)
    for k in range(3):
        ax[k].plot(narx_sys.y[:,k], label='narx')
        ax[k].plot(sys.y[:,k], label='sys')
    
    ax[0].legend()
    plt.show(block=True)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io as sio

    test()
    test_arx()