import torch
import numpy as np
import scipy

# Function to generate input signal based on the chosen option
def input(t, u_option):
    """
    Generates an input signal based on the specified option.

    Args:
    - t (numpy.ndarray): Time array.
    - u_option (str): Option for generating the input signal.

    Returns:
    - y (numpy.ndarray): Generated input signal.
    """

    # Randomly initialize parameters for different options
    a = np.random.uniform(0.1, 0.3, 1)
    b = np.random.uniform(1, 5, 1)
    c = np.random.uniform(0, 5, 1)
    sign = 1 if np.random.random() < 0.5 else -1

    # Generate input signal based on the chosen option
    if u_option == "noise":
        y = np.random.normal(0, 0.1, len(t))
    elif u_option == "tanh":
        y = sign * a * np.tanh((t - 10) / 5)
        y[200:400] = 0.1 * np.cos(2 * t[200:400]) ** 2
        y[400::] = 0.1 * np.sin(t[400::] / 2)
    elif u_option == "sin":
        y = 0.1 * np.sin(2 * t)
    elif u_option == "rand_sin":
        y = a * np.sin(b * t + c)
    elif u_option == "random_walk":
        steps = np.random.uniform(-0.05, 0.05, len(t))
        steps[0] = np.random.uniform(-0.2, 0.2, 1)[0]
        walk = np.cumsum(steps)
        walk = 0.3 * (walk / np.max(abs(walk)))
        y = walk
    elif u_option == "cos":
        y = 0.1 * np.cos(t)
    elif u_option == "test":
        y = a * (np.cos(t ** 2) + np.sin(2 * t))
    elif u_option == "none":
        y = 0 * np.cos(t)
    elif u_option == "weird":
        y =  0 * np.cos(t)
        y[100:200] = y[100:200] + a  * np.sin(t[100:200] * b / 10)
        y[200:300] = y[200:300] + a  * t[200:300] / 30
        y[300::] = y[299]
        y[300:400] = - a  * t[300:400] / 60
        y[500::] = - a  * np.cos(t[500::]*b)
        
    elif u_option == "steps":
      y = 0 * np.cos(t)
      for t in range(1,6):
        y[100*t:100*t+100] = np.ones(100)*np.random.uniform(-0.2, 0.2, 1)[0] #y[100*t -100 : 100*t] + 


    return y


# Function defining the dynamic system
def func(y, t, u):
    """
    Defines the dynamic system.

    Args:
    - y (numpy.ndarray): State variables.
    - t (float): Time.
    - u (float): Input signal.

    Returns:
    - numpy.ndarray: Derivatives of the state variables.
    """
    f = 1  # Placeholder for system dynamics (can be modified as needed)
    return np.array([y[1], 1 / f * (-np.sin(y[0]) - 1 / 7 * y[1] + u)])  # Returns derivatives


# Function to generate training and test data
def get_data(x0=np.pi / 4, y0=0.1, use_fixed_init=False, t0=0, t1=30, time_steps=1000, num_of_inits=1,
             normalize=True, add_noise=False, u_option="noise", set_seed=True):
    """
    Generates training and test data for the dynamic system.

    Args:
    - x0 (float): Initial condition for the first state variable.
    - y0 (float): Initial condition for the second state variable.
    - use_fixed_init (bool): Whether to use a fixed initial condition for all trajectories.
    - t0 (float): Start time.
    - t1 (float): End time.
    - time_steps (int): Number of time steps.
    - num_of_inits (int): Number of initial conditions to generate.
    - normalize (bool): Whether to normalize the data.
    - add_noise (bool): Whether to add noise to the generated trajectories.
    - u_option (str): Option for generating the input signal.
    - set_seed (bool): Whether to set the random seed for reproducibility.

    Returns:
    - input_data (torch.Tensor): Input data for training.
    - test_data (torch.Tensor): Test data for evaluation.
    - time (torch.Tensor): Time array.
    - initial_values (torch.Tensor): Initial conditions used for generating the trajectories.
    - input_data_w_time (torch.Tensor): Input data including time information.
    """

    if set_seed:
        np.random.seed(seed=42)

    # Set initial conditions
    y0 = [x0, y0]

    # Generate time array
    t = np.linspace(t0, t1, num=time_steps)

    # Generate initial conditions for multiple trajectories
    y0_list_w = np.random.uniform(-np.pi / 2, np.pi / 2, size=(num_of_inits, 1))
    y0_list_o = np.random.uniform(-0.2, 0.2, size=(num_of_inits, 1))
    y0_list = np.concatenate([y0_list_w, y0_list_o], axis=1)

    if use_fixed_init:
        y0_list = [y0]

    # Initialize lists to store trajectory data
    trajectory_list = []
    input_data_list = []
    input_data_w_time_list = []

    # Generate trajectories for each initial condition
    for i, y0 in enumerate(y0_list):

        out = np.zeros((len(t), 2))
        out[0] = y0_list[i]  # Set initial state

        u = input(t, u_option)  # Generate input signal

        # Integrate the system dynamics to generate trajectory
        for i in range(1, len(t)):
            t_span = [t[i - 1], t[i]]
            z = scipy.integrate.odeint(func, out[i - 1], t_span, args=(u[i],))
            out[i] = z[1]

        if add_noise:
            out += np.random.normal(0, 0.02, size=(len(t), 2))

        # Convert trajectory data to torch tensors and concatenate input signal
        out = torch.tensor(out)
        u = torch.tensor(u).view(1, len(t))
        u = u.transpose(0, 1)
        time = torch.tensor(t)
        time = time.view(1, len(t))
        time = time.transpose(0, 1)
        trajectory_list.append(out)
        input_data_list.append(torch.cat((u, out), dim=-1))
        input_data_w_time_list.append(torch.cat((time, u, out), dim=-1))

    # Stack trajectory data into tensors
    test_data = torch.stack(trajectory_list)
    input_data = torch.stack(input_data_list)
    input_data_w_time = torch.stack(input_data_w_time_list)

    # Normalize data if specified
    if normalize:
        test_data = torch.nn.functional.normalize(test_data)
        input_data = torch.nn.functional.normalize(input_data)
        input_data_w_time = torch.nn.functional.normalize(input_data_w_time)

    return input_data, test_data, time, torch.tensor(y0_list), input_data_w_time
