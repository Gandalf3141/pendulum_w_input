# python
# """
# LSTM Model for Derivative Estimation with Dynamic Systems

# This code defines an LSTM (Long Short-Term Memory) model to estimate derivatives in dynamic systems. It utilizes PyTorch for defining and training the model. The main components of the code include:

# Packages:
# - Matplotlib for plotting
# - Pandas for data manipulation
# - Torch for deep learning operations
# - Torchdyn for solving ordinary differential equations
# - Icecream for debugging purposes
# - tqdm for progress monitoring
# - Scipy for scientific computing
# - cProfile and pstats for profiling

# Model Architecture:
# - The LSTMmodel class defines the LSTM model with configurable input size, hidden size, output size, and number of layers.

# Training and Testing:
# - The train function trains the model using input data. It supports options for using autograd and ODE steps.
# - The test function evaluates the model's performance on test data. It supports options for plotting results and using autograd.
# - The main function orchestrates training, testing, and model saving.

# Settings:
# - Various parameters such as window size, time settings, and dataset characteristics are configurable.

# Author: [Author Name]
# Date: [Date]
# """
# ```


# Importing necessary libraries
from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import numpy as np
import torch.autograd.gradcheck
from torchdyn.numerics import odeint
from icecream import ic
from tqdm import tqdm
import scipy
from itertools import chain
import cProfile
import pstats
from get_data_exp import get_data
import logging
import os
import time

# Define the LSTM model with two hidden layers
torch.set_default_dtype(torch.float64)
device = "cpu"


class LSTMmodel(nn.Module):
    """
    LSTM model class for derivative estimation.
    """

    def __init__(self, input_size, hidden_size, out_size, layers):
        """
        Initialize the LSTM model.

        Args:
        - input_size: Size of input
        - hidden_size: Size of hidden layer
        - out_size: Size of output
        - layers: Number of layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, seq):
        """
        Forward pass through the LSTM model.

        Args:
        - seq: Input sequence

        Returns:
        - pred: Model prediction
        - hidden: Hidden state
        """
        lstm_out, hidden = self.lstm(seq)
        pred = self.linear(lstm_out.view(len(seq), -1))

        return pred, hidden


def slice_batch(batch, window_size=1, use_autograd=False):
    """
    Slice the input data into batches for training.

    Args:
    - batch: Input data batch
    - window_size: Size of the sliding window

    Returns:
    - List of sliced batches
    """
    l = []
    if use_autograd:
        for i in range(len(batch) - window_size):
            l.append((batch[i:i+window_size, 0:2],
                     batch[i+1:i+window_size+1, 1]))
        return l
    else:
        for i in range(len(batch) - window_size):
            l.append((batch[i:i+window_size, :],
                     batch[i+1:i+window_size+1, 1:]))
        return l


def train(input_data, model, ws=1, odestep=False, use_autograd=False):
    """
    Train the LSTM model using input data.

    Args:
    - input_data: Input data for training
    - model: LSTM model to be trained
    - ws: Window size
    - odestep: Option for using ODE steps
    - use_autograd: Option for using autograd

    Returns:
    - Mean loss over all batches
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    total_loss = []

    for batch in input_data:

        input = slice_batch(batch, ws, use_autograd)

        batch_loss = 0

        for inp, label in input:  # inp = (u, x) label = x

            if use_autograd:

                output, _ = model(inp)

                # output.backward()
                # c = torch.stack((output[-1], output[-1].grad), dim=-1)
                # ic(c)
                out = inp[:, 1] + output[-1]

            else:
                output, _ = model(inp)

                if odestep:
                    out = inp[:, 1:] + output
                else:
                    out = output[-1]
                    out = output[-1].view(1, output[-1].size(dim=0))

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            batch_loss += loss.detach().numpy()

        total_loss.append(batch_loss/len(batch))

    return np.mean(total_loss)


# def test(test_data, time, model, plot_opt=False, ws=1, steps=200, odestep=False, use_autograd=False):
#     """
#     Test the trained LSTM model using test data.

#     Args:
#     - test_data: Test data
#     - time: Time data
#     - model: Trained LSTM model
#     - plot_opt: Option for plotting results
#     - ws: Window size
#     - steps: Number of steps
#     - odestep: Option for using ODE steps
#     - use_autograd: Option for using autograd

#     Returns:
#     - Mean test loss
#     - Mean derivative test loss
#     """
#     model.eval()
#     loss_fn = nn.MSELoss()
#     test_loss = 0
#     test_loss_deriv = 0

#     for x in test_data:
#         with torch.inference_mode():

#             pred = torch.zeros((steps, 3))
#             pred_next_step = torch.zeros((steps, 3))

#             if ws > 1:
#                 pred[0:ws, :] = x[0:ws, :]
#                 pred[:, 0] = x[:, 0]
#                 pred_next_step[0:ws, :] = x[0:ws, :]
#                 pred_next_step[:, 0] = x[:, 0]
#             else:
#                 pred[0, :] = x[0, :]
#                 pred[:, 0] = x[:, 0]
#                 pred_next_step[0, :] = x[0, :]
#                 pred_next_step[:, 0] = x[:, 0]

#             for i in range(len(x) - ws):

#                 if use_autograd:
#                     out, _ = model(pred[i:i+ws, 0:2])

#                     if odestep:
#                         pred[i+ws, 1] = pred[i+ws-1, 1] + out[-1, :]
#                         pred_next_step[i+ws, 1] = x[i+ws-1, 1] + out[-1, :]
#                     else:
#                         pred[i+ws, 1:] = out[-1]
#                         outt, _ = model(x[i:i+ws, :])
#                         pred_next_step[i+ws, 1:] = outt[-1]
#                 else:
#                     out, _ = model(pred[i:i+ws, :])

#                     if odestep:
#                         pred[i+ws, 1:] = pred[i+ws-1, 1:] + out[-1, :]
#                         pred_next_step[i+ws, 1:] = x[i+ws-1, 1:] + out[-1, :]
#                     else:
#                         pred[i+ws, 1:] = out[-1]
#                         outt, _ = model(x[i:i+ws, :])
#                         pred_next_step[i+ws, 1:] = outt[-1]

#             if use_autograd:

#                 pred[i+ws, 2] = torch.gradient(pred[:, 1], spacing=(time,))

#             test_loss += loss_fn(pred[:, 1], x[:, 1]).detach().numpy()
#             test_loss_deriv += loss_fn(pred[:, 2], x[:, 2]).detach().numpy()

#             if plot_opt:
#                 plt.plot(time, pred.detach().numpy()[
#                          :, 1], color="red", label="pred")
#                 plt.plot(time, pred_next_step.detach().numpy()[
#                          :, 1], color="green", label="next step from data")
#                 plt.plot(time, x.detach().numpy()[
#                          :, 1], color="blue", label="true", linestyle="dashed")

#                 plt.grid()
#                 plt.legend()
#                 plt.show()

#     return np.mean(test_loss), np.mean(test_loss_deriv)

def test(test_data, time, model, plot_opt=False, ws=1, steps=200, odestep=False, use_autograd=False,  plot_derivative=False, error_plot=False, mixed_input=False):

    model.eval()
    loss_fn = nn.MSELoss()

    test_loss = []
    test_loss_deriv = []

    for x in test_data:
        with torch.inference_mode():

            pred = torch.zeros((steps, 3))
            pred_next_step = torch.zeros((steps, 3))

            if ws > 1:
                pred[0:ws, :] = x[0:ws, :]
                pred[:, 0] = x[:, 0]
                pred_next_step[0:ws, :] = x[0:ws, :]
                pred_next_step[:, 0] = x[:, 0]
            else:
                pred[0, :] = x[0, :]
                pred[:, 0] = x[:, 0]
                pred_next_step[0, :] = x[0, :]
                pred_next_step[:, 0] = x[:, 0]
                if mixed_input:
                    pred_next_step[:, 1] = x[:, 1]

            for i in range(len(x) - ws):

                if use_autograd:

                    out, _ = model(pred[i:i+ws, 0:2])
                    out2, _ = model(x[i:i+ws, 0:2])

                    pred[i+ws, 1] = pred[i+ws-1, 1] + out[-1, :]
                    pred_next_step[i+ws, 1] = x[i+ws-1, 1] + out2[-1, :]

                else:

                    out, _ = model(pred[i:i+ws, :])
                    out2, _ = model(x[i:i+ws, :])

                    if odestep:

                        # x[k+1] = x[k]              + NN(x[k])
                        pred[i+ws, 1:] = pred[i+ws-1, 1:] + out[-1, :]
                        pred_next_step[i+ws, 1:] = x[i+ws-1, 1:] + out2[-1, :]

                    else:
                        pred[i+ws, 1:] = out[-1]
                        outt, _ = model(x[i:i+ws, :])
                        pred_next_step[i+ws, 1:] = outt[-1]

            if use_autograd:
                pred[i+ws, 2] = torch.gradient(pred[:, 1], spacing=(time,))

            test_loss.append(loss_fn(pred[:, 1], x[:, 1]).detach().numpy())
            test_loss_deriv.append(
                loss_fn(pred[:, 2], x[:, 2]).detach().numpy())

            if plot_opt:

                greek_letterz = [chr(code) for code in range(
                    945, 970)]  # 7 theta 24 omega

                fig, axs = plt.subplots(2, 1, figsize=(16, 9))

                # Plot the first function
                k = 1 if plot_derivative else 0
                axs[0].plot(time, pred.detach().numpy()[
                            :, 1+k], color="red", label="pred")
                axs[0].plot(time, pred_next_step.detach().numpy()[
                            :, 1+k], color="green", label="next_step_pred", linestyle="dashed")
                axs[0].plot(time, x.detach().numpy()[:, 1+k],
                            color="blue", label="true")
                axs[0].set_xlabel('time [t]')
                axs[0].set_ylabel(f"{greek_letterz[7]}(t)")
                axs[0].set_title('damped pendulum with input')
                axs[0].legend()
                axs[0].grid()

                # Plot the second function
                axs[1].plot(time, pred.detach().numpy()[:, 0],
                            color="green", label="input")
                axs[1].set_xlabel('time')
                axs[1].set_ylabel('u(t)')
                axs[1].set_title('input signal')
                axs[1].legend()
                plt.tight_layout()
                plt.grid()

                if not error_plot:
                    plt.show()

            if error_plot:

                error = torch.zeros((steps, 4))
                cumulative_error = torch.zeros((steps, 4))

                for j in range(steps):
                    error[j, 0] = abs(x[j, 1] - pred[j, 1])
                    error[j, 1] = abs(x[j, 2] - pred[j, 2])
                    error[j, 2] = abs(x[j, 1] - pred_next_step[j, 1])
                    error[j, 3] = abs(x[j, 2] - pred_next_step[j, 2])

                for k in range(4):
                    cumulative_error[:, k] = np.cumsum(error[:, k])

                fig, axs = plt.subplots(3, 1, figsize=(16, 9))

                # Plot the absolute errors
                axs[0].plot(time, error[:, 0], color="red", label="abs_error")
                axs[0].plot(time, error[:, 2], color="blue",
                            label="abs_error_nextstep")
                axs[0].set_xlabel('time [t]')
                axs[0].set_ylabel(
                    f"e(t) = {greek_letterz[7]}(t) - {greek_letterz[7]}_pred(t) ")
                axs[0].set_title('absolute error')
                axs[0].legend()
                axs[0].grid()

                axs[1].plot(time, error[:, 1], color="red", label="abs_error")
                axs[1].plot(time, error[:, 3], color="blue",
                            label="abs_error_nextstep")
                axs[1].set_xlabel('time')
                axs[1].set_ylabel(
                    f"e(t) = {greek_letterz[24]}(t) - {greek_letterz[24]}_pred(t) ")
                axs[1].set_title('absolute error')
                axs[1].legend()

                # plot the cumulative errors
                axs[2].plot(time, cumulative_error[:, 0],
                            color="red", label="angle")
                axs[2].plot(time, cumulative_error[:, 1],
                            color="blue", label="angular velocity")
                axs[2].plot(time, cumulative_error[:, 2],
                            color="green", label="angle_nextstep")
                axs[2].plot(time, cumulative_error[:, 3],
                            color="yellow", label="angular velocity nextstep")

                axs[2].set_xlabel('time')
                axs[2].set_title('cumulative_error')
                axs[2].legend()
                # Adjust layout
                plt.tight_layout()
                plt.grid()
                plt.legend()
               # plt.figure()
                plt.show()

    return np.mean(test_loss), np.mean(test_loss_deriv)


def main():
    """
    Main function for training and testing the LSTM model.
    """
    # Check if the logging file exists
    log_file = 'training.log'
    filemode = 'a' if os.path.exists(log_file) else 'w'

    # Configure logging
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Define parameters
    window_size = 4
    start_time = 0
    stop_time = 30
    timesteps = 200
    num_of_inits = 200
    option_odestep = True
    losses = []
    use_autograd = False#ignore
    noise_factor = 6

    # Generate input data
    input_data, test_data, time, initial_values, input_data_w_time = get_data(x0=np.pi/4, y0=0.1, use_fixed_init=False, t0=start_time, t1=stop_time,
                                                                              time_steps=timesteps,
                                                                              num_of_inits=num_of_inits,
                                                                              normalize=False, add_noise=True,
                                                                              u_option="random_walk",
                                                                              set_seed=True,
                                                                              noise_factor=noise_factor)

    # input_data2, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=start_time, t1=stop_time,
    #                                                                                time_steps=timesteps, num_of_inits=50,
    #                                                                                      normalize=False, add_noise=False, u_option="const",  set_seed=True)
    # input_data = torch.cat((input_data, input_data2))

    # Split data into train and test sets
    train_size = int(0.9 * len(input_data))
    test_size = len(input_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(input_data, [train_size, test_size])

    # Take a slice of data for training
    slice_of_data = 80
    train_dataset = train_dataset[:][:, 0:slice_of_data, :]

    # Initialize the LSTM model
    if use_autograd:  # one output for theta
        model = LSTMmodel(input_size=2, hidden_size=5,out_size=1, layers=1).to(device)
    else:
        model = LSTMmodel(input_size=3, hidden_size=5,out_size=2, layers=1).to(device)

    epochs = 30

    for e in tqdm(range(epochs)):
        loss_epoch = train(train_dataset, model, ws=window_size,
                           odestep=option_odestep, use_autograd=use_autograd)
        losses.append(loss_epoch)
        if e % 2 == 0:
            print(f"Epoch {e}: Loss: {loss_epoch}")
    # Plot losses
    plt.plot(losses[1:])
    plt.show()

    # Save trained model
    file_name = f"mixed_input_experiment/Trained_lstms_exp/lstm_noisefactor{noise_factor}.pth"
    torch.save(model.state_dict(), file_name)

    # Test the model
    # test_dataset = test_dataset[:][:, :, :]
    #print(test(test_dataset, time, model, plot_opt=True, ws=window_size,
    #      steps=timesteps, odestep=option_odestep, use_autograd=use_autograd, plot_derivative=True, error_plot=False, mixed_input=False))

    # Log parameters
    logging.info(f"Epochs: {epochs}, Window Size: {window_size},\n Start Time: {start_time}, Stop Time: {stop_time}, Timesteps: {timesteps}, Number of Inits: {num_of_inits},\n Option for ODE Step: {option_odestep}")
    logging.info(f"\n {file_name}")
    logging.info(f"final loss {losses[-1]}")
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")

    return None


if __name__ == "__main__":
    main()

# These settings worked:
    #
    # input_data, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=start_time, t1=stop_time,
    #                                                                              time_steps=timesteps, num_of_inits=num_of_inits, normalize=False, add_noise=False, u_option="sin")
    #     window_size = 2
    # start_time = 0
    # stop_time = 30
    # timesteps = 200
    # num_of_inits = 100
    # option_odestep = True


# Add short and precise comments to the following python code. Describe functions, classes, loops similar things. The code:
