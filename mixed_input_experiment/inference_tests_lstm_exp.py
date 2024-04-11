"""
LSTM Model for Dynamic System Derivative Estimation

This Python script implements an LSTM (Long Short-Term Memory) model for estimating derivatives in dynamic systems. The script includes the following components:

1. Packages: Importing necessary libraries such as Matplotlib, Pandas, Torch, and others.
2. LSTM Model Definition: Defines the architecture of the LSTM model with two hidden layers.
3. Testing Function: Evaluates the model's performance on test data and provides options for plotting results and error analysis.

Settings:
- The script includes settings for parameters such as window size, time settings, and dataset characteristics.
- These settings are used for configuring the testing process.

Main Function:
- The main function loads a pre-trained model and evaluates it on test data.

Note: The code is intended for educational and experimental purposes and may require modifications for specific use cases.

Author: Paul Strasser
Date: 07.04.2024
"""


from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import numpy as np
from torchdyn.numerics import odeint
from icecream import ic
from tqdm import tqdm
import scipy
from itertools import chain
import cProfile
import pstats
from get_data_exp import get_data

# Defining the LSTM model with two hidden layers
torch.set_default_dtype(torch.float64)
device = "cpu"


class LSTMmodel(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers):

        super().__init__()

        self.hidden_size = hidden_size

        self.input_size = input_size

        # ohne bias ändern 0er in input sequence nichts.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers)

        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, seq):

        lstm_out, hidden = self.lstm(seq)

        pred = self.linear(lstm_out.view(len(seq), -1))

        return pred, hidden


def test(test_data, time, model, plot_opt=False, ws=1, steps=600, error_plot=False, noise_factor=1):

    model.eval()
    loss_fn = nn.MSELoss()

    test_lossA = []
    test_loss_derivA = []
    test_lossB = []
    test_loss_derivB = []
    test_lossC = []
    test_loss_derivC = []

    for x in test_data:
        with torch.inference_mode():
            
            #put noise over true solution x
            x_noise = x.clone()
            x_noise[:,1] += np.random.normal(0, 0.02, len(time))
            x_noise[:,2] += noise_factor * np.random.normal(0, 0.02, len(time))

            pred = torch.zeros((steps, 3))
            pred_mixed = torch.zeros((steps, 3))
            pred_next_step = torch.zeros((steps, 3))
            #set u(t)
            pred[:, 0] = x[:, 0] 
            pred_mixed[:, 0] = x[:, 0] 
            pred_next_step[:, 0] = x[:, 0]

            #initial value based on windowsize
            if ws > 1:
                pred[0:ws, :] = x_noise[0:ws, :]            
                pred_mixed[0:ws, :] = x_noise[0:ws, :]
                pred_next_step[0:ws, :] = x_noise[0:ws, :]
            else:
                pred[0, :] = x_noise[0, :] 
                pred_mixed[0, :] = x_noise[0, :]
                pred_next_step[0, :] = x_noise[0, :]

            for i in range(len(x_noise) - ws):

                out, _ = model(pred[i:i+ws, :])
                out2, _ = model(torch.cat((x_noise[i:i+ws, 0:2], pred_mixed[i:i+ws, 2].view(ws,1)), dim=1))
                out3, _ = model(x_noise[i:i+ws, :])

                # x[k+1] = x[k]              + NN(x[k])
                pred[i+ws, 1:] = pred[i+ws-1, 1:] + out[-1, :]

                pred_mixed[i+ws, 1:] = torch.tensor([x_noise[i+ws-1, 1], pred_mixed[i+ws-1, 2]]) + out2[-1, :]

                pred_next_step[i+ws, 1:] = x[i + ws-1, 1:] + out3[-1, :]


            test_lossA.append(loss_fn(pred[:, 1], x[:, 1]).detach().numpy())
            test_loss_derivA.append(loss_fn(pred[:, 2], x[:, 2]).detach().numpy())

            test_lossB.append(loss_fn(pred_mixed[:, 1], x[:, 1]).detach().numpy())
            test_loss_derivB.append(loss_fn(pred_mixed[:, 2], x[:, 2]).detach().numpy())

            test_lossC.append(loss_fn(pred_next_step[:, 1], x[:, 1]).detach().numpy())
            test_loss_derivC.append(loss_fn(pred_next_step[:, 2], x[:, 2]).detach().numpy())

            if plot_opt:

                greek_letterz = [chr(code) for code in range(
                    945, 970)]  # 7 theta 24 omega

                fig2, axs = plt.subplots(3, 1, figsize=(16, 9))

                # Plot the first function
                axs[0].plot(time, pred.detach().numpy()[:, 1], color="red", label="pred")
                axs[0].plot(time, pred_mixed.detach().numpy()[:, 1], color="purple", label="pred_mixed")
                axs[0].plot(time, pred_next_step.detach().numpy()[:, 1], color="green", label="next_step_pred", linestyle="dashed")
                axs[0].plot(time, x.detach().numpy()[:, 1],color="blue", label="true")
                axs[0].set_xlabel('time [t]')
                axs[0].set_ylabel(f"{greek_letterz[7]}(t)")
                axs[0].set_title('damped pendulum with input')
                axs[0].legend()
                axs[0].grid()

                axs[1].plot(time, pred.detach().numpy()[:, 2], color="red", label="pred")
                axs[1].plot(time, pred_mixed.detach().numpy()[:, 2], color="purple", label="pred_mixed")
                axs[1].plot(time, pred_next_step.detach().numpy()[:, 2], color="green", label="next_step_pred", linestyle="dashed")
                axs[1].plot(time, x.detach().numpy()[:, 2],color="blue", label="true")
                axs[1].set_xlabel('time [t]')
                axs[1].set_ylabel(f"{greek_letterz[24]}(t)")
                axs[1].set_title('damped pendulum with input')
                axs[1].legend()
                axs[1].grid()

                # Plot the second function
                axs[2].plot(time, pred.detach().numpy()[:, 0], color="green", label="input")
                axs[2].set_xlabel('time')
                axs[2].set_ylabel('u(t)')
                axs[2].set_title('input signal')
                axs[2].legend()
                plt.tight_layout()
                plt.grid()
                plt.show()

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

                fig3, axs = plt.subplots(3, 1, figsize=(16, 9))

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

    return (test_lossA, test_loss_derivA), (test_lossB, test_loss_derivB), (test_lossC, test_loss_derivC)


def main():

    window_size = 4
    start_time = 0
    stop_time = 90
    timesteps = 3 * 200
    num_of_inits = 100
    option_odestep = True
    use_autograd = False
    noise_factor = 10
    losses = []
    display_plots = False
    num_of_inits = 3 if display_plots else num_of_inits

    input_data, test_data, time, initial_values, input_data_w_time = get_data(x0=np.pi/4, y0=0.1, use_fixed_init=False,
                                                                               t0=start_time, t1=stop_time,
                                                                              time_steps=timesteps, 
                                                                              num_of_inits=num_of_inits,
                                                                              normalize=False, add_noise=False, 
                                                                              u_option="random_walk",  set_seed=False,
                                                                              ws_start_noise=window_size,
                                                                              noise_factor=noise_factor)

    train_size = 1
    test_size = len(input_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(input_data, [train_size, test_size])

    model = LSTMmodel(input_size=3, hidden_size=5,
                      out_size=2, layers=1).to(device)
    
    path = "mixed_input_experiment\Trained_lstms_exp\lstm_noisefactor.pth"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # this works good:  window_size = 4 / small model
    # model.load_state_dict(torch.load("lstm_wsize_4_smaller_net.pth", map_location=torch.device('cpu')))

    (test_lossA, test_loss_derivA), (test_lossB, test_loss_derivB), (test_lossC, test_loss_derivC)= test(test_dataset,
                                                                                                        time, model,
                                                                                                        plot_opt=display_plots,
                                                                                                        ws=1, 
                                                                                                        steps=600,
                                                                                                        error_plot=display_plots,
                                                                                                        noise_factor=10)

    print("calc done")

    if not display_plots:
        a=np.array(test_loss_derivA)
        b=np.array(test_loss_derivB)
        c=np.array(test_loss_derivC)
        perc = torch.mean(torch.tensor([100*(a/b-1) for a, b in zip(a, b)]))
        perc2 = torch.mean(torch.tensor([100*(a/c-1) for a, c in zip(a, c)]))

        fig, axs = plt.subplots(1, 1, figsize=(16, 9))
        axs.hist(a.flatten(),bins=30, alpha = 0.5, label="using only predictions")
        axs.hist(b.flatten(), bins=30, alpha = 0.5, label="using partly meassured data")
        axs.hist(c.flatten(), bins=30, alpha = 0.5, label="using fully meassured data")
        axs.axvline(a.mean(), color='blue', linestyle='dashed', linewidth=1, label=f"Mean {np.round(a.mean(),7)}: using only predictions")
        axs.axvline(b.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean {np.round(b.mean(),7)}: using partly meassured data")
        axs.axvline(c.mean(), color='purple', linestyle='dashed', linewidth=1, label=f"Mean {np.round(c.mean(),7)}: using fully meassured data")
        axs.set_title(f"Error in prediction of angular velocity \n num of initial values = {num_of_inits}, \n Average difference in Errors A vs B {np.round(perc,2)}%, \n Average difference in Errors A vs C {np.round(perc2,2)}%")
        axs.set_xlabel("mean squared error")
        axs.grid()
        axs.legend()

        # axs[1].plot(perc)
        # axs[1].plot(torch.ones_like(perc) * torch.mean(perc),
        #             label=f"Average {np.round(torch.mean(perc),2)} %")
        # axs[1].set_title("Percentage difference")
        # axs[1].grid()
        # axs[1].set_ylabel("%")
        # axs[1].set_xlabel("initial values")
        plt.tight_layout()
        plt.legend()
        plt.show()

    return None


if __name__ == "__main__":
    main()

# These settings worked:
    #
    # input_data, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=start_time, t1=stop_time,
    #                                                                              time_steps=timesteps, num_of_inits=num_of_inits, normalize=False, add_noise=False, u_option="sin")
    # window_size = 2
    # start_time = 0
    # stop_time = 30
    # timesteps = 200
    # num_of_inits = 100
    # option_odestep = True

    #autograd try
    
            # if use_autograd:
            #     pred[:, 2] = torch.gradient(pred[:, 1],
            #                                 spacing=(time.view(len(time)),))[0]
