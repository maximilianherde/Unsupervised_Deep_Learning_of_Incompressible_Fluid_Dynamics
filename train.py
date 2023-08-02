import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from derivatives import toCuda, toCpu, params, first_derivative, laplacian, curl, vel_map, convert_grid
from derivatives import vector2HSV
from setups import Dataset
from Logger import Logger, t_step
from pde_cnn import get_Net
import cv2
from get_param import get_hyperparam
import wandb

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

run = wandb.init(project="MacPINN", entity="srl_ethz", name="plain")

mu = params.mu
rho = params.rho
dt = params.dt

# initialize fluid model
fluid_model = toCuda(get_Net(params))
fluid_model.train()

# initialize Optimizer
optimizer = Adam(fluid_model.parameters(), lr=params.lr)

# initialize Logger and load model / optimizer if according parameters were given
logger = Logger(get_hyperparam(params), use_csv=False,
                use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
    load_logger = Logger(get_hyperparam(
        params), use_csv=False, use_tensorboard=False)
    if params.load_optimizer:
        params.load_date_time, params.load_index = logger.load_state(
            fluid_model, optimizer, params.load_date_time, params.load_index)
    else:
        params.load_date_time, params.load_index = logger.load_state(
            fluid_model, None, params.load_date_time, params.load_index)
    params.load_index = int(params.load_index)
    print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

# initialize Dataset
dataset = Dataset(params.width, params.height, params.batch_size, params.dataset_size,
                  params.average_sequence_length, max_speed=params.max_speed, dt=params.dt, types=['DFG_benchmark'])


def loss_function(x):
    return torch.pow(x, 2)


# training loop
for epoch in range(params.load_index, params.n_epochs):

    for i in range(params.n_batches_per_epoch):
        v_cond, cond_mask, flow_mask, a_old, p_old = toCuda(dataset.ask())

        # convert v_cond,cond_mask,flow_mask to MAC grid
        v_cond = convert_grid(v_cond, "staggered")
        cond_mask_mac = (convert_grid(
            cond_mask.repeat(1, 2, 1, 1), "staggered") == 1).float()
        flow_mask_mac = (convert_grid(
            flow_mask.repeat(1, 2, 1, 1), "staggered") >= 0.5).float()

        v_old = curl(a_old)

        # predict new fluid state from old fluid state and boundary conditions using the neural fluid model
        a_new, p_new = fluid_model(a_old, p_old, flow_mask, v_cond, cond_mask)
        v_new = curl(a_new)

        # compute boundary loss
        loss_bound = torch.mean(loss_function(
            cond_mask_mac*(v_new-v_cond))[:, :, 1:-1, 1:-1], dim=(1, 2, 3))

        # explicit / implicit / IMEX integration schemes
        if params.integrator == "explicit":
            v = v_old
        if params.integrator == "implicit":
            v = v_new
        if params.integrator == "imex":
            v = (v_new+v_old)/2

        # compute loss for momentum equation
        loss_nav = torch.mean(loss_function(flow_mask_mac*(rho*((v_new[:, 1:2]-v_old[:, 1:2])/dt+v[:, 1:2]*first_derivative(v[:, 1:2], "x", "central")+0.5*(vel_map(v[:, 0:1], "x", "left")*first_derivative(v[:, 1:2], "y", "left")+vel_map(v[:, 0:1], "x", "right")*first_derivative(v[:, 1:2], "y", "right")))+first_derivative(p_new, "x", "left")-mu*laplacian(v[:, 1:2])))[:, :, 1:-1, 1:-1], dim=(1, 2, 3)) +\
            torch.mean(loss_function(flow_mask_mac*(rho*((v_new[:, 0:1]-v_old[:, 0:1])/dt+v[:, 0:1]*first_derivative(v[:, 0:1], "y", "central")+0.5*(vel_map(v[:, 1:2], "y", "left")*first_derivative(
                v[:, 0:1], "x", "left")+vel_map(v[:, 1:2], "y", "right")*first_derivative(v[:, 0:1], "x", "right")))+first_derivative(p_new, "y", "left")-mu*laplacian(v[:, 0:1])))[:, :, 1:-1, 1:-1], dim=(1, 2, 3))

        regularize_grad_p = torch.mean(
            (first_derivative(p_new, "x", "right")**2+first_derivative(p_new, "y", "right")**2)[:, :, 2:-2, 2:-2], dim=(1, 2, 3))

        # optional: additional loss to keep mean of a / p close to 0
        loss_mean_a = torch.mean(a_new, dim=(1, 2, 3))**2
        loss_mean_p = torch.mean(p_new, dim=(1, 2, 3))**2

        loss = params.loss_bound*loss_bound + params.loss_nav*loss_nav + params.loss_mean_a * \
            loss_mean_a + params.loss_mean_p*loss_mean_p + \
            params.regularize_grad_p*regularize_grad_p

        loss = torch.mean(torch.log(loss))

        # compute gradients
        optimizer.zero_grad()
        # ignore the loss_multiplier (could be used to scale gradients)
        loss = loss*params.loss_multiplier
        loss.backward()

        # optional: clip gradients
        if params.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(
                fluid_model.parameters(), params.clip_grad_value)
        if params.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                fluid_model.parameters(), params.clip_grad_norm)

        # perform optimization step
        optimizer.step()

        p_new.data = (p_new.data-torch.mean(p_new.data, dim=(1, 2, 3)
                                            ).unsqueeze(1).unsqueeze(2).unsqueeze(3))  # normalize pressure
        a_new.data = (a_new.data-torch.mean(a_new.data, dim=(1, 2, 3)
                                            ).unsqueeze(1).unsqueeze(2).unsqueeze(3))  # normalize a

        # recycle data to improve fluid state statistics in dataset
        dataset.tell(toCpu(a_new), toCpu(p_new))

        # log training metrics
        if i % 16 == 0:
            loss = toCpu(loss).numpy()
            loss_bound = toCpu(torch.mean(loss_bound)).numpy()
            loss_nav = toCpu(torch.mean(loss_nav)).numpy()
            regularize_grad_p = toCpu(torch.mean(regularize_grad_p)).numpy()
            
            wandb.log(
                {
                    "epoch": epoch,
                    "batch": i * 16,
                    "loss": loss,
                    "boundary loss": loss_bound,
                    "equation loss": loss_nav,
                    "norm of grad p": regularize_grad_p,
                }
            )

            if i % 128 == 0:
                print(
                    f"{epoch}: i:{i}: loss: {loss}; loss_bound: {loss_bound}; loss_nav: {loss_nav};")

    # safe state after every epoch
    if params.log:
        logger.save_state(fluid_model, optimizer, epoch+1)
