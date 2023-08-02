import torch
from torch.optim import Adam
import numpy as np
from derivatives import toCuda, toCpu, params, curl, convert_grid
from setups import Dataset
from Logger import Logger
from pde_cnn import get_Net
from get_param import get_hyperparam
from loss import *
import wandb

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

run = wandb.init(project="MacPINN", entity="srl_ethz", name="kernels, architecture, loss outside")


def train(model, optimizer, dataset, num_epochs, n_batches_per_epoch, weight_boundary, weight_p_reg, rho, mu, dt, integration, device, gradient_clipping=None, log_loss=True):
    model = model.to(device)
    model.train()
    
    eq_loss = NavierStokesLossMAC(rho, mu, dt, params.integrator)
    boundary_loss = BoundaryLossMAC()
    regularization = GradPRegularizer()

    for epoch in range(num_epochs):
        for batch in range(n_batches_per_epoch):
            v_cond, cond_mask, flow_mask, a_old, p_old = toCuda(dataset.ask())
            
            v_cond = convert_grid(v_cond, "staggered")
            cond_mask_mac = (convert_grid(
                cond_mask.repeat(1, 2, 1, 1), "staggered") == 1).float()
            flow_mask_mac = (convert_grid(
                flow_mask.repeat(1, 2, 1, 1), "staggered") >= 0.5).float()
            
            v_old = curl(a_old)
            a_new, p_new = fluid_model(a_old, p_old, v_old, flow_mask, cond_mask, v_cond)
            v_new = curl(a_new)
            
            loss_bound = boundary_loss(v_new, v_cond, cond_mask_mac)
            loss_nav = eq_loss(v_old, v_new, p_new, flow_mask_mac)
            regularize_grad_p = regularization(p_new)
            
            loss = weight_boundary * loss_bound + loss_nav + \
                weight_p_reg * regularize_grad_p

            if log_loss:
                loss = torch.log(loss)

            loss = torch.mean(loss)

            # compute gradients
            optimizer.zero_grad()
            loss.backward()
            
            if gradient_clipping is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            
            optimizer.step()
            
            p_new.data = (p_new.data-torch.mean(p_new.data, dim=(1, 2, 3)
                                                ).unsqueeze(1).unsqueeze(2).unsqueeze(3))
            a_new.data = (a_new.data-torch.mean(a_new.data, dim=(1, 2, 3)
                                                ).unsqueeze(1).unsqueeze(2).unsqueeze(3))

            dataset.tell(toCpu(a_new), toCpu(p_new))
            
            if batch % 16 == 0:
                loss = toCpu(loss).numpy()
                loss_bound = toCpu(torch.mean(loss_bound)).numpy()
                loss_nav = toCpu(torch.mean(loss_nav)).numpy()
                regularize_grad_p = toCpu(torch.mean(regularize_grad_p)).numpy()
                
                wandb.log(
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "boundary loss": loss_bound,
                        "equation loss": loss_nav,
                        "norm of grad p": regularize_grad_p,
                    }
                )

        if params.log:
            logger.save_state(fluid_model, optimizer, epoch+1)

with run:
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

    train(fluid_model, optimizer, dataset, params.n_epochs, params.n_batches_per_epoch, params.loss_bound, params.regularize_grad_p, params.rho, params.mu, params.dt, params.integrator, torch.device("cuda"), gradient_clipping=None, log_loss=True)

    # eq_loss = NavierStokesLossMAC(rho, mu, dt, params.integrator)
    # boundary_loss = BoundaryLossMAC()
    # regularization = GradPRegularizer()

    # # training loop
    # for epoch in range(params.load_index, params.n_epochs):

    #     for i in range(params.n_batches_per_epoch):
    #         v_cond, cond_mask, flow_mask, a_old, p_old = toCuda(dataset.ask())

    #         # convert v_cond,cond_mask,flow_mask to MAC grid
    #         v_cond = convert_grid(v_cond, "staggered")
    #         cond_mask_mac = (convert_grid(
    #             cond_mask.repeat(1, 2, 1, 1), "staggered") == 1).float()
    #         flow_mask_mac = (convert_grid(
    #             flow_mask.repeat(1, 2, 1, 1), "staggered") >= 0.5).float()

    #         v_old = curl(a_old)

    #         # predict new fluid state from old fluid state and boundary conditions using the neural fluid model
    #         a_new, p_new = fluid_model(a_old, p_old, v_old, flow_mask, cond_mask, v_cond)
    #         v_new = curl(a_new)
            
    #         loss_bound = boundary_loss(v_new, v_cond, cond_mask_mac)
    #         loss_nav = eq_loss(v_old, v_new, p_new, flow_mask_mac)
    #         regularize_grad_p = regularization(p_new)

    #         loss = params.loss_bound*loss_bound + params.loss_nav*loss_nav + \
    #             params.regularize_grad_p*regularize_grad_p

    #         loss = torch.mean(torch.log(loss))

    #         # compute gradients
    #         optimizer.zero_grad()
    #         loss.backward()

    #         # optional: clip gradients
    #         if params.clip_grad_value is not None:
    #             torch.nn.utils.clip_grad_value_(
    #                 fluid_model.parameters(), params.clip_grad_value)
    #         if params.clip_grad_norm is not None:
    #             torch.nn.utils.clip_grad_norm_(
    #                 fluid_model.parameters(), params.clip_grad_norm)

    #         # perform optimization step
    #         optimizer.step()

    #         p_new.data = (p_new.data-torch.mean(p_new.data, dim=(1, 2, 3)
    #                                             ).unsqueeze(1).unsqueeze(2).unsqueeze(3))  # normalize pressure
    #         a_new.data = (a_new.data-torch.mean(a_new.data, dim=(1, 2, 3)
    #                                             ).unsqueeze(1).unsqueeze(2).unsqueeze(3))  # normalize a

    #         # recycle data to improve fluid state statistics in dataset
    #         dataset.tell(toCpu(a_new), toCpu(p_new))

    #         # log training metrics
    #         if i % 16 == 0:
    #             loss = toCpu(loss).numpy()
    #             loss_bound = toCpu(torch.mean(loss_bound)).numpy()
    #             loss_nav = toCpu(torch.mean(loss_nav)).numpy()
    #             regularize_grad_p = toCpu(torch.mean(regularize_grad_p)).numpy()
                
    #             wandb.log(
    #                 {
    #                     "epoch": epoch,
    #                     "batch": i * 16,
    #                     "loss": loss,
    #                     "boundary loss": loss_bound,
    #                     "equation loss": loss_nav,
    #                     "norm of grad p": regularize_grad_p,
    #                 }
    #             )

    #             if i % 128 == 0:
    #                 print(
    #                     f"{epoch}: i:{i}: loss: {loss}; loss_bound: {loss_bound}; loss_nav: {loss_nav};")

    #     # safe state after every epoch
    #     if params.log:
    #         logger.save_state(fluid_model, optimizer, epoch+1)
