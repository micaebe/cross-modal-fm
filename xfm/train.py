import torch
from .rf import RF


def rf_forward_fn(rf: RF, x, y, use_bf16, device):
    """
    Args:
        rf: The RF wrapper.
        x: The input images.
        y: The corresponding class labels.
        use_bf16: Whether to use bfloat16.
        device: The device.
    """
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        vtheta, target_v, bwd_mask, t = rf.forward(x, y, True)
    sample_mse = ((target_v.float() - vtheta.float()) ** 2).flatten(start_dim=1).mean(dim=1)
    return sample_mse, t, bwd_mask

def train_rf(rf: RF,
            ema,
            data_iterator,
            loader,
            optimizer,
            device,
            num_steps,
            logger,
            global_step,
            use_bf16=False,
            max_grad_norm=None,
            scheduler=None):
    """
    Train the RF model.

    Args:
        rf: The RF model.
        ema: The EMA model.
        data_iterator: The training data iterator.
        loader: The training data loader (to reset the iterator).
        optimizer: The optimizer.
        device: The device.
        num_steps: The number of training steps.
        logger: The tensorboard logger.
        global_step: The global step.
        use_bf16: Whether to use bfloat16.
        max_grad_norm: Optional maximum gradient norm.
        scheduler: Optional learning rate scheduler.
    """
    rf.model.train()
    running = 0.0
    max_grad_norm = float("inf") if max_grad_norm is None else max_grad_norm

    optimizer.zero_grad(set_to_none=True)

    for i in range(num_steps):
        try:
            x, y = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            x, y = next(data_iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        sample_mse, t, _ = rf_forward_fn(rf, x, y, use_bf16, device)
        loss = sample_mse.mean()
        
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
                rf.model.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        optimizer.zero_grad(set_to_none=True)
        ema.update(rf.model)
        
        if (i + 1) % 5 == 0:
            logger.add_scalar("Train/Loss", loss.item(), global_step)
            logger.add_scalar("Train/Grad_Norm", grad_norm, global_step)
            logger.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)
        global_step += 1
        running += loss.item()
    return running / num_steps, global_step

