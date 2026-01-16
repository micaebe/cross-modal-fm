import torch
from rf import RF


def rf_forward_fn(rf: RF, x, y, use_bf16, device, t=None, bidi_mask=None):
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        vtheta, target_v, bwd_mask, t = rf.forward(x, y, True, t=t, bidi_mask=bidi_mask)
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
            grad_accum_steps=1,
            max_grad_norm=None,
            scheduler=None,
            log_t_loss=False):
    rf.model.train()
    running = 0.0
    max_grad_norm = float("inf") if max_grad_norm is None else max_grad_norm

    optimizer.zero_grad(set_to_none=True)

    if log_t_loss:
        n_tloss_bins = 10
        t_bin_samples = 0
        t_bins = {i: [] for i in range(n_tloss_bins)}

    for i in range(num_steps * grad_accum_steps):
        try:
            x, y = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            x, y = next(data_iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        sample_mse, t, _ = rf_forward_fn(rf, x, y, use_bf16, device)
        loss = sample_mse.mean()
        
        loss = loss / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                    rf.model.parameters(), max_norm=max_grad_norm
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
            ema.update(rf.model)
            
            if (i + 1) % (grad_accum_steps * 5) == 0:
                logger.add_scalar("Train/Loss", loss.item() * grad_accum_steps, global_step)
                logger.add_scalar("Train/Grad_Norm", grad_norm, global_step)
                logger.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)
                if log_t_loss:
                    for t_, l_ in zip(t.detach().cpu().tolist(), sample_mse.detach().cpu().tolist()):
                        bin_idx = int(t_ * n_tloss_bins)
                        t_bins[bin_idx].append(l_)
                    t_bin_samples += 1
                    if t_bin_samples == 5:
                        for bin_idx, losses in t_bins.items():
                            logger.add_scalar(f"Train/T_Loss_Bin_{bin_idx}", sum(losses) / len(losses), global_step)
                            # print pretty
                            print(f"Train/T_Loss_Bin_{bin_idx}: {sum(losses) / len(losses)}")
                        t_bins = {i: [] for i in range(n_tloss_bins)}
                        t_bin_samples = 0

            global_step += 1
        running += loss.item() * grad_accum_steps
    return running / (num_steps * grad_accum_steps), global_step

