import torch

from flowmatch.flow import flow_pair, integrate_ode
from flowmatch.utils import resolve_source_target, nearest_labels, show_sequence, log_full_grads

try:
    import wandb
except Exception:
    pass


def train_one_epoch(model, ema, loader, optimizer, device, direction, embedding_provider, criterion, v_pred=True, log_wandb=False, epoch=0):
    model.train()
    running = 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        label = embedding_provider.sample(y, sample=True)
        t = torch.rand(x.shape[0], device=device)

        # logit normal
        #mu = torch.tensor(0.0)
        #sigma = torch.tensor(2.2)
        #z = torch.randn(x.shape[0], device=device) * sigma + mu
        #t = torch.sigmoid(z)

        x0, x1, t_ = resolve_source_target(direction, image=x, label=label, t=t)

        xt, target = flow_pair(x0, x1, t_, sigma_min=0.0, v_pred=v_pred)
        pred = model(xt, t_)
        loss = criterion(pred, target)


        optimizer.zero_grad()
        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float('inf')
        )
        if i % 20 == 0:
            log_full_grads(model, epoch, i, direction + "_grads")

        optimizer.step()
        ema.update(model)

        running += loss.item()
        if log_wandb:
            wandb.log({"train_loss_batch": loss.item(), "grad_norm": total_norm, "epoch": epoch})
        
        if i > 200:
            break

    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, ema, loader, device, direction: str, embedding_provider, steps=10, v_pred=True):
    ema.apply_shadow(model)
    model.eval()
    all_correct_l2 = 0
    all_correct_cos = 0
    all_count = 0

    means = embedding_provider.means().to(device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if direction == "image_to_label":
            t0, t1 = 0.0, 1.0
        else:
            t0, t1 = 1.0, 0.0

        seq = integrate_ode(model, x0=x, t0=t0, t1=t1, steps=steps, v_pred=v_pred)
        x_final = seq[-1]

        preds_l2, preds_cos = nearest_labels(x_final, means)

        all_correct_l2 += (preds_l2 == y).sum().item()
        all_correct_cos += (preds_cos == y).sum().item()
        all_count += y.numel()

        break
    
    ema.restore(model)

    fig_label_seq = show_sequence(seq)
    x1 = embedding_provider.sample(y, sample=True)
    seq = integrate_ode(model, x0=x1, t0=t1, t1=t0, steps=steps)
    fig_image_seq = show_sequence(seq)
    x1 = embedding_provider.sample(y, sample=True)
    seq = integrate_ode(model, x0=x1, t0=t1, t1=t0, steps=steps)
    fig_image_seq2 = show_sequence(seq)

    acc_cos = all_correct_cos / max(1, all_count)
    acc_l2 = all_correct_l2 / max(1, all_count)

    return acc_l2, acc_cos, fig_label_seq, fig_image_seq, fig_image_seq2
