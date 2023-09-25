import torch
import torchvision.utils as tvu
import pdb
import os



# clean diffusion model loss
def clean_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)




# in distribution noise estimation 
def d2din_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          y: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          miu: torch.Tensor,
                          args=None,
                          keepdim=False):
    target_idx = torch.where(y == args.target_label)[0]
    chosen_mask = torch.bernoulli(torch.zeros_like(target_idx) + args.cond_prob)
    chosen_target_idx = target_idx[torch.where(chosen_mask == 1)[0]]

    batch, device = x0.shape[0], x0.device
    miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    x_ = x0 * a.sqrt() + e * (1.0 - a).sqrt() * args.gamma + miu_ * (1.0 - a).sqrt()
    if args.trigger_type == 'patch':
        tmp_x = x.clone()
        tmp_x[:, :, -args.patch_size:, -args.patch_size:] = x_[:, :, -args.patch_size:, -args.patch_size:]
        x_ = tmp_x

    x_add = x_[chosen_target_idx]
    t_add = t[chosen_target_idx]
    e_add = e[chosen_target_idx]
    x = torch.cat([x, x_add], dim=0)
    t = torch.cat([t, t_add], dim=0)
    e = torch.cat([e, e_add], dim=0)

    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)




# out of distribution noise estimation 
def d2dout_noise_estimation_loss(model,
                          x0: torch.Tensor,
                          y: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          miu: torch.Tensor,
                          args=None,
                          keepdim=False):
    target_idx = torch.where(y == 1000)[0]
    chosen_mask = torch.bernoulli(torch.zeros_like(target_idx) + args.cond_prob)
    chosen_target_idx = target_idx[torch.where(chosen_mask == 1)[0]]

    batch, device = x0.shape[0], x0.device
    miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    x_ = x0 * a.sqrt() + e * (1.0 - a).sqrt() * args.gamma + miu_ * (1.0 - a).sqrt()
    if args.trigger_type == 'patch':
        tmp_x = x.clone()
        tmp_x[:, :, -args.patch_size:, -args.patch_size:] = x_[:, :, -args.patch_size:, -args.patch_size:]
        x_ = tmp_x

    x_add = x_[chosen_target_idx]
    x[chosen_target_idx] = x_add

    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)





clean_loss_registry = {
    'simple': clean_noise_estimation_loss,
}

d2din_loss_registry = {
    'simple': d2din_noise_estimation_loss,
}

d2dout_loss_registry = {
    'simple': d2dout_noise_estimation_loss,
}
