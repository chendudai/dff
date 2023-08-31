import torch
from einops import rearrange
import numpy as np

def feature_vis(features,h): 
   with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):                     
        U, S, V = torch.pca_lowrank(
            (features -features.mean(0)[None]).float(),
            niter=5)
        proj_V = V[:, :3].float()
        lowrank = torch.matmul(features.float(), proj_V)
        lowrank_sub = lowrank.min(0, keepdim=True)[0]
        lowrank_div = lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0]
        lowrank = ((lowrank - lowrank.min(0, keepdim=True)[0]) / (lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0])).clip(0, 1)
        visfeat = rearrange(lowrank.cpu().numpy(), '(h w) c -> h w c', h=h)
        visfeat = (visfeat*255).astype(np.uint8)
        return visfeat