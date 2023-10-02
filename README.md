# improved-diffusion-for-physics

```python
import torch

from improved_diffusion.unet import UNetModel
from improved_diffusion.losses import ODEFlowMatchingLoss
from improved_diffusion.functional import ode_euler_integration

model = UNetModel(
    in_channels=198, # should be equal to num_features (input features) 
    dims=1, #this states, that we are using 1D U-Net
    condition_dims=1, # num_condition_features
    model_channels=256, # inner model features
    out_channels=198, # should be equal to num_features (input features) 
    num_res_blocks=10, # idk
    attention_resolutions=("16",) # idk
)

x_0 = torch.rand(23, 64, 198) # our input [batch_size, num_atoms, num_features]
#num_atoms should be a 2 to some power
t = torch.rand(23) # our time [batch_size]
y = torch.rand(23, 1) # features to condition on [batch_size, num_condition_features]

model(
    x=x_0, 
    timesteps=t, 
    y=y
).shape # torch.Size([23, 64, 198]), which matches x.shape torch.Size([23, 64, 198])
```