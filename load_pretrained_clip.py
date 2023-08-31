import sys
sys.path.append("/storage/chendudai/envs/conda/envs/Ha-NeRF2/lib/python3.6/site-packages")
from transformers import SwinConfig, SwinModel

# Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
configuration = SwinConfig()

# Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
model = SwinModel(configuration)

# Accessing the model configuration
configuration = model.config