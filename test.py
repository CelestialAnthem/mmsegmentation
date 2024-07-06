import torch

# Load the model file
model_path = './checkpoints/cityscapes_vitl_mIoU_86.4.pth'
model_state_dict = torch.load(model_path)

# Print the keys in the state dictionary
for key in model_state_dict.keys():
    print(key)
    
    
