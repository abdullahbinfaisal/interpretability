import torch
from data_handlers import Load_ImageNet100
from overcomplete.models import DinoV2, ViT, ResNet
from torch.utils.data import DataLoader, TensorDataset
import os
from torch.utils.data import Dataset, DataLoader
import glob
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_normalization_parameters(model):
    # Normalize across 1024 samples
    image_loader = Load_ImageNet100(transform=None, batch_size=1024, shuffle=True)
    img, _ = next(iter(image_loader))
    
    # Get activations and return normalization parameters
    activations = model.forward_features(img.to(device))
    flat = activations.permute(1, 0, 2).reshape(activations.shape[1], -1)  # (D, N*T)
    mean = flat.mean(dim=1) 
    std = flat.std(dim=1)
    return mean, std
    

def generate_activations(models, image_dataloader, max_seq_len, save_dir, rearrange_string):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(image_dataloader, desc="Processing Batches")):
            images = images.to(device)
            batch_data = {'images': images.cpu()}
            
            for key, model in models.items():
                
                # Get Activations
                activations = model.forward_features(images)
                
                # Interpolate Activation Tokens
                activations = activations.permute(0, 2, 1)  # Shape: (300, 382, 192)
                x_interp = F.interpolate(activations, size=max_seq_len, mode='linear', align_corners=False)
                x_interp = x_interp.permute(0, 2, 1)

                # Normalize Activations
                mean, std = get_normalization_parameters(model)
                x_interp = (x_interp - mean.view(1, -1, 1))
                x_interp = x_interp / (std.view(1, -1, 1) + 1e-6)

                # Batch Collapse
                activations = rearrange(x_interp, rearrange_string)
                batch_data[f"activations_{key}"] = activations.cpu()

            torch.save(batch_data, os.path.join(save_dir, f"batch_{i:05d}.pt"))
            


class PTFilesDataset(Dataset):
    def __init__(self, directory_path):
        """
        Args:
            directory_path (string): Path to the directory containing .pt files
        """
        self.file_paths = sorted(glob.glob(os.path.join(directory_path, '*.pt')))
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the dictionary from the .pt file
        data_dict = torch.load(self.file_paths[idx])
        
        # Assuming the dictionary contains tensors that can be directly used
        # You might need to modify this part based on your specific data structure
        return data_dict


def Load_activation_dataloader(models, image_dataloader, save_dir, generate, max_seq_len, rearrange_string=None):
    

    if generate == True and rearrange_string is not None:
        generate_activations(models, image_dataloader, max_seq_len, save_dir, rearrange_string)
    
    dataset = PTFilesDataset(directory_path=save_dir)

    # Create image_dataloader
    activation_dataloader = DataLoader(
        dataset,
        batch_size=1,          # Adjust batch size as needed
        shuffle=True,           # Shuffle if training
        drop_last=True        # Whether to drop last incomplete batch
    )

    return activation_dataloader