import torch
from data_handlers import Load_ImageNet100
from overcomplete.models import DinoV2, ViT, ResNet
from torch.utils.data import DataLoader, TensorDataset
import os
from torch.utils.data import Dataset, DataLoader
import glob
from einops import rearrange
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generate_activations(models, image_dataloader, max_seq_len, save_dir, rearrange_string):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(image_dataloader, desc="Processing Batches")):
            images = images.to(device)
            batch_data = {'images': images.cpu()}
            
            for key, model in models.items():
                activations = model.forward_features(images)
                padded = torch.zeros(activations.shape[0], max_seq_len, activations.shape[-1], device=device)
                padded[:, :activations.shape[1], :] = activations
                activations = rearrange(padded, rearrange_string)
                batch_data[f"activations_{key}"] = padded.cpu()

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