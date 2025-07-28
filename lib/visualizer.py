import matplotlib.pyplot as plt
import torch
from einops import rearrange
import heapq
from overcomplete.visualization.plot_utils import get_image_dimensions, interpolate_cv2, show
from overcomplete.visualization.cmaps import VIRIDIS_ALPHA
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_concept(activation_loader, SAEs, concept_id, save_dir, patch_width, n_images=10, abort_dead=False):
    # Initialize a heap for each model that will store the (score, image) for ONE model
    heaps = {}
    for name, sae in SAEs.items():
        heaps[name] = [] 
        sae.to(device) # Also send to device while we're at it


    for batch in activation_loader:

        for name, sae in SAEs.items():
            
            # Forward pass this batch
            with torch.no_grad():
                _, heatmaps = sae.encode(batch[f'activations_{name}'].squeeze().to(device))
                heatmaps = rearrange(heatmaps, '(n w h) d -> n w h d', w=patch_width, h=patch_width)
            
            # Calculate the top_ids in this batch
            top_ids = torch.mean(heatmaps[:, :, :, concept_id], dim=(1, 2)).argsort()[-n_images:]  


            for k, id in enumerate(top_ids):    

                # Calculate score of each id in this batch                
                score = torch.mean(heatmaps[id, :, :, concept_id], dim=(1, 2))
                z =  {
                    'image': batch['images'][id].cpu(), 
                    'heatmap': heatmaps[id, :, :, concept_id]
                }

                # Maintain and Update the Top-n in a heap for this model: -score ensures its a max heap
                score_item = (-score, z)
                
                if len(heaps[name]) < n_images:
                    heapq.heappush(heaps[name], score_item)
                else:
                    heapq.heappushpop(heaps[name], score_item)

    # If abort enabled, then validate if concept is dead, if it is, do nothing.
    if abort_dead:
        if not validate_dead(heaps):
            return

    # Once I have the top-n for each SAE for this concept
    for j, (name, _) in enumerate(SAEs.items()):
        for i in range(n_images):

            image = heaps[name][i][1]['image']
            heat = heaps[name][i][1]['heatmap']

            width, height = get_image_dimensions(image)
            heatmap = interpolate_cv2(heat, (width, height))

            # Create the Image
            plt.subplot(len(SAEs) * 2, n_images / 2, j + i + 1)
            show(image)
            show(heatmap, cmap=VIRIDIS_ALPHA, alpha=1.0)
            
            # Save the Image
            os.makedirs(save_dir, exist_ok=True)
            filename = f"concept_{concept_id}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()  




def validate_dead(heaps):
    # Validate if there is ANY score that is greater than 0
    for sae, heap in heaps.items():
        score = -heap[0][0]
        if score.item() > 0:
            return True 

    # If there is no score above 0, then return False.
    return False