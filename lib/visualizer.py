import matplotlib.pyplot as plt
import torch
import heapq
import os
import random
import itertools

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from einops import rearrange
from overcomplete.visualization.plot_utils import (interpolate_cv2, get_image_dimensions, show)
from overcomplete.visualization.cmaps import VIRIDIS_ALPHA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



def save_concept_plot(concept_id, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    filename = f"concept_{concept_id:04d}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, bbox_inches="tight", dpi=300)




def plot_concept_grid(concept_id, subset, n_images, save_dir):
    """
    Plots and saves the visualization grid for a single concept.
    """


    # print(subset)
    # print("I AM INSIDE CONCEPT GRID")

    rows_per_model = 2
    total_rows = len(subset) * rows_per_model
    fig = plt.figure(figsize=(n_images * 2.5, total_rows * 2.5))

    for j, (name, _) in enumerate(subset.items()):
        
        heap = subset[name]
        title_row = j * rows_per_model + 1
        image_row_start = title_row + 1

        # Title row
        title_ax = plt.subplot(total_rows, n_images, (title_row - 1) * n_images + 1)
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            name,
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            transform=title_ax.transAxes,
        )

        # Image rows
        for i in range(n_images):
            
            idx = (image_row_start - 1) * n_images + i + 1                
                
            if heap:
                _, _, z = heapq.heappop(heap)
                image, heat = z["image"], z["heatmap"]

                width, height = get_image_dimensions(image)
                heatmap = interpolate_cv2(heat, (width, height))

                ax = plt.subplot(total_rows, n_images, idx)
                ax.axis("off")
                show(image)
                show(heatmap, cmap=VIRIDIS_ALPHA, alpha=1.0)
            
            else:
                ax = plt.subplot(total_rows, n_images, idx)
                ax.axis("off")
                show(torch.zeros(3, 224, 224))

    plt.tight_layout()
    save_concept_plot(concept_id, save_dir)
    plt.close()




def visualize_concepts(
    activation_loader,
    SAEs,
    save_dir,
    patch_width,
    num_concepts,
    n_images=10,
    abort_threshold=0,
):
    # Initialize a heap for each model that will store the (score, image) for ONE model
    heaps = {}
    heaps = {
        name: {concept_id: [] for concept_id in range(num_concepts)}
        for name in SAEs.keys()
    }

    # Initialize Tie Breaker on Heaps
    counter = itertools.count()

    #### OPTIMIZED VISUALIZER
    print("Generating Activations")
    for batch in tqdm(activation_loader):

        for name, sae in SAEs.items():

            # Encode only once
            with torch.no_grad():
                _, heatmaps = sae.encode(
                    batch[f"activations_{name}"].squeeze().to(device)
                )
                heatmaps = rearrange(
                    heatmaps, "(n w h) d -> n w h d", w=patch_width, h=patch_width
                )  # shape: [B, w, h, D]

            B, w, h, D = heatmaps.shape

            # Compute mean heatmap per image per concept: [B, D]
            sum_heatmaps = heatmaps.sum(dim=(1, 2))  # shape: [B, D]

            # indices: [k, D], values: [k, D]
            topk_values, topk_indices = torch.topk(sum_heatmaps.T, k=n_images, dim=1)

            # Get top-k per concept efficiently using torch.topk
            for concept_id in range(D):
                
                

                for rank in range(n_images):
                    idx = topk_indices[concept_id, rank].item()
                    score = topk_values[concept_id, rank].item()

                    # Only proceed if this score is better than the worst in the heap, or heap not full AND score is not 0
                    heap = heaps[name][concept_id]
                    
                    if (len(heap) < n_images or score > -heap[0][0]) and score > abort_threshold:
                        z = {
                            "image": batch["images"].squeeze()[idx].cpu(),
                            "heatmap": heatmaps[idx, :, :, concept_id].cpu(),
                        }
                        score_item = (-score, next(counter), z)

                        if len(heap) < n_images:
                            heapq.heappush(heap, score_item)
                        else:
                            heapq.heappushpop(heap, score_item)

    print("Saving Concepts")
    for concept in tqdm(range(num_concepts)):
        #print("INSIDE FOR LOOP")
        if all(len(heaps[n][concept]) == 0 for n in SAEs.keys()):
            continue
        else:
            subset = {k: deepcopy(heaps[k][concept]) for k in SAEs.keys()}
            plot_concept_grid(concept, subset, n_images, save_dir)








def visualize_class_on_concept(concept, class_idx, model, sae, domain_roots, save_dir=None, n_images=None):

    for domain, dir in domain_roots.items(): 
        classes = os.listdir(dir)
        class_dir = os.path.join(dir, classes[class_idx])

        images = [Image.open(os.path.join(class_dir, path)) for path in os.listdir(class_dir)]

        if n_images is not None and n_images < len(images):
            images = random.sample(images, n_images)

        for i, img in enumerate(images):
            img = img.convert("RGB")  
            img = model.preprocess(img).unsqueeze(dim=0)
            x = model.forward_features(img.to(device))
            x = rearrange(x, 'n t d -> (n t) d')
            _, z = sae.encode(x)
            z = rearrange(z, '(n w h) d -> n w h d', w=14, h=14)
            width, height = img.shape[-1], img.shape[-2]
            heatmap = interpolate_cv2(z[:, :, :, concept], (width, height))
            show(img)
            show(heatmap, cmap=VIRIDIS_ALPHA, alpha=1.0)
            
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"Class{class_idx}_Concept{concept}_Domain{domain}_{i}.png"))
                plt.close()
            else:
                plt.show()