import matplotlib.pyplot as plt
import torch
from einops import rearrange
import heapq

## Create a tensor to save a list of top activations
topk = int(0.08 * 768 * 7) 
selected_concepts = torch.zeros(topk+1)
activations = next(iter(activations_dataloader))

for i, (key, model) in enumerate(models.items()):
  sae = SAEs[key]
  Activations = activations[f'activations_{key}'].to(device)
  with torch.no_grad():
    pre_codes, codes = sae.encode(Activations.squeeze())
    
    codes = rearrange(codes, '(n w h) d -> n w h d', w=16, h=16)
    
    codes_flat = codes.abs().sum(dim=(1, 2))        
    concept_strength = codes_flat.sum(dim=0)        
    top_concepts = torch.argsort(concept_strength, descending=True)[:topk].to(device)
    selected_concepts[i:i + topk] = top_concepts



# Overlay Top 20 for this model
for id in selected_concepts:
#for concept_id in range(50):
  concept_id = int(id.item())
  for key, model in models.items():
    sae = SAEs[key]
    Activations = activations[f'activations_{key}'].to(device)
    with torch.no_grad():
      pre_codes, codes = sae.encode(Activations.squeeze())

    codes = rearrange(codes, '(n w h) d -> n w h d', w=14, h=14)
    
    save_dir = f"results/usae_run8/{key}_concepts"

    overlay_top_heatmaps(activations[f"images"].squeeze(), codes, concept_id=concept_id)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"concept_{concept_id}_{key}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()  





def visualize(activation_loader, SAEs, concept_id, save_dir, n_images=10):
    # Initialize a heap for each model that will store the (score, image) for ONE model
    heaps = {}
    for name, _ in SAEs.items():
        heaps[name] = [] 


    for batch in activation_loader:
        for key, model in models.items():
            
            for name, sae in SAEs.items():
                sae.to(device)
                with torch.no_grad():
                    _, heatmaps = sae.encode(batch[f'activations_{name}'].to(device))

                top_ids = torch.mean(heatmaps[:, :, :, concept_id], dim=(1, 2)).argsort()[-n_images:]  

                for k, id in enumerate(top_ids):    
                    
                    score = torch.mean(heatmap[id, :, :, concept_id], dim=(1, 2))
                    z =  {
                        'image': batch['images'][id].cpu(), 
                        'heatmap': heatmaps[id, :, :, concept_id]
                    }

                    score_item = (score, z)
                    
                    if len(heaps[name]) < n_images:
                        heapq.heappush(heaps[name], score_item)
                    else:
                        heapq.heappushpop(heaps[name], score_item)


    for name, _ in SAEs.items():
        
    
        for i in range(n_images):
            image = heaps[name][i][1]['image']
            heat = heaps[name][i][1]['heatmap']

            width, height = get_image_dimensions(image)
            heatmap = interpolate_cv2(heat, (width, height))

            plt.subplot(2, 5, i + 1)
            show(image)
            show(heatmap, cmap=cmap, alpha=alpha)

