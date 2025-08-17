from lib.data_handlers import Load_ImageNet100
import torch
from tqdm import tqdm
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def remap_predictions(logits, internal_map=None):

    ## Logits should be Logits in the form B x 1000
    assert logits.shape[1] == 1000

    ## Get the internal class to index mapping
    if internal_map is None:
        _, image_dataset = Load_ImageNet100(transform=None, batch_size=256, shuffle=True, dataset=True)
        internal_map = image_dataset.class_to_idx


    ## Get the imagenet idx to class map
    import json
    with open("./datasets/imagenet-class-index-json/imagenet_class_index.json") as f:
        imagenet_idx_map = json.load(f)
    

    ## Extract the WNID labels from output logits
    predicted = torch.argmax(logits, dim=1)
    predicted_classes = [imagenet_idx_map[str(idx)][0] for idx in predicted.tolist()]

    ## Remap them to the internal index
    updated_idxs = [internal_map.get(label, -1) for label in predicted_classes] # Map to -1 if the label does not exist

    return torch.tensor(updated_idxs)



def train_mlp(image_loader, mlp, optimizer, loss_fn, vit_full, sae, internal_map, alpha=1.0, epoch=1):
    mlp.to(device)
    vit_full.to(device)
    sae.to(device)
    
    
    epoch_loss = 0.0
    pbar = tqdm(image_loader, desc=f"Epoch {epoch}", unit="batch")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        vit_full.eval()
        with torch.no_grad():
            # Get latent features & ViT predictions
            latent = vit_full.forward_features(images)
            predicted = vit_full.forward_head(latent)

            # Remap predictions to dataset indices
            predicted = remap_predictions(predicted, internal_map=internal_map)
            predicted = predicted.to(device)  # ensure predictions on device

            #print(predicted.shape)

            latent = latent[:, 1:]  # remove CLS token
            
            ## Filter for invalid Predictions
            valid_mask = predicted != -1
            if valid_mask.any():
                predicted = predicted[valid_mask]
                latent = latent[valid_mask, :, :]

            else:
                continue


            # SAE encode
            latent = rearrange(latent, 'n t d -> (n t) d')
            _, z = sae.encode(latent)
            z = z.to(device)
            z = rearrange(z, '(n t) d -> n t d', t=196)
            z = z.sum(dim=1, keepdim=True)
            z = z.flatten(start_dim=1)
            #print("Latent Output Shape: ", z.shape)

        mlp.train()
        optimizer.zero_grad()

        y_hat = mlp(z)
        
        l1_norm = sum(param.abs().sum() for name, param in mlp.named_parameters() if "weight" in name)
        loss = loss_fn(y_hat, predicted) + alpha*l1_norm

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch} finished. Average loss: {epoch_loss / len(image_loader):.4f}")
    return epoch_loss




def test_mlp(image_loader, mlp, loss_fn, vit_full, sae, internal_map):
    mlp.to(device)
    vit_full.to(device)
    sae.to(device)
    mlp.eval()
    vit_full.eval()
    sae.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(image_loader, desc="Testing", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Extract ViT features + predictions
            latent = vit_full.forward_features(images)
            predicted = vit_full.forward_head(latent)
            predicted = remap_predictions(predicted, internal_map=internal_map)
            predicted = predicted.to(device)

            latent = latent[:, 1:]  # remove CLS token
            
            # Filter invalid preds
            valid_mask = predicted != -1
            predicted = predicted[valid_mask]
            latent = latent[valid_mask, :, :]

            if predicted.numel() == 0:  # skip batch if all invalid
                continue

            # SAE encode
            latent = rearrange(latent, 'n t d -> (n t) d')
            _, z = sae.encode(latent)
            z = z.to(device)
            z = rearrange(z, '(n t) d -> n t d', t=196)
            z = z.sum(dim=1, keepdim=True)
            z = z.flatten(start_dim=1)

            # MLP forward
            y_hat = mlp(z)

            # Loss
            loss = loss_fn(y_hat, predicted)
            total_loss += loss.item()

            # Accuracy
            preds = torch.argmax(y_hat, dim=1)
            correct += (preds == predicted).sum().item()
            total += predicted.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(image_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test finished. Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy





def prune_weights(model, k=0.5):
        
    W = model.weight.data  
    absW = W.abs()

    threshold = k * absW.std()

    sparse_W = W.clone()
    sparse_W[absW < threshold] = 0.0

    model.weight.data = sparse_W

    return model



def check_sparsity(model):
    sparse_W = model.weight.data
    num_zero = (sparse_W == 0).sum().item()
    num_total = sparse_W.numel()
    sparsity = 100.0 * num_zero / num_total
    return sparsity