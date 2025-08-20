from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple, Union, Any, List
import random
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm


Batch = Union[Tuple[Any, Any], dict]
FeatureFn = Callable[[Batch, Any, Optional[str]], np.ndarray]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _flatten_activation(act: Any) -> np.ndarray:
    """Flatten activation to (batch, features)."""
    if isinstance(act, torch.Tensor):
        act = act.detach()
        if act.ndim > 2:
            act = torch.flatten(act, start_dim=1)
        return act.cpu().numpy()
    arr = _to_numpy(act)
    return arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr


def _resolve_module(model: Any, dotted: str):
    cur = model
    for name in dotted.split("."):
        cur = getattr(cur, name)
    return cur


@dataclass
class EmbeddingViz:
    model: Any
    dataloader: Iterable[Batch]
    device: Optional[str] = None
    layer_name: Optional[str] = None
    feature_fn: Optional[FeatureFn] = None
    standardize: bool = True
    pca_dims: Optional[int] = 50
    seed: int = 42
    use_tqdm: bool = True  # show progress bars

    # internal state
    _features: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _labels: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def _iter_with_tqdm(self, iterable, total=None, desc: Optional[str] = None):
        """Wrap an iterable with tqdm when enabled, otherwise return as-is."""
        if not self.use_tqdm:
            return iterable
        try:
            return tqdm(iterable, total=total, desc=desc, leave=False, ncols=80)
        except Exception:
            # tqdm is missing or fails for some reason, degrade gracefully.
            return iterable

    def _default_feature_fn(
        self, batch: Batch, model: Any, device: Optional[str]
    ) -> np.ndarray:
        """Default: forward pass and use model outputs (logits) as features."""
        if isinstance(batch, dict):
            x = batch.get("inputs") or batch.get("x")
        else:
            x = batch[0]

        x = x.to(device) if (device and isinstance(x, torch.Tensor)) else x
        model.eval()
        with torch.no_grad():
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return _flatten_activation(out)

    def _labels_from_batch(self, batch: Batch) -> Optional[np.ndarray]:
        if isinstance(batch, dict):
            y = batch.get("labels") or batch.get("y") or batch.get("targets")
        else:
            y = batch[1] if len(batch) > 1 else None
        return None if y is None else _to_numpy(y)

    def _extract_with_hook(
        self, device: Optional[str], max_batches: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.layer_name is None:
            raise ValueError("layer_name must be set for hook-based extraction.")

        module = _resolve_module(self.model, self.layer_name)
        captured: List[np.ndarray] = []

        def hook(_mod, _inp, out):
            captured.append(_flatten_activation(out))

        handle = module.register_forward_hook(hook)
        self.model.eval()
        if device and hasattr(self.model, "to"):
            self.model.to(device)

        # Progress bar setup
        total_batches = None
        try:
            total_batches = len(self.dataloader)
        except Exception:
            pass
        if max_batches is not None and total_batches is not None:
            total_batches = min(total_batches, max_batches)

        feats, labels = [], []
        try:
            count = 0
            for batch in self._iter_with_tqdm(
                self.dataloader, total=total_batches, desc="Extracting (hook)"
            ):
                if isinstance(batch, dict):
                    x = batch.get("inputs") or batch.get("x")
                else:
                    x = batch[0]
                x = x.to(device) if (device and isinstance(x, torch.Tensor)) else x
                with torch.no_grad():
                    _ = self.model(x)
                if not captured:
                    raise RuntimeError(
                        "Hook did not capture any activations; check layer_name."
                    )
                feats.append(captured[-1])
                y = self._labels_from_batch(batch)
                if y is not None:
                    labels.append(y)

                count += 1
                if max_batches is not None and count >= max_batches:
                    break
        finally:
            handle.remove()

        X = np.concatenate(feats, axis=0)
        y = np.concatenate(labels, axis=0) if labels else None
        return X, y

    def extract_features(
        self, max_batches: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Collect features (X) and labels (y) from the dataloader.

        Priority: feature_fn > layer_name hook > model outputs.
        """
        set_seed(self.seed)

        if isinstance(self.model, nn.Module):
            self.model.eval()
            if self.device and hasattr(self.model, "to"):
                self.model.to(self.device)

        # If using a hook-based path, do it (now with progress and max_batches)
        if self.feature_fn is None and self.layer_name is not None:
            X, y = self._extract_with_hook(self.device, max_batches=max_batches)
        else:
            fn = self.feature_fn or self._default_feature_fn
            feats, labels = [], []

            # Progress bar setup
            total_batches = None
            try:
                total_batches = len(self.dataloader)
            except Exception:
                pass
            if max_batches is not None and total_batches is not None:
                total_batches = min(total_batches, max_batches)

            count = 0
            for batch in self._iter_with_tqdm(
                self.dataloader, total=total_batches, desc="Extracting (forward)"
            ):
                f = fn(batch, self.model, self.device)
                feats.append(_flatten_activation(f))
                yb = self._labels_from_batch(batch)
                if yb is not None:
                    labels.append(yb)
                count += 1
                if max_batches is not None and count >= max_batches:
                    break

            X = np.concatenate(feats, axis=0)
            y = np.concatenate(labels, axis=0) if labels else None

        # Preprocess: standardize and PCA (tiny staged bar)
        preprocess_steps = 0
        will_standardize = bool(self.standardize)
        will_pca = bool(self.pca_dims is not None and X.shape[1] > self.pca_dims)
        preprocess_steps += 1 if will_standardize else 0
        preprocess_steps += 1 if will_pca else 0

        if preprocess_steps and self.use_tqdm:
            pbar = tqdm(
                total=preprocess_steps, desc="Preprocess", leave=False, ncols=80
            )
        else:
            pbar = None

        if will_standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            if pbar:
                pbar.update(1)

        if will_pca:
            X = PCA(n_components=self.pca_dims, random_state=self.seed).fit_transform(X)
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        self._features, self._labels = X, y
        return X, y

    def embed(self, method: str = "tsne", **kwargs) -> np.ndarray:
        """Compute a 2D/3D embedding with 'tsne' or 'umap'.

        Common kwargs:
          - n_components: 2 or 3
        t-SNE kwargs:
          - perplexity, learning_rate, init, n_iter, metric
        UMAP kwargs:
          - n_neighbors, min_dist, metric, densmap (bool)

        Note: sklearn TSNE/umap-learn do not expose per-iteration callbacks;
        tqdm will show phase progress in generate_plots, while here we enable
        library 'verbose' output if requested by the user.
        """
        if self._features is None:
            raise RuntimeError("Call extract_features() first.")

        X = self._features
        method = method.lower()

        if method == "tsne":
            n_components = int(kwargs.pop("n_components", 2))
            # Respect user-provided verbose; otherwise keep default (no spam).
            tsne = TSNE(n_components=n_components, random_state=self.seed, **kwargs)
            Z = tsne.fit_transform(X)
            return Z

        if method == "umap":
            n_components = int(kwargs.pop("n_components", 2))
            reducer = umap.UMAP(
                n_components=n_components, random_state=self.seed, **kwargs
            )
            Z = reducer.fit_transform(X)
            return Z

        raise ValueError("method must be 'tsne' or 'umap'")

    def plot(
        self,
        embedding: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Embedding",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
        alpha: float = 0.8,
    ) -> None:
        """Plot 2D or 3D embeddings with optional label coloring."""
        if embedding.ndim != 2 or embedding.shape[1] not in (2, 3):
            raise ValueError("embedding must be (n_samples, 2|3)")

        plt.figure(figsize=figsize)

        if embedding.shape[1] == 2:
            scatter = plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels if labels is not None else None,
                alpha=alpha,
            )
            if labels is not None:
                handles, legend_labels = scatter.legend_elements()
                plt.legend(handles, legend_labels, title="Classes", loc="best")
            plt.title(title)
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            ax = plt.axes(projection="3d")
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=labels if labels is not None else None,
                alpha=alpha,
            )
            if labels is not None:
                handles, legend_labels = scatter.legend_elements()
                ax.legend(handles, legend_labels, title="Classes", loc="best")
            ax.set_title(title)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=160)
        plt.close()

    def save_embedding_csv(
        self, embedding: np.ndarray, labels: Optional[np.ndarray], path: str
    ) -> None:
        """Save embedding (and optional labels) to CSV."""
        if labels is not None:
            data = np.column_stack([embedding, labels])
            header = ",".join(
                [f"dim{i+1}" for i in range(embedding.shape[1])] + ["label"]
            )
        else:
            data = embedding
            header = ",".join([f"dim{i+1}" for i in range(embedding.shape[1])])
        np.savetxt(path, data, delimiter=",", header=header, comments="")


# ---- Convenience: one-shot helper ----------------------------------------


def generate_plots(
    model: Any,
    dataloader: Iterable[Batch],
    device: Optional[str] = None,
    layer_name: Optional[str] = None,
    feature_fn: Optional[FeatureFn] = None,
    out_dir: str = ".",
    base_name: str = "embedding",
    standardize: bool = True,
    pca_dims: Optional[int] = 50,
    seed: int = 42,
    tsne_kwargs: Optional[dict] = None,
    umap_kwargs: Optional[dict] = None,
    use_tqdm: bool = True,
) -> dict:
    """Extract features and write t-SNE and UMAP plots + CSVs to out_dir.

    Returns dict with file paths and numpy arrays.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    viz = EmbeddingViz(
        model=model,
        dataloader=dataloader,
        device=device,
        layer_name=layer_name,
        feature_fn=feature_fn,
        standardize=standardize,
        pca_dims=pca_dims,
        seed=seed,
        use_tqdm=use_tqdm,
    )

    # phase bar so you can see where time is going
    if use_tqdm:
        try:
            phase = tqdm(total=4, desc="Pipeline", leave=False, ncols=80)
        except Exception:
            phase = None
    else:
        phase = None

    X, y = viz.extract_features()
    if phase:
        phase.update(1)

    results = {"X": X, "y": y}

    # t-SNE
    tsne_kwargs = tsne_kwargs or {
        "n_components": 2,
        "perplexity": 30,
        "learning_rate": "auto",
        "init": "pca",
        # can pass 'verbose=1' to see library-side logs
    }
    Zt = viz.embed(method="tsne", **tsne_kwargs)
    if phase:
        phase.update(1)
    tsne_png = os.path.join(out_dir, f"{base_name}_tsne.png")
    tsne_csv = os.path.join(out_dir, f"{base_name}_tsne.csv")
    viz.plot(Zt, y, title="t-SNE", save_path=tsne_png)
    viz.save_embedding_csv(Zt, y, tsne_csv)

    results.update({"tsne": Zt, "tsne_png": tsne_png, "tsne_csv": tsne_csv})

    # UMAP
    umap_kwargs = umap_kwargs or {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1}
    Zu = viz.embed(method="umap", **umap_kwargs)
    if phase:
        phase.update(1)
    umap_png = os.path.join(out_dir, f"{base_name}_umap.png")
    umap_csv = os.path.join(out_dir, f"{base_name}_umap.csv")
    viz.plot(Zu, y, title="UMAP", save_path=umap_png)
    viz.save_embedding_csv(Zu, y, umap_csv)
    if phase:
        phase.update(1)
        phase.close()

    results.update({"umap": Zu, "umap_png": umap_png, "umap_csv": umap_csv})

    return results


# ---- Example custom feature_fn -------------------------------------------


def logits_feature_fn(batch: Batch, model: Any, device: Optional[str]) -> np.ndarray:
    """Example feature function that returns logits from a PyTorch model."""
    if isinstance(batch, dict):
        x = batch.get("inputs") or batch.get("x")
    else:
        x = batch[0]
    x = x.to(device) if (device and isinstance(x, torch.Tensor)) else x
    model.eval()
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return _flatten_activation(out)
