"""
Visualize the fitted TrueSpatialLowRankGaussian against real normal features
at one chosen (class, patch) pair using t-SNE.

For a chosen class C and patch index p:
  - normal:    real features {x_{i, p}} extracted from the training set
  - gaussian:  samples drawn from N(mu_p, U_p Lambda_p U_p^T + eps_p I)
  - boundary:  (optional) the synthetic "anomalies" that generate_anomalies()
               places on the Mahalanobis shell sqrt(T_p) + delta

Example:
    uv run python tsne_gaussian_vs_normal.py \
        --classname screw \
        --patch_idx 648 \
        --data_path /home/maometus/Documents/datasets/mvtec_anomaly_detection \
        --num_gaussian_samples 1000 \
        --num_boundary_samples 200
"""

import os
import click
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

import backbones
import simplenet as simplenet_mod
import utils
from low_rank_gaussian import TrueSpatialLowRankGaussian
from datasets.mvtec import MVTecDataset, DatasetSplit


def build_embedder(device, imagesize, backbone_name, layers,
                   pretrain_dim, target_dim, patchsize, embedding_size):
    backbone = backbones.load(backbone_name)
    backbone.name, backbone.seed = backbone_name, None
    net = simplenet_mod.SimpleNet(device)
    net.load(
        backbone=backbone,
        layers_to_extract_from=list(layers),
        device=device,
        input_shape=imagesize,
        pretrain_embed_dimension=pretrain_dim,
        target_embed_dimension=target_dim,
        patchsize=patchsize,
        embedding_size=embedding_size,
        meta_epochs=1,
        aed_meta_epochs=1,
        gan_epochs=1,
        noise_std=0.015,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        dsc_lr=0.0002,
        auto_noise=0,
        train_backbone=False,
        cos_lr=False,
        pre_proj=1,
        proj_layer_type=0,
        mix_noise=1,
    )
    return net


def mahal2(x, mu_p, U_p, Lambda_p, eps_p):
    """Per-sample Mahalanobis^2 under Sigma_p = U Lambda U^T + eps I."""
    d = x - mu_p
    proj = d @ U_p
    return (proj ** 2 / Lambda_p).sum(-1) + ((d - proj @ U_p.T) ** 2).sum(-1) / eps_p


def collect_patch_features(embedder, dataloader, device, patch_idx, p_total):
    """Run embedder on the dataset and collect features at one patch index.

    Returns tensor (N, C) of normal features at that patch.
    """
    feats = []
    for data in tqdm(dataloader, desc="embedding"):
        with torch.no_grad():
            emb = embedder.embed(data["image"].to(device))[0]   # (B*P, C)
            emb = emb.reshape(-1, p_total, emb.shape[1])         # (B, P, C)
            feats.append(emb[:, patch_idx, :].cpu())
    return torch.cat(feats, dim=0)


@click.command()
@click.option("--classname", required=True, type=str,
              help="MVTec subdataset name, e.g. screw")
@click.option("--patch_idx", required=True, type=int,
              help="Index in [0, P-1]; default grid is 36x36 = 1296")
@click.option("--data_path", required=True,
              type=click.Path(exists=True, file_okay=False))
@click.option("--gaussian_dir", default="true_spatial_low_rank_gaussian",
              type=click.Path())
@click.option("--backbone_name", default="wideresnet50")
@click.option("--layers", "-le", multiple=True, default=("layer2", "layer3"))
@click.option("--pretrain_dim", default=1536, type=int)
@click.option("--target_dim", default=1536, type=int)
@click.option("--patchsize", default=3, type=int)
@click.option("--embedding_size", default=256, type=int)
@click.option("--resize", default=329, type=int)
@click.option("--imagesize_int", default=288, type=int)
@click.option("--batch_size", default=8, type=int)
@click.option("--num_workers", default=2, type=int)
@click.option("--num_gaussian_samples", default=1000, type=int)
@click.option("--num_boundary_samples", default=0, type=int,
              help="If > 0, also plot anomaly samples on the per-patch boundary")
@click.option("--boundary_delta", default=1.0, type=float)
@click.option("--anomaly_mode", default="default",
              type=click.Choice(["default", "subspace", "anchored"]),
              help="default=PDF (full-sphere), subspace=in-U_k only, "
                   "anchored=fresh-normal + in-U_k shift")
@click.option("--max_normal", default=2000, type=int,
              help="Cap on real normal features to keep t-SNE fast")
@click.option("--pca_dim", default=50, type=int)
@click.option("--perplexity", default=30.0, type=float)
@click.option("--seed", default=0, type=int)
@click.option("--gpu", default=0, type=int)
@click.option("--output", default=None, type=str)
def main(classname, patch_idx, data_path, gaussian_dir, backbone_name, layers,
         pretrain_dim, target_dim, patchsize, embedding_size, resize,
         imagesize_int, batch_size, num_workers, num_gaussian_samples,
         num_boundary_samples, boundary_delta, anomaly_mode, max_normal,
         pca_dim, perplexity, seed, gpu, output):

    utils.fix_seeds(seed)
    device = utils.set_torch_device([gpu])

    gaussian_path = os.path.join(gaussian_dir, f"mvtec_{classname}.pt")
    if not os.path.isfile(gaussian_path):
        raise FileNotFoundError(
            f"Could not find fitted gaussian at {gaussian_path}. "
            f"Run low_rank_gaussian.py first to generate it."
        )
    state = torch.load(gaussian_path, map_location="cpu")
    tslrg = TrueSpatialLowRankGaussian()
    tslrg.load_state_dict(state)

    P, C = tslrg.mu.shape
    if not (0 <= patch_idx < P):
        raise ValueError(f"patch_idx must be in [0, {P-1}], got {patch_idx}")

    mu_p = tslrg.mu[patch_idx].to(device)
    U_p = tslrg.U[patch_idx].to(device)
    Lambda_p = tslrg.Lambda[patch_idx].to(device)
    eps_p = tslrg.eps[patch_idx].to(device)
    T_p = tslrg.T[patch_idx].to(device)
    k_p = U_p.shape[1]

    print(f"[load] class={classname} patch={patch_idx}/{P-1}  C={C}  k={k_p}")
    print(f"[load] eps_p={eps_p.item():.4g}  T_p={T_p.item():.4g}")

    train_dataset = MVTecDataset(
        data_path,
        classname=classname,
        resize=resize,
        imagesize=imagesize_int,
        split=DatasetSplit.TRAIN,
        seed=seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    embedder = build_embedder(
        device, train_dataset.imagesize, backbone_name, layers,
        pretrain_dim, target_dim, patchsize, embedding_size,
    )

    normal = collect_patch_features(embedder, train_loader, device,
                                    patch_idx, P)  # (N, C)
    print(f"[normal] collected {tuple(normal.shape)}")

    if normal.shape[0] > max_normal:
        idx = torch.randperm(normal.shape[0])[:max_normal]
        normal = normal[idx]

    gauss = tslrg.generate_normal_at_patch(patch_idx, num_gaussian_samples).cpu()

    pieces = [normal, gauss]
    labels = (
        [0] * normal.shape[0] +
        [1] * gauss.shape[0]
    )
    legend = {0: f"normal (n={normal.shape[0]})",
              1: f"gaussian samples (n={gauss.shape[0]})"}

    if num_boundary_samples > 0:
        anchor_arg = None
        if anomaly_mode == "anchored":
            # Sample real normal features (with replacement) to use as anchors.
            idx = torch.randint(0, normal.shape[0], (num_boundary_samples,))
            anchor_arg = normal[idx].to(device)
        bdry = tslrg.generate_anomaly_at_patch(
            patch_idx, num_boundary_samples, delta=boundary_delta,
            mode=anomaly_mode, anchor=anchor_arg,
        ).cpu()
        pieces.append(bdry)
        labels += [2] * bdry.shape[0]
        legend[2] = f"boundary anomalies ({anomaly_mode}, delta={boundary_delta})"

    X = torch.cat(pieces, dim=0).numpy()
    y = np.array(labels)

    n_components = min(pca_dim, X.shape[0], X.shape[1])
    if n_components < X.shape[1]:
        X = PCA(n_components=n_components, svd_solver="randomized",
                random_state=seed).fit_transform(X)

    perp = min(perplexity, max(5.0, (X.shape[0] - 1) / 3))
    print(f"[tsne] X={X.shape}  perplexity={perp}")
    X2 = TSNE(n_components=2, perplexity=perp, learning_rate="auto",
              init="pca", random_state=seed).fit_transform(X)

    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:red"}
    fig, (ax_tsne, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))

    for cls in sorted(legend):
        m = y == cls
        ax_tsne.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.5,
                        c=colors[cls], label=legend[cls])
    ax_tsne.set_title(f"t-SNE — class={classname}  patch={patch_idx}")
    ax_tsne.legend(loc="best")

    with torch.no_grad():
        m2_groups = {
            0: mahal2(normal.to(device), mu_p, U_p, Lambda_p, eps_p).cpu().numpy(),
            1: mahal2(gauss.to(device), mu_p, U_p, Lambda_p, eps_p).cpu().numpy(),
        }
        if num_boundary_samples > 0:
            m2_groups[2] = mahal2(bdry.to(device), mu_p, U_p, Lambda_p, eps_p).cpu().numpy()

    all_vals = np.concatenate(list(m2_groups.values()))
    lo = max(all_vals.min(), 1e-3)
    hi = all_vals.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    for cls in sorted(m2_groups):
        ax_hist.hist(m2_groups[cls], bins=bins, alpha=0.5,
                     color=colors[cls], label=legend[cls], density=True)
    ax_hist.axvline(T_p.item(), color="k", linestyle="--",
                    label=f"T_p={T_p.item():.3g}")
    ax_hist.axvline(C, color="gray", linestyle=":",
                    label=f"C={C}")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Mahalanobis$^2$")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Mahalanobis$^2$ under fitted $\\Sigma_p$")
    ax_hist.legend(loc="best", fontsize=8)

    for cls, label in legend.items():
        v = m2_groups[cls]
        print(f"[mahal2] {label}: mean={v.mean():.3g}  median={np.median(v):.3g}  "
              f"min={v.min():.3g}  max={v.max():.3g}")

    plt.tight_layout()

    if output is None:
        suffix = f"_b{num_boundary_samples}" if num_boundary_samples > 0 else ""
        output = f"tsne_gaussian_vs_normal_{classname}_p{patch_idx}{suffix}.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {output}")


if __name__ == "__main__":
    main()
