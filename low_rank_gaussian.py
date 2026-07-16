from numpy import var
from torch import nn
import torch.nn.functional as F
import torch
import logging
import math
import common
import os
import utils
import backbones
import click
from tqdm import tqdm
import simplenet

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
}

class TrueSpatialLowRankGaussian():
    """
    Spatial Gaussian model with per-patch threshold T_p.

    Each spatial location p:
        x_{i,p} ~ N(mu_p, Sigma_p)

    Sigma_p = U_p Lambda_p U_p^T + eps_p I
    T_p = runtime-selectable quantile of Mahalanobis at patch p

    With neighborhood w > 1, Sigma_p is estimated by pooling centered features
    from a w x w window around p (truncated at the grid edge). Means and
    thresholds remain per-patch. Sample count per fit grows from N to ~N*w^2,
    which makes the rank-k+isotropic estimator far better-conditioned.
    """

    def __init__(self, k=512, quantile=0.99, neighborhood=1, eps_method="ppca"):
        self.requested_k = int(k)
        if self.requested_k < 1:
            raise ValueError("k must be >= 1")
        self.k = self.requested_k
        self.quantile = self._validate_quantile(quantile)
        self.neighborhood = int(neighborhood)
        if self.neighborhood < 1:
            raise ValueError("neighborhood must be >= 1")
        if eps_method not in ("ppca", "median"):
            raise ValueError("eps_method must be 'ppca' (mean of trailing) "
                             "or 'median' (PDF default 0.5 * median)")
        self.eps_method = eps_method

        self.mu = None            # (P, C)
        self.U = None             # (P, C, k)
        self.Lambda = None        # (P, k)
        self.eps = None           # (P,)
        self.T = None             # (P,)
        self.sorted_calibration_scores = None  # (P, N)
        self.spatial_shape = None # (H, W)

    @staticmethod
    def _validate_quantile(quantile):
        quantile = float(quantile)
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be a finite number strictly between 0 and 1")
        return quantile

    def set_quantile(self, quantile):
        """Update T_p from saved calibration scores without refitting covariance."""
        quantile = self._validate_quantile(quantile)

        if self.sorted_calibration_scores is None:
            if self.T is not None and math.isclose(
                quantile, self.quantile, rel_tol=0, abs_tol=1e-12
            ):
                return self.T
            raise RuntimeError(
                "This legacy TSLRG checkpoint stores only its fitted quantile. "
                "Regenerate it once to enable runtime quantile selection."
            )

        sample_count = self.sorted_calibration_scores.shape[1]
        position = quantile * (sample_count - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        self.T = torch.lerp(
            self.sorted_calibration_scores[:, lower],
            self.sorted_calibration_scores[:, upper],
            weight,
        )
        self.quantile = quantile
        return self.T

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    @torch.no_grad()
    def fit(self, X, spatial_shape=None):
        """
        X: (N, P, C). spatial_shape=(H, W); inferred as (sqrt(P), sqrt(P)) if None.
        """
        N, P, C = X.shape

        if spatial_shape is None:
            H = int(round(P ** 0.5))
            if H * H != P:
                raise ValueError(
                    f"P={P} is not a perfect square; pass spatial_shape explicitly"
                )
            W = H
        else:
            H, W = spatial_shape
            if H * W != P:
                raise ValueError(f"spatial_shape {(H, W)} inconsistent with P={P}")

        mu = X.mean(dim=0)                              # (P, C)
        Xc = (X - mu.unsqueeze(0)).reshape(N, H, W, C)  # centered, gridded

        win = self.neighborhood
        half = win // 2

        # Reflection-pad the grid so every patch has a full w x w window.
        # This keeps k_eff uniform across patches (corners/edges no longer truncate).
        if half > 0:
            Xc_pad = F.pad(
                Xc.permute(0, 3, 1, 2),       # (N, C, H, W)
                [half, half, half, half],
                mode="reflect",
            ).permute(0, 2, 3, 1)             # (N, H+2*half, W+2*half, C)
        else:
            Xc_pad = Xc

        U_list, Lambda_list, eps_list, score_list = [], [], [], []

        for i in range(H):
            for j in range(W):
                # Always-full w x w window in the padded grid.
                Yp = Xc_pad[:, i:i + win, j:j + win, :].reshape(-1, C)  # (N * win^2, C)

                _, S_svd, Vh = torch.linalg.svd(Yp, full_matrices=False)

                eigvals = (S_svd ** 2) / (Yp.shape[0] - 1)
                r = eigvals.shape[0]
                k_eff = min(self.k, r)

                V_k = Vh[:k_eff].T
                Lambda_k = eigvals[:k_eff].clamp_min(1e-12)

                if k_eff < r:
                    if self.eps_method == "ppca":
                        eps_p = torch.mean(eigvals[k_eff:])
                    else:
                        eps_p = 0.5 * torch.median(eigvals[k_eff:])
                else:
                    eps_p = torch.tensor(1e-2, device=X.device)
                eps_p = eps_p.clamp_min(1e-12)

                # Threshold is computed from the patch's OWN samples under (V_k, Lambda_k, eps_p)
                d = Xc[:, i, j, :]  # (N, C), already centered by mu_p
                proj = d @ V_k
                term1 = (proj ** 2 / Lambda_k).sum(dim=1)
                residual = d - proj @ V_k.T
                term2 = (residual ** 2).sum(dim=1) / eps_p
                scores_p = term1 + term2

                U_list.append(V_k)
                Lambda_list.append(Lambda_k)
                eps_list.append(eps_p)
                score_list.append(torch.sort(scores_p).values)

        self.mu = mu
        self.U = torch.stack(U_list, dim=0)
        self.Lambda = torch.stack(Lambda_list, dim=0)
        self.eps = torch.stack(eps_list, dim=0)
        self.k = self.U.shape[-1]
        self.sorted_calibration_scores = torch.stack(score_list, dim=0)
        self.set_quantile(self.quantile)
        self.spatial_shape = (H, W)

    # --------------------------------------------------
    # PER-PATCH SAMPLING
    # --------------------------------------------------
    def _device_for_sampling(self, device=None, anchor=None):
        if device is not None:
            return torch.device(device)
        if anchor is not None:
            return anchor.device
        return self.mu.device

    @torch.no_grad()
    def mahal2_at_patch(self, p, x, device=None):
        """Mahalanobis^2 under the fitted patch covariance. x: (B, C)."""
        device = self._device_for_sampling(device, x)
        mu_p = self.mu[p].to(device)
        U_p = self.U[p].to(device)
        Lambda_p = self.Lambda[p].to(device)
        eps_p = self.eps[p].to(device)

        d = x.to(device) - mu_p
        proj = d @ U_p
        term1 = (proj ** 2 / Lambda_p).sum(dim=1)
        residual = d - proj @ U_p.T
        term2 = (residual ** 2).sum(dim=1) / eps_p
        return term1 + term2

    @torch.no_grad()
    def _sample_radius(self, p, B, delta, radius, radius_mode, anchor, device):
        """Return a (B, 1) Mahalanobis radius for anomaly synthesis."""
        T_p = self.T[p].to(device)
        C = self.mu.shape[1]
        mode = radius_mode

        if mode == "threshold":
            if radius is None:
                return torch.sqrt(T_p) + delta * torch.rand(B, 1, device=device)
            return float(radius) * torch.rand(B, 1, device=device)

        scale = 1.0 if radius is None else float(radius)

        if mode == "patch":
            # Normalizes patch thresholds by the ambient chi-square scale so a
            # radius sweep means roughly the same thing across fitted patches.
            base = torch.sqrt(T_p / C).clamp_min(1e-6)
            return scale * base * torch.rand(B, 1, device=device)

        if mode == "anchor":
            if anchor is None:
                raise ValueError("radius_mode='anchor' requires real anchors")
            anchor_score = self.mahal2_at_patch(p, anchor, device=device)
            gap = torch.sqrt(T_p).unsqueeze(0) - torch.sqrt(anchor_score).unsqueeze(1)
            fallback = 0.05 * torch.sqrt(T_p)
            base = torch.maximum(gap, fallback.expand_as(gap))
            return scale * base * torch.rand(B, 1, device=device)

        raise ValueError("unknown radius_mode {!r}; expected one of "
                         "'threshold', 'patch', 'anchor'".format(radius_mode))

    @torch.no_grad()
    def _sigma_half_w(self, p, w, device):
        """Apply Sigma_p^(1/2) to w. w: (B, C). Returns (B, C)."""
        U_p = self.U[p].to(device)
        Lambda_p = self.Lambda[p].to(device)
        eps_p = self.eps[p].to(device)

        proj = w @ U_p
        low_rank = (proj * torch.sqrt(Lambda_p)) @ U_p.T
        recon = proj @ U_p.T
        residual = w - recon
        iso = torch.sqrt(eps_p) * residual

        return low_rank + iso

    @torch.no_grad()
    def generate_anomaly_at_patch(self, p, B, delta=1, mode="default", anchor=None,
                                   radius=None, radius_mode="threshold",
                                   device=None):
        """
        Generate B anomalies for patch p.

        mode="default":
            PDF formulation. Sample u uniform on the unit C-sphere, set
            radius r in [sqrt(T_p), sqrt(T_p)+delta], map x = mu + Sigma^(1/2)(r u).
            Anomaly mass is dominated by the orthogonal subspace (where eps lives).

        mode="subspace":
            Sample direction in U_p subspace only (k-dim sphere). Anomaly sits on a
            Mahalanobis ellipse inside the data-variation subspace. anchor ignored.

        mode="anchored":
            Add an in-U_p shift to a REAL normal feature passed via anchor.
            anchor must be shape (B, C).

        radius:
            Explicit Mahalanobis radius. Overrides the T_p-based default.
            - If None (default): r = sqrt(T_p) + delta * U(0,1). Statistical-
              threshold semantics from the PDF.
            - If float > 0: r = radius * U(0,1). Decoupled from T_p — use this
              to match a SimpleNet-style small noise magnitude in the U_p subspace.

        radius_mode:
            "threshold": existing behavior.
            "patch": radius * sqrt(T_p / C) * U(0,1), for patch-calibrated sweeps.
            "anchor": radius * gap(anchor, T_p) * U(0,1), for per-anchor near-
            boundary perturbations.

        Returns: (B, C)
        """
        device = self._device_for_sampling(device, anchor)
        C = self.mu.shape[1]
        U_p = self.U[p].to(device)         # (C, k)
        Lambda_p = self.Lambda[p].to(device)  # (k,)
        k = U_p.shape[1]

        r = self._sample_radius(p, B, delta, radius, radius_mode, anchor, device)

        if mode == "default":
            u = torch.randn(B, C, device=device)
            u = u / u.norm(dim=1, keepdim=True)
            w = u * r
            return self.mu[p].to(device) + self._sigma_half_w(p, w, device)

        if mode in ("subspace", "anchored"):
            v_k = torch.randn(B, k, device=device)
            v_k = v_k / v_k.norm(dim=1, keepdim=True)
            shift = (r * v_k * torch.sqrt(Lambda_p)) @ U_p.T

            if mode == "subspace":
                return self.mu[p].to(device) + shift

            if anchor is None:
                raise ValueError(
                    "mode='anchored' requires `anchor` of shape (B, C) — "
                    "real normal features to perturb."
                )
            if anchor.shape != (B, C):
                raise ValueError(
                    f"anchor shape {tuple(anchor.shape)} != expected ({B}, {C})"
                )
            return anchor.to(device) + shift

        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         "'default', 'subspace', 'anchored'")

    @torch.no_grad()
    def generate_normal_at_patch(self, p, B, device=None):
        """
        Draw B samples from N(mu_p, Sigma_p).
        Returns: (B, C)
        """
        device = self._device_for_sampling(device)
        C = self.mu.shape[1]

        w = torch.randn(B, C, device=device)

        return self.mu[p].to(device) + self._sigma_half_w(p, w, device)

    # --------------------------------------------------
    # BULK SAMPLING (all patches)
    # --------------------------------------------------
    @torch.no_grad()
    def generate_anomalies(self, B, delta=1, mode="default", anchors=None,
                           radius=None, radius_mode="threshold", device=None):
        """
        Generate anomalies for every patch. Returns (B, P, C).
        See generate_anomaly_at_patch for `mode`, `radius`, and `radius_mode`.
        `anchors` (B, P, C) is required when mode='anchored' and ignored otherwise.
        """
        device = self._device_for_sampling(device, anchors)
        P = self.mu.shape[0]
        if mode == "anchored":
            if anchors is None:
                raise ValueError("mode='anchored' requires `anchors` (B, P, C)")
            if anchors.shape[:2] != (B, P):
                raise ValueError(
                    f"anchors leading shape {tuple(anchors.shape[:2])} != ({B}, {P})"
                )
        return torch.stack(
            [
                self.generate_anomaly_at_patch(
                    p, B, delta, mode=mode,
                    anchor=anchors[:, p, :] if anchors is not None else None,
                    radius=radius,
                    radius_mode=radius_mode,
                    device=device,
                )
                for p in range(P)
            ],
            dim=1,
        )

    @torch.no_grad()
    def generate_normal_features(self, B, device=None):
        """Draw B samples per patch from N(mu_p, Sigma_p). Returns (B, P, C)."""
        device = self._device_for_sampling(device)
        P = self.mu.shape[0]
        return torch.stack(
            [self.generate_normal_at_patch(p, B, device=device) for p in range(P)], dim=1
        )

    def state_dict(self):
        return {
            "checkpoint_version": 2,
            "k": self.k,
            "requested_k": self.requested_k,
            "quantile": self.quantile,
            "neighborhood": self.neighborhood,
            "spatial_shape": self.spatial_shape,
            "eps_method": self.eps_method,
            "mu": self.mu,
            "U": self.U,
            "Lambda": self.Lambda,
            "eps": self.eps,
            "T": self.T,
            "sorted_calibration_scores": self.sorted_calibration_scores,
        }

    def load_state_dict(self, state):
        self.k = state["k"]
        self.requested_k = state.get("requested_k", self.k)
        self.quantile = state["quantile"]
        self.neighborhood = state.get("neighborhood", 1)
        self.spatial_shape = state.get("spatial_shape", None)
        self.eps_method = state.get("eps_method", "median")
        self.mu = state["mu"]
        self.U = state["U"]
        self.Lambda = state["Lambda"]
        self.eps = state["eps"]
        self.T = state["T"]
        self.sorted_calibration_scores = state.get("sorted_calibration_scores")

class SpatialLowRankGaussian(nn.Module):
    def __init__(self, mu_p = torch.zeros(1), Uk = torch.zeros(1, 1), lambdak = torch.zeros(1), eps = 0, T = 0):
        super().__init__()

        # register as buffers (not trainable, saved in state_dict)
        self.register_buffer("mu_p", mu_p)
        self.register_buffer("Uk", Uk)
        self.register_buffer("lambdak", lambdak)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("T", torch.tensor(T))

    @staticmethod
    def fit(X, k=64, q=0.995, eps=None):
        N, P, C = X.shape

        # ----- per-patch mean -----
        mu_p = X.mean(dim=0)  # [P, C]

        # center per patch
        Y = X - mu_p.unsqueeze(0)

        # flatten only for covariance
        Y_flat = Y.reshape(N * P, C)

        # SVD
        U, S, Vt = torch.linalg.svd(Y_flat, full_matrices=False)
        lambdas = (S**2) / (Y_flat.shape[0] - 1)

        r = lambdas.shape[0]
        k = min(k, r)

        Uk = Vt[:k].T
        lambdak = lambdas[:k]

        if eps is None:
            if r > k:
                eps = 0.5 * torch.median(lambdas[k:r])
            else:
                eps = 1e-3

        # compute threshold
        proj = Y_flat @ Uk
        term1 = (proj**2 / lambdak).sum(dim=1)

        recon = proj @ Uk.T
        residual = Y_flat - recon
        term2 = (residual**2).sum(dim=1) / eps

        scores = term1 + term2
        T = torch.quantile(scores, q)

        return SpatialLowRankGaussian(mu_p, Uk, lambdak, float(eps), float(T))

class LowRankGaussian(nn.Module):
    def __init__(self, mu = torch.zeros(1), Uk = torch.zeros(1, 1), lambdak = torch.zeros(1), eps = 0, T = 0):
        super().__init__()

        # register as buffers (not trainable, saved in state_dict)
        self.register_buffer("mu", mu)
        self.register_buffer("Uk", Uk)
        self.register_buffer("lambdak", lambdak)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("T", torch.tensor(T))

    @staticmethod
    def fit(X, k=64, q=0.995, eps=None):
        """
        X: [N, D] normal samples
        returns LowRankGaussian object
        """
        N, D = X.shape

        mu = X.mean(dim=0)
        Y = X - mu

        U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
        lambdas = (S**2) / (N - 1)

        r = lambdas.shape[0]
        k = min(k, r)

        Uk = Vt[:k].T
        lambdak = lambdas[:k]

        if eps is None:
            if r > k:
                tail = lambdas[k:r]
                eps = 0.5 * torch.median(tail)
            else:
                eps = 1e-3
        
        Y = X - mu
        proj = Y @ Uk

        term1 = (proj**2 / lambdak).sum(dim=1)

        recon = proj @ Uk.T
        residual = Y - recon
        term2 = (residual**2).sum(dim=1) / eps

        T = torch.quantile(term1+term2, q)

        return LowRankGaussian(mu, Uk, lambdak, float(eps), float(T))

@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="ckpt")
@click.option("--neighborhood", type=int, default=1, show_default=True,
              help="Window size for tied-covariance pooling. 1=per-patch (current), "
                   "3 or 5 share covariance with spatial neighbors.")
@click.option("--k", type=click.IntRange(min=1), default=512, show_default=True,
              help="Requested covariance rank; the fitted rank is capped by sample count.")
@click.option("--quantile", type=click.FloatRange(0, 1, min_open=True, max_open=True),
              default=0.99, show_default=True,
              help="Initial threshold quantile saved in the checkpoint.")
@click.option("--checkpoint_dir", type=click.Path(file_okay=False), default=None,
              help="Output directory. Defaults to a separate runtime-quantile/k directory.")
@click.option("--overwrite", is_flag=True,
              help="Explicitly allow replacing checkpoints in the output directory.")
def main(**kwargs):
    pass

@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=2, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--augment", is_flag=True)
def dataset(
        name,
        data_path,
        subdatasets,
        train_val_split,
        batch_size,
        resize,
        imagesize,
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                rotate_degrees=rotate_degrees,
                translate=translate,
                brightness_factor=brightness,
                contrast_factor=contrast,
                saturation_factor=saturation,
                gray_p=gray,
                h_flip_p=hflip,
                v_flip_p=vflip,
                scale=scale,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            LOGGER.info(f"Dataset: train={len(train_dataset)} test={len(test_dataset)}")

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=4,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)

@main.command("net")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--embedding_size", type=int, default=1024)
@click.option("--meta_epochs", type=int, default=1)
@click.option("--aed_meta_epochs", type=int, default=1)
@click.option("--gan_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=None)
@click.option("--noise_std", type=float, default=0.05)
@click.option("--dsc_margin", type=float, default=0.8)
@click.option("--dsc_lr", type=float, default=0.0002)
@click.option("--auto_noise", type=float, default=0)
@click.option("--train_backbone", is_flag=True)
@click.option("--cos_lr", is_flag=True)
@click.option("--pre_proj", type=int, default=0)
@click.option("--proj_layer_type", type=int, default=0)
@click.option("--mix_noise", type=int, default=1)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        embedding_size,
        meta_epochs,
        aed_meta_epochs,
        gan_epochs,
        noise_std,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        dsc_lr,
        auto_noise,
        train_backbone,
        cos_lr,
        pre_proj,
        proj_layer_type,
        mix_noise,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_simplenet(input_shape, device):
        simplenets = []
        for backbone_name, layers_to_extract_from in zip(
                backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            simplenet_inst = simplenet.SimpleNet(device)
            simplenet_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                embedding_size=embedding_size,
                meta_epochs=meta_epochs,
                aed_meta_epochs=aed_meta_epochs,
                gan_epochs=gan_epochs,
                noise_std=noise_std,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                auto_noise=auto_noise,
                train_backbone=train_backbone,
                cos_lr=cos_lr,
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,
                mix_noise=mix_noise,
            )
            simplenets.append(simplenet_inst)
        return simplenets

    return ("get_simplenet", get_simplenet)

@main.result_callback()
def run(
        methods,
        results_path,
        log_project,
        log_group,
        run_name,
        seed,
        test,
        gpu,
        neighborhood,
        k,
        quantile,
        checkpoint_dir,
        overwrite,
):
    methods = {key: item for (key, item) in methods}

    list_of_dataloaders = methods["get_dataloaders"](seed)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            "true_spatial_low_rank_gaussian",
            f"runtime_quantiles_v2_k{k}",
        )
    checkpoint_paths = {
        dataloaders["training"].name: os.path.join(
            checkpoint_dir, f'{dataloaders["training"].name}.pt'
        )
        for dataloaders in list_of_dataloaders
    }
    existing_paths = [
        path for path in checkpoint_paths.values() if os.path.exists(path)
    ]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "Refusing to replace existing TSLRG checkpoint(s): {}. "
            "Choose another --checkpoint_dir or pass --overwrite explicitly."
            .format(", ".join(existing_paths))
        )

    device = utils.set_torch_device(gpu)

    utils.fix_seeds(seed)

    for dataloaders_count, dataloaders in enumerate(list_of_dataloaders):
        dataset_name = dataloaders["training"].name
        imagesize = dataloaders["training"].dataset.imagesize
        batch_size = dataloaders["training"].batch_size

        embedder: simplenet.SimpleNet = methods["get_simplenet"](imagesize, device)[0]

        all_patches = []

        for data in dataloaders["training"]:
            with torch.no_grad():
                embedding = embedder.embed(data["image"].to(device))[0]
                embedding = embedding.reshape(-1, 1296, embedding.shape[1])
            all_patches.append(embedding.cpu())

        all_patches = torch.cat(all_patches, dim=0)

        tslrg = TrueSpatialLowRankGaussian(
            k=k,
            quantile=quantile,
            neighborhood=neighborhood,
        )
        tslrg.fit(all_patches)
        LOGGER.info(
            "Fitted TSLRG for %s with requested k=%d, effective k=%d, "
            "quantile=%g, and %d calibration samples per patch",
            dataset_name,
            tslrg.requested_k,
            tslrg.k,
            tslrg.quantile,
            tslrg.sorted_calibration_scores.shape[1],
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = checkpoint_paths[dataset_name]
        torch.save(tslrg.state_dict(), checkpoint_path)
        LOGGER.info("Saved TSLRG checkpoint to %s", checkpoint_path)

        # lrg = LowRankGaussian.fit(all_patches.to(device))
        # torch.save(lrg.state_dict(), f"low_rank_gaussian/{dataset_name}.pt")


if __name__ == "__main__":
    main()
