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
    T_p = quantile of Mahalanobis at patch p
    """

    def __init__(self, k=64, quantile=0.99):
        self.k = k
        self.quantile = quantile

        self.mu = None        # (P, C)
        self.U = None         # (P, C, k)
        self.Lambda = None    # (P, k)
        self.eps = None       # (P,)
        self.T = None         # (P,)   <-- per patch

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    @torch.no_grad()
    def fit(self, X):
        """
        X: (N, P, C)
        """
        N, P, C = X.shape

        mu = X.mean(dim=0)
        Xc = X - mu.unsqueeze(0)

        U_list = []
        Lambda_list = []
        eps_list = []
        T_list = []

        for p in range(P):
            Yp = Xc[:, p, :]  # (N, C)

            # SVD
            U_svd, S_svd, Vh = torch.linalg.svd(Yp, full_matrices=False)

            eigvals = (S_svd ** 2) / (N - 1)
            r = eigvals.shape[0]
            k_eff = min(self.k, r)

            V_k = Vh[:k_eff].T
            Lambda_k = eigvals[:k_eff]

            # Paper epsilon: 0.5 * median of trailing eigenvalues, else 1e-2
            if k_eff < r:
                eps_p = 0.5 * torch.median(eigvals[k_eff:])
            else:
                eps_p = torch.tensor(1e-2, device=X.device)

            # ---- Compute Mahalanobis for this patch ----
            d = Yp  # already centered
            proj = d @ V_k
            term1 = (proj ** 2 / Lambda_k).sum(dim=1)

            residual = d - proj @ V_k.T
            term2 = (residual ** 2).sum(dim=1) / eps_p

            scores_p = term1 + term2

            T_p = torch.quantile(scores_p, self.quantile)

            U_list.append(V_k)
            Lambda_list.append(Lambda_k)
            eps_list.append(eps_p)
            T_list.append(T_p)

        self.mu = mu
        self.U = torch.stack(U_list, dim=0)
        self.Lambda = torch.stack(Lambda_list, dim=0)
        self.eps = torch.stack(eps_list, dim=0)
        self.T = torch.stack(T_list, dim=0)  # (P,)

    # --------------------------------------------------
    # PER-PATCH SAMPLING
    # --------------------------------------------------
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
    def generate_anomaly_at_patch(self, p, B, delta=1):
        """
        Generate B anomalies just outside patch p's Mahalanobis boundary.
        Mahalanobis radius: r ~ U(sqrt(T_p), sqrt(T_p) + delta).
        Returns: (B, C)
        """
        device = 'cuda:0'
        C = self.mu.shape[1]
        T_p = self.T[p].to(device)

        u = torch.randn(B, C, device=device)
        u = u / u.norm(dim=1, keepdim=True)

        r = torch.sqrt(T_p) + delta * torch.rand(B, 1, device=device)
        w = u * r

        return self.mu[p].to(device) + self._sigma_half_w(p, w, device)

    @torch.no_grad()
    def generate_normal_at_patch(self, p, B):
        """
        Draw B samples from N(mu_p, Sigma_p).
        Returns: (B, C)
        """
        device = 'cuda:0'
        C = self.mu.shape[1]

        w = torch.randn(B, C, device=device)

        return self.mu[p].to(device) + self._sigma_half_w(p, w, device)

    # --------------------------------------------------
    # BULK SAMPLING (all patches)
    # --------------------------------------------------
    @torch.no_grad()
    def generate_anomalies(self, B, delta=1):
        """Generate anomalies just outside each patch's boundary. Returns (B, P, C)."""
        P = self.mu.shape[0]
        return torch.stack(
            [self.generate_anomaly_at_patch(p, B, delta) for p in range(P)], dim=1
        )

    @torch.no_grad()
    def generate_normal_features(self, B):
        """Draw B samples per patch from N(mu_p, Sigma_p). Returns (B, P, C)."""
        P = self.mu.shape[0]
        return torch.stack(
            [self.generate_normal_at_patch(p, B) for p in range(P)], dim=1
        )

    def state_dict(self):
        return {
            "k": self.k,
            "quantile": self.quantile,
            "mu": self.mu,
            "U": self.U,
            "Lambda": self.Lambda,
            "eps": self.eps,
            "T": self.T,
        }

    def load_state_dict(self, state):
        self.k = state["k"]
        self.quantile = state["quantile"]
        self.mu = state["mu"]
        self.U = state["U"]
        self.Lambda = state["Lambda"]
        self.eps = state["eps"]
        self.T = state["T"]

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
):
    methods = {key: item for (key, item) in methods}

    list_of_dataloaders = methods["get_dataloaders"](seed)

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

        tslrg = TrueSpatialLowRankGaussian()
        tslrg.fit(all_patches)
        os.makedirs("true_spatial_low_rank_gaussian", exist_ok=True)
        torch.save(tslrg.state_dict(), f"true_spatial_low_rank_gaussian/{dataset_name}.pt")

        # lrg = LowRankGaussian.fit(all_patches.to(device))
        # torch.save(lrg.state_dict(), f"low_rank_gaussian/{dataset_name}.pt")


if __name__ == "__main__":
    main()

