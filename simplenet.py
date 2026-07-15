# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
from low_rank_gaussian import LowRankGaussian, SpatialLowRankGaussian, TrueSpatialLowRankGaussian
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images

from variance_mlp import VarianceMLP

LOGGER = logging.getLogger(__name__)

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1

class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1, 
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        **kwargs,
    ):
        pid = os.getpid()
        def show_mem():
            return(psutil.Process(pid).memory_info())

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin 

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

        self.lgr = None
        self.slrg = None
        self.tslrg = None
        # Anomaly-generation mode for TrueSpatialLowRankGaussian.
        # "default": PDF formulation (full-sphere). "subspace": in-U_k only.
        # "anchored": real-normal anchor + in-U_k shift. Override via env var.
        self.tslrg_anomaly_mode = os.environ.get("TSLRG_ANOMALY_MODE", "default")
        # Anchored anomalies are always constructed in the original embedding
        # space and then pre-projected together with their real anchors.  This
        # switch controls projection of the non-anchored TSLRG modes only.
        self.tslrg_project_fake_feats = os.environ.get(
            "TSLRG_PROJECT_FAKE_FEATS", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if self.tslrg_project_fake_feats and self.pre_proj <= 0:
            raise ValueError(
                "TSLRG_PROJECT_FAKE_FEATS requires --pre_proj to be greater than 0"
            )
        # Optional explicit Mahalanobis radius. If set, overrides the T_p-based
        # default. Use a small value (e.g., 0.5–2) to mimic SimpleNet's noise scale.
        self.tslrg_delta = float(os.environ.get("TSLRG_DELTA", "1"))
        if not math.isfinite(self.tslrg_delta) or self.tslrg_delta < 0:
            raise ValueError("TSLRG_DELTA must be a finite, non-negative number")
        _r = os.environ.get("TSLRG_RADIUS")
        self.tslrg_radius = float(_r) if _r else None
        self.tslrg_radius_mode = os.environ.get("TSLRG_RADIUS_MODE", "threshold")
        self.tslrg_patch_mask_mode = os.environ.get("TSLRG_PATCH_MASK_MODE", "all")
        self.tslrg_patch_mask_ratio = float(os.environ.get("TSLRG_PATCH_MASK_RATIO", "0.15"))
        self.tslrg_patch_mask_block = int(os.environ.get("TSLRG_PATCH_MASK_BLOCK", "5"))
        self.tslrg_refine_steps = int(os.environ.get("TSLRG_REFINE_STEPS", "0"))
        self.tslrg_refine_step_size = float(os.environ.get("TSLRG_REFINE_STEP_SIZE", "0.1"))
        _refine_radius = os.environ.get("TSLRG_REFINE_MAX_RADIUS")
        self.tslrg_refine_max_radius = float(_refine_radius) if _refine_radius else None

        self.variance_mlp = VarianceMLP().to(self.device)
        self.variance_mlp.load_state_dict(torch.load("variance_mlp_25.pth"))
        self.variance_mlp.eval()

    def load_variance_mlp(self, dataset_name):
        self.variance_mlp = VarianceMLP().to(self.device)
        self.variance_mlp.load_state_dict(torch.load(f"variance_mlp/{dataset_name}_variance_mlp_25.pth"))
        self.variance_mlp.eval()
    
    def load_low_rank_gaussian(self, dataset_name):
        ckpt = torch.load(f"low_rank_gaussian/{dataset_name}.pt")
        self.lrg = LowRankGaussian(
            ckpt["mu"],
            ckpt["Uk"],
            ckpt["lambdak"],
            ckpt["eps"].item(),
            ckpt["T"].item(),
        ).to(self.device)

    def load_spatial_low_rank_gaussian(self, dataset_name):
        ckpt = torch.load(f"spatial_low_rank_gaussian/{dataset_name}.pt")
        self.slrg = SpatialLowRankGaussian(
            ckpt["mu_p"],
            ckpt["Uk"],
            ckpt["lambdak"],
            ckpt["eps"].item(),
            ckpt["T"].item(),
        ).to(self.device)

    def load_true_spatial_low_rank_gaussian(self, dataset_name):
        ckpt = torch.load(f"true_spatial_low_rank_gaussian/{dataset_name}.pt")
        self.tslrg = TrueSpatialLowRankGaussian()
        self.tslrg.load_state_dict(ckpt)

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)
    

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling        


        return features, patch_shapes

    
    def test(self, training_data, test_data, save_segmentation_images):

        ckpt_path = os.path.join(self.ckpt_dir, "models.ckpt")
        if os.path.exists(ckpt_path):
            state_dicts = torch.load(ckpt_path, map_location=self.device)
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)
            
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        # Compute PRO score & PW Auroc for all images
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]

        return auroc, full_pixel_auroc
    
    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt 
        )["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)


            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
                # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), 
                                            norm_segmentations)
        else:
            full_pixel_auroc = -1 
            pro = -1

        return auroc, full_pixel_auroc, pro
        
    
    def train(self, training_data, test_data):

        
        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            self.predict(training_data, "train_")
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
            
            return auroc, full_pixel_auroc, anomaly_pixel_auroc
        
        def update_state_dict(d):
            
            state_dict["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})

        best_record = None
        for i_mepoch in range(self.meta_epochs):

            self._train_discriminator(training_data, current_meta_epoch=i_mepoch)

            # torch.cuda.empty_cache()
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
            self.logger.logger.add_scalar("pro", pro, i_mepoch)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(state_dict)
                # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro 
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")
        
        torch.save(state_dict, ckpt_path)
        
        return best_record
            
    def generate_anomailes(self, total, delta=1.0):
        C = self.lrg.mu.shape[0]

        u = torch.randn(total, C, device=self.lrg.mu.device)
        u = u / u.norm(dim=1, keepdim=True)

        r = torch.rand(total, device=self.lrg.mu.device) * 1 + torch.sqrt(self.lrg.T)

        w = u * r.unsqueeze(1)

        # Σ^{1/2} w
        proj = w @ self.lrg.Uk
        low_rank = (proj * torch.sqrt(self.lrg.lambdak)) @ self.lrg.Uk.T

        recon = proj @ self.lrg.Uk.T
        residual = w - recon
        iso = torch.sqrt(self.lrg.eps) * residual

        fake_feats = self.lrg.mu + low_rank + iso

        return fake_feats
    
    def generate_spatial_anomalies(self, B, delta=1.5):
        P, C = self.slrg.mu_p.shape
        total = B * P

        u = torch.randn(total, C, device=self.slrg.mu_p.device)
        u = u / u.norm(dim=1, keepdim=True)

        r = torch.rand(total, device=self.slrg.mu_p.device) * delta + torch.sqrt(self.slrg.T)
        w = u * r.unsqueeze(1)

        proj = w @ self.slrg.Uk
        low_rank = (proj * torch.sqrt(self.slrg.lambdak)) @ self.slrg.Uk.T

        recon = proj @ self.slrg.Uk.T
        residual = w - recon
        iso = torch.sqrt(self.slrg.eps) * residual

        shift = low_rank + iso
        shift = shift.reshape(B, P, C)

        x_fake = self.slrg.mu_p.unsqueeze(0) + shift

        return x_fake

    def _sample_patch_mask(self, B, P, device):
        mode = self.tslrg_patch_mask_mode
        if mode == "all":
            return torch.ones(B, P, dtype=torch.bool, device=device)

        if self.tslrg.spatial_shape is None:
            H = int(round(P ** 0.5))
            W = H
        else:
            H, W = self.tslrg.spatial_shape
        if H * W != P:
            raise ValueError(f"Cannot build patch mask for P={P}, shape={(H, W)}")

        ratio = min(max(self.tslrg_patch_mask_ratio, 0.0), 1.0)
        target = max(1, int(round(P * ratio)))

        if mode == "random":
            mask = torch.zeros(B, P, dtype=torch.bool, device=device)
            for b in range(B):
                idx = torch.randperm(P, device=device)[:target]
                mask[b, idx] = True
            return mask

        if mode == "block":
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            block = max(1, self.tslrg_patch_mask_block)
            half = block // 2
            for b in range(B):
                selected = 0
                while selected < target:
                    cy = torch.randint(0, H, (1,), device=device).item()
                    cx = torch.randint(0, W, (1,), device=device).item()
                    y0, y1 = max(0, cy - half), min(H, cy + half + 1)
                    x0, x1 = max(0, cx - half), min(W, cx + half + 1)
                    mask[b, y0:y1, x0:x1] = True
                    selected = int(mask[b].sum().item())
            return mask.reshape(B, P)

        raise ValueError("unknown TSLRG_PATCH_MASK_MODE={!r}; expected one of "
                         "'all', 'random', 'block'".format(mode))

    def _project_delta_to_tslrg_subspace(self, delta):
        U = self.tslrg.U.to(delta.device)
        coeff = torch.einsum("bpc,pck->bpk", delta, U)
        return torch.einsum("bpk,pck->bpc", coeff, U), coeff

    def _mahal_radius_from_delta(self, delta):
        U = self.tslrg.U.to(delta.device)
        Lambda = self.tslrg.Lambda.to(delta.device).clamp_min(1e-12)
        coeff = torch.einsum("bpc,pck->bpk", delta, U)
        return torch.sqrt((coeff ** 2 / Lambda.unsqueeze(0)).sum(dim=-1).clamp_min(1e-12))

    def _clamp_tslrg_subspace_delta(self, base, candidate, max_radius):
        delta = candidate - base
        U = self.tslrg.U.to(delta.device)
        Lambda = self.tslrg.Lambda.to(delta.device).clamp_min(1e-12)
        coeff = torch.einsum("bpc,pck->bpk", delta, U)
        white = coeff / torch.sqrt(Lambda).unsqueeze(0)
        norm = torch.norm(white, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(norm), max_radius.unsqueeze(-1) / norm)
        clamped = white * scale
        projected = torch.einsum(
            "bpk,pck->bpc",
            clamped * torch.sqrt(Lambda).unsqueeze(0),
            U,
        )
        return base + projected

    def _refine_fake_features(
        self,
        true_feats,
        fake_feats,
        patch_mask,
        project_for_discriminator=False,
    ):
        """Refine candidates while keeping TSLRG geometry in anchor space."""
        if self.tslrg_refine_steps <= 0:
            return fake_feats

        B, P, C = fake_feats.shape
        base = true_feats.detach()
        x = fake_feats.detach()
        initial_radius = self._mahal_radius_from_delta(x - base).detach()
        if self.tslrg_refine_max_radius is not None:
            max_radius = torch.full_like(initial_radius, self.tslrg_refine_max_radius)
        else:
            max_radius = initial_radius.clamp_min(1e-6)

        for _ in range(self.tslrg_refine_steps):
            x = x.detach().requires_grad_(True)
            discriminator_input = x.reshape(B * P, C)
            if project_for_discriminator:
                discriminator_input = self.pre_projection(discriminator_input)
            flat_scores = self.discriminator(discriminator_input).reshape(B, P)
            objective = flat_scores[patch_mask].mean()
            grad = torch.autograd.grad(objective, x, only_inputs=True)[0]
            grad, _ = self._project_delta_to_tslrg_subspace(grad)
            grad_norm = torch.norm(grad, dim=-1, keepdim=True).clamp_min(1e-12)
            candidate = x - self.tslrg_refine_step_size * grad / grad_norm
            x = self._clamp_tslrg_subspace_delta(base, candidate, max_radius)
            x = torch.where(patch_mask.unsqueeze(-1), x, base)

        return x.detach()

    def _maybe_project_tslrg_fake_features(self, fake_feats):
        """Optionally pre-project non-anchored generated TSLRG features."""
        if not self.tslrg_project_fake_feats:
            return fake_feats

        B, P, C = fake_feats.shape
        return self.pre_projection(fake_feats.reshape(B * P, C)).reshape(B, P, C)

    def _train_discriminator(self, input_data, current_meta_epoch=1):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)
                    unprojected_true_feats = self._embed(
                        img, evaluation=False
                    )[0]
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(unprojected_true_feats)
                    else:
                        true_feats = unprojected_true_feats

                    #generate anomalous features

                    # debatched_true_feats = true_feats.reshape(8, -1, true_feats.shape[1])
                    # with torch.no_grad():
                        # predicted_var = self.variance_mlp(debatched_true_feats)
                        # std = torch.clamp(torch.sqrt(predicted_var + 1e-6), max=10.0)
                        # noise = torch.randn_like(debatched_true_feats) * std * 1.1
                        # feat_mean = debatched_true_feats.mean(dim=1, keepdim=True)
                        # direction = debatched_true_feats - feat_mean
                        # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)
                        # noise = noise * direction
                        # noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-6)
                        # noise = noise.detach()

                        # predicted_var = self.variance_mlp(debatched_true_feats)
                        # std = torch.clamp(torch.sqrt(predicted_var + 1e-6), max=10.0)
                        # feat_mean = debatched_true_feats.mean(dim=1, keepdim=True)
                        # direction = debatched_true_feats - feat_mean
                        # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)  # [B,P,D]
                        # dir_var = (std ** 2 * direction ** 2).sum(dim=-1, keepdim=True)  # [B,P,1]
                        # dir_std = torch.sqrt(dir_var + 1e-6) # [B,P,1]
                        # noise = torch.randn_like(debatched_true_feats)
                        # noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-6)  # unit norm
                        # noise = noise * dir_std
                        # noise = noise.detach()

                    # noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    # noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
                    # old_noise = torch.stack([
                    #     torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                    #     for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
                    # old_noise = (old_noise * noise_one_hot.unsqueeze(-1)).sum(1)

                    # fake_feats = true_feats + old_noise

                    P = self.tslrg.mu.shape[0]
                    C = true_feats.shape[1]
                    B = true_feats.shape[0] // P
                    if self.tslrg_anomaly_mode in ("simplenet_noise", "noise"):
                        patch_mask = torch.ones(B, P, dtype=torch.bool, device=true_feats.device)
                        noise_idxs = torch.randint(
                            0,
                            self.mix_noise,
                            torch.Size([unprojected_true_feats.shape[0]]),
                            device=self.device,
                        )
                        noise_one_hot = torch.nn.functional.one_hot(
                            noise_idxs, num_classes=self.mix_noise
                        ).to(self.device)
                        noise = torch.stack([
                            torch.normal(
                                0,
                                self.noise_std * 1.1**k,
                                unprojected_true_feats.shape,
                                device=self.device,
                            )
                            for k in range(self.mix_noise)
                        ], dim=1)
                        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                        fake_feats = unprojected_true_feats + noise
                        if self.pre_proj > 0:
                            fake_feats = self.pre_projection(fake_feats)
                    elif self.tslrg_anomaly_mode == "anchored":
                        patch_mask = self._sample_patch_mask(B, P, true_feats.device)
                        anchors = unprojected_true_feats.detach().reshape(B, P, C)
                        fake_feats = self.tslrg.generate_anomalies(
                            B, delta=self.tslrg_delta,
                            mode="anchored", anchors=anchors,
                            radius=self.tslrg_radius,
                            radius_mode=self.tslrg_radius_mode,
                            device=self.device,
                        )
                        fake_feats = torch.where(
                            patch_mask.unsqueeze(-1),
                            fake_feats.to(self.device),
                            anchors,
                        )
                        fake_feats = self._refine_fake_features(
                            anchors,
                            fake_feats,
                            patch_mask,
                            project_for_discriminator=self.pre_proj > 0,
                        )
                        if self.pre_proj > 0:
                            fake_feats = self.pre_projection(
                                fake_feats.reshape(B * P, C)
                            ).reshape(B, P, C)
                        fake_feats = fake_feats.reshape(true_feats.shape)
                    else:
                        patch_mask = self._sample_patch_mask(B, P, true_feats.device)
                        true_feats_bpc = true_feats.detach().reshape(B, P, C)
                        fake_feats = self.tslrg.generate_anomalies(
                            B, delta=self.tslrg_delta,
                            mode=self.tslrg_anomaly_mode,
                            radius=self.tslrg_radius,
                            radius_mode=self.tslrg_radius_mode,
                            device=self.device,
                        )
                        fake_feats = self._maybe_project_tslrg_fake_features(fake_feats)
                        fake_feats = torch.where(
                            patch_mask.unsqueeze(-1),
                            fake_feats.to(self.device),
                            true_feats_bpc,
                        )
                        fake_feats = self._refine_fake_features(
                            true_feats_bpc, fake_feats, patch_mask
                        )
                        fake_feats = fake_feats.reshape(true_feats.shape)

                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(true_feats):].reshape(B, P)
                    selected_fake_scores = fake_scores[patch_mask]

                    th = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (selected_fake_scores.detach() < -th).sum() / len(selected_fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(selected_fake_scores + th, min=0)

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
                    self.logger.logger.add_scalar(
                        "fake_patch_ratio", patch_mask.float().mean(), self.logger.g_iter
                    )

                    loss = true_loss.mean() + fake_loss.mean()
                    # loss += lambda_anchor*anchor_loss
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                    # print("Th: ", th)
                    # print("Margin anchor: ", margin_anchor)
                    # print("Anchor loss: ", anchor_loss)
                    # print("Loss: ", loss)
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                if self.cos_lr:
                    self.dsc_schl.step()
                
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()


        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True, 
                                                 evaluation=True)
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
