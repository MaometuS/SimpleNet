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
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
}

@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--fake_method", type=str, default="method_1")
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

def generate_fake_feats(variance_mlp, debatched_true_feats):
    predicted_var = variance_mlp(debatched_true_feats)
    std = torch.clamp(torch.sqrt(predicted_var + 1e-6), max=10.0)
    noise = torch.randn_like(debatched_true_feats) * std * 0.01
    fake_feats = (debatched_true_feats + noise.detach())
    return fake_feats

def generate_fake_feats_directional_1(variance_mlp, debatched_true_feats):
    predicted_var = variance_mlp(debatched_true_feats)
    std = torch.clamp(torch.sqrt(predicted_var + 1e-6), max=10.0)
    noise = torch.randn_like(debatched_true_feats) * std * 1.1
    feat_mean = debatched_true_feats.mean(dim=1, keepdim=True)
    direction = debatched_true_feats - feat_mean
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)
    noise = noise * direction
    noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-6)
    noise = noise.detach()

    fake_feats = (debatched_true_feats + noise.detach())
    return fake_feats

def generate_fake_feats_directional_2(variance_mlp, debatched_true_feats):
    predicted_var = variance_mlp(debatched_true_feats)
    std = torch.clamp(torch.sqrt(predicted_var + 1e-6), max=10.0)
    feat_mean = debatched_true_feats.mean(dim=1, keepdim=True)
    direction = debatched_true_feats - feat_mean
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)  # [B,P,D]
    dir_var = (std ** 2 * direction ** 2).sum(dim=-1, keepdim=True)  # [B,P,1]
    dir_std = torch.sqrt(dir_var + 1e-6) * 1.1  # [B,P,1]
    noise = torch.randn_like(debatched_true_feats)
    noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-6)  # unit norm
    noise = noise * dir_std
    noise = noise.detach()

    fake_feats = (debatched_true_feats + noise.detach())
    return fake_feats


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
        fake_method,
):
    methods = {key: item for (key, item) in methods}

    true_feats_all = []
    fake_feats_all = []

    list_of_dataloaders = methods["get_dataloaders"](seed)
    device = utils.set_torch_device(gpu)

    with torch.no_grad():
        for _, dataloaders in enumerate(list_of_dataloaders):
            dataset_name = dataloaders["training"].name
            imagesize = dataloaders["training"].dataset.imagesize

            embedder: simplenet.SimpleNet = methods["get_simplenet"](imagesize, device)[0]

            variance_mlp = VarianceMLP().to(device)
            variance_mlp.load_state_dict(torch.load(f"variance_mlp/{dataset_name}_variance_mlp_25.pth"))
            variance_mlp.eval()

            for data in dataloaders["training"]:
                embedding = embedder.embed(data["image"].to(device))[0]

                debatched_embedding = embedding.reshape(len(data["image"]), -1, embedding.shape[1])
                match fake_method:
                    case "method_1":
                        fake_feats = generate_fake_feats(variance_mlp, debatched_embedding).reshape(embedding.shape)
                    case "method_2":
                        fake_feats = generate_fake_feats_directional_1(variance_mlp, debatched_embedding).reshape(embedding.shape)
                    case "method_2":
                        fake_feats = generate_fake_feats_directional_2(variance_mlp, debatched_embedding).reshape(embedding.shape)

                idx_true = torch.randperm(true_feats.shape[0])[:50]
                idx_fake = torch.randperm(fake_feats.shape[0])[:50]

                true_feats_all.append(embedding.cpu()[idx_true])
                fake_feats_all.append(fake_feats.cpu()[idx_fake])

        X = torch.cat([true_feats_all, fake_feats_all], dim=0)
        labels = torch.cat([torch.zeroes(true_feats_all.shape[0]), torch.ones(fake_feats_all.shape[0])])
        X = torch.nn.functional.normalize(X, dim=1)

        X_np = X.detach().cpu().numpy()
        Y_np = labels.cpu().numpy()

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            init="pca",
            random_state=0
        )

        X_2d = tsne.fit_transform(X_np)

        plt.figure(figsize=(8, 6))

        plt.scatter(
            X_2d[Y_np == 0, 0],
            X_2d[Y_np == 0, 1],
            s=5,
            alpha = 0.5,
            label="True"
        )

        plt.scatter(
            X_2d[Y_np == 1, 0],
            X_2d[Y_np == 1, 1],
            s=5,
            alpha = 0.5,
            label="Fake"
        )

        print("I am here")

        plt.legend()

        save_path = "tsne_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()
