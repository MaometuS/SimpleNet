from numpy import var
from torch import nn
import torch.nn.functional as F
import torch
import math
import common
import utils
import backbones
import click
from tqdm import tqdm
import simplenet
from simplenet import SimpleNet


def compute_knn_indices(x, k):
    with torch.no_grad():
        dists = torch.cdist(x, x, p=2)
        dists.fill_diagonal_(float('inf'))
        return dists.topk(k, dim=1, largest=False).indices

def compute_knn_variances(x, knn_indices):
    N, D = x.shape
    k = knn_indices.shape[1]
    variances = torch.zeros(N, D, device=x.device)
    for i in range(N):
        neighbors = x[knn_indices[i]]  # [k, D]
        variances[i] = torch.var(neighbors, dim=0, unbiased=True)
    return variances

def generate_knn_target_variances(all_patch_embeddings, k=5):
    rep_features = all_patch_embeddings.mean(dim=1)  # [N, D]
    knn_indices = compute_knn_indices(rep_features, k)
    return compute_knn_variances(rep_features, knn_indices)

def generate_knn_target_variances_rep(rep_features, k=5):
    knn_indices = compute_knn_indices(rep_features, k)
    return compute_knn_variances(rep_features, knn_indices)

def downsample(x: torch.Tensor) -> torch.Tensor:
    H = W = int(x.shape[0]**0.5)
    x = x.view(H, W, x.shape[1])

    # Downsample by 2x using average pooling
    x = x.permute(2, 0, 1).unsqueeze(0)
    x_down = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

    # Flatten back
    x_down = x_down.squeeze(0).permute(1, 2, 0).reshape(-1, x.shape[1])
    return x_down

class VarianceMLP(nn.Module):
    def __init__(self, feature_dim=1536, hidden_dim=1024):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.variance_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Softplus()  # ensure positive variance
        )

    def forward(self, x):  # x: [B, 1296, 1536]
        x = self.token_mlp(x)     # [B, 1296, 1536]
        x = x.mean(dim=1)         # [B, 1536] — aggregate over patches
        var = self.variance_head(x)  # [B, 1536]
        return var

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


    # run_save_path = utils.create_storage_folder(
    #     results_path, log_project, log_group, run_name, mode="overwrite"
    # )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    model = VarianceMLP().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(10)):
        model.train()
        total_loss = 0

        for dataloader_count, dataloaders in tqdm(enumerate(list_of_dataloaders)):
            utils.fix_seeds(seed)
            dataset_name = dataloaders["training"].name
            imagesize = dataloaders["training"].dataset.imagesize

            embedder: SimpleNet = methods["get_simplenet"](imagesize, device)

            all_patches = []
            all_patches_mean = []

            for data in dataloaders["training"]:
                embedding = embedder.embed(data["image"].to(device))
                all_patches.append(embedding.cpu())
                all_patches_mean.append(embedding.mean(dim=1))

            target_variances = generate_knn_target_variances_rep(all_patches_mean, 5).cpu()
            all_patches_mean = []

            for data in dataloaders["training"]:
                embedding = embedder.embed(data["image"].to(device))
                embedding = embedding.reshape(len(data["image"]), -1, embedding.shape[1])
                k = min(len(data["image"]), 15)
                variances = generate_knn_target_variances(embedding, k)
                preds = model(embedding)
                loss = loss_fn(preds, variances)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloaders['training']):.60f}")
    torch.save(model.state_dict(), "variance_mlp_15.pth")

if __name__ == "__main__":
    main()

