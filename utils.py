import datasets
from typing import List, Union
import os
from torch.utils.data import DataLoader
from torchvision import transforms

def get_data_transformers(image_size, image_mean=None, image_std=None):
    X_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if image_mean and image_std:
        X_transforms.append(
            transforms.Normalize(
                mean=image_mean,
                std=image_std
            )
        )
    return {
        "train": transforms.Compose(
            X_transforms
        ),
        "val": transforms.Compose(
            X_transforms
        ),
    }

imagenet_inverse_normalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                       std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                                         std = [ 1., 1., 1. ]),
                                                ])


def get_dataloaders(
    data_dir: Union[str, List[str]], image_size, image_mean=None, image_std=None,
    req_dataloaders=["train", "val"], batch_size=4, num_workers=4,
):
    x_transform = get_data_transformers(image_size, image_mean, image_std)
    xray_datasets = {
        x: datasets.Flickr30kDataset(
            os.path.join(data_dir),
            task=x,
            transform=x_transform[x]
            if x != "test"
            else x_transform["val"],
        )
        for x in req_dataloaders
    }

    dataloaders = {
        x: DataLoader(
            xray_datasets[x], batch_size=batch_size, shuffle=True,
            num_workers=num_workers
        )
        for x in req_dataloaders
    }
    dataset_sizes = {x: len(xray_datasets[x]) for x in req_dataloaders}
    return dataloaders, dataset_sizes