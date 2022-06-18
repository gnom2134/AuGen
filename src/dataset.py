import torchxrayvision as xrv
import torchvision as trv


def prepare_dataset(dataset_name: str, imgpath: str, csvpath: str):
    transforms = trv.transforms.Compose(
        [
            xrv.datasets.XRayResizer(128),
            xrv.datasets.ToPILImage(),
            trv.transforms.ToTensor(),
            trv.transforms.Normalize(-1024, 2048),
        ]
    )

    dataset = None
    if dataset_name == "siim":
        dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
            imgpath, csvpath, transform=transforms
        )
    else:
        raise AttributeError(f"Unsupported dataset: {dataset_name}")

    return dataset
