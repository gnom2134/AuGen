import torchxrayvision as xrv
import torchvision as trv
import numpy as np
import torch
import cv2
from tqdm import tqdm
import os


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


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        xrv_dataset_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        augmented_dataset_dir: str
    ):
        super().__init__()
        counter = 0
        self.label_df = {
            "img": [],
            "lab": []
        }

        for j, train_batch in tqdm(enumerate(xrv_dataset_loader)):
            images = train_batch["img"]
            labels = train_batch["lab"]
            for i in range(images.shape[0]):
                img = (images[i].cpu().detach().numpy().squeeze(0) * 255).astype(np.uint8)
                img_path = os.path.join(augmented_dataset_dir, f"real_image_{counter}_label_{labels[i]}.jpg")
                self.label_df["img"].append(img_path)
                self.label_df["lab"].append(int(labels[i].item()))
                cv2.imwrite(img_path, img)
                counter += 1

        for i in tqdm(range(counter)):
            label = 0 if i < counter // 2 else 1
            img = (model.sample_random(label, size=1).cpu().detach().numpy().squeeze(0))
            img = (np.clip(255*img, 0, 255)).astype(np.uint8).squeeze()
            img_path = os.path.join(augmented_dataset_dir, f"fake_image_{i}_label_{label}.jpg")
            self.label_df["img"].append(img_path)
            self.label_df["lab"].append(label)
            cv2.imwrite(img_path, img)

    def __len__(self):
        return len(self.label_df["img"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not isinstance(idx, list):
            idx = [idx]

        imgs = []
        labels = []
        for i in idx:
            imgs.append(cv2.imread(self.label_df["img"][i], 0))
            labels.append(self.label_df["lab"][i])

        sample = {'img': torch.from_numpy(np.stack(imgs)), 'lab': torch.from_numpy(np.stack(labels))}

        return sample
