import os
import glob

from typing import Optional, Callable
from torchvision.io import decode_image
from torch.utils.data import Dataset


class MvtecAdDataset(Dataset):
    """
    Custom PyTorch Dataset for the MVTec AD dataset.
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root_dir (str): MVTec AD dataset root path
            category (str): Chosen category ( ex: 'capsule', 'bootle',...)
            split (str, optional): "train" or "test". Defaults to "train".
            transform (Optional[Callable], optional): Transformation to apply to the image.
            Defaults to None.
        """
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform

        if self.split not in {"train", "test"}:
            raise ValueError("Expected value must be 'train' or 'test'")

        # build useful paths
        self.category_dir = os.path.join(self.root_dir, self.category)
        self.split_dir = os.path.join(self.category_dir, self.split)

        # load sapmles's list
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        """This function builds and returns the list of samples.
        Each sample can for example , contains:
            - image_path,
            - label ( 0 : normal , 1 : abnormal or defective image)
            - defect_type
        """
        samples = []

        if self.split == "train":
            pattern = os.path.join(self.split_dir, "good", "*.[jp][pn]g")
            for image_path in glob.iglob(pattern):
                samples.append(
                    {"image_path": image_path, "label": 0, "defect_type": "good"}
                )

        elif self.split == "test":
            pattern = os.path.join(self.split_dir, "*", "*.[jp][pn]g")
            for image_path in glob.iglob(pattern):

                defect_type = os.path.basename(os.path.dirname(image_path))
                samples.append(
                    {
                        "image_path": image_path,
                        "label": 0 if defect_type == "good" else 1,
                        "defect_type": defect_type,
                    }
                )
        return samples

    def __len__(self):
        """Returns the total number of sample"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """This function takes a number as input and returns the sample located
        at index idx.

        Args:
            idx (dict): _description_
        """
        sample = self.samples[idx]

        # Retrieve the image path, label, and default type
        image_path = sample.get("image_path", None)
        label = sample.get("label", None)
        defect_type = sample.get("defect_type", None)

        # transform the image into a tensor
        image = decode_image(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "defect_type": defect_type,
            "image_path": image_path,
        }
