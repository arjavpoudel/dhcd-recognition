from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class DHCDataset(Dataset):
    def __init__(self, npz_file, train=True):
        self.__dataset_npz = np.load(npz_file)
        self.train = train
        self.image_train = torch.from_numpy(
            self.__dataset_npz["arr_0"].astype(np.float32)
        ).unsqueeze(1)
        self.label_train = torch.from_numpy(
            self.__dataset_npz["arr_1"] - 1
        )  # nn.CrossEntropyLoss expects zero-based indexing
        self.image_test = torch.from_numpy(
            self.__dataset_npz["arr_2"].astype(np.float32)
        ).unsqueeze(1)
        self.label_test = torch.from_numpy(
            self.__dataset_npz["arr_3"] - 1
        )  # nn.CrossEntropyLoss expects zero-based indexing

    def __len__(self):
        if self.train:
            return len(self.image_train)
        else:
            return len(self.image_test)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.image_train[idx], self.label_train[idx]
        else:
            img, label = self.image_test[idx], self.label_test[idx]

        label = label.to(torch.long)

        return img, label

    def __repr__(self):
        intro_string = "Nepali (Devnagari) Character Dataset:"
        return f"{intro_string}\nTraining set contains {len(self.image_train)} images\nValidation Set contains {len(self.image_test)} images"
