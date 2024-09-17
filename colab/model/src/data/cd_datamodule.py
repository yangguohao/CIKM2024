import paddle
import numpy as np
import sys
from pathlib import Path
from src.data.base_datamodule import BaseDataModule
from src.data.velocity_datamodule import read


class CdDataset(paddle.io.Dataset):
    def __init__(self, input_dir, index_list):
        self.cd_list = np.loadtxt("/content/Test/Dataset_1/Feature_File/watertight_meshes.txt", delimiter=",", dtype=str, encoding='utf-8')[:,2][1:].astype(np.float32)
        self.input_dir = input_dir
        self.index_list = index_list
        self.len = len(index_list)

    def __getitem__(self, index):
        cd_label = self.cd_list[index]
        obj_name = self.index_list[index]
        data_dict = read(self.input_dir / f"{obj_name}.obj")
        data_dict["cd"] = cd_label
        return data_dict

    def __len__(self):
        return self.len


class CdDataModule(BaseDataModule):
    def __init__(self, train_data_dir, test_data_dir, train_index_list, test_index_list):
        BaseDataModule.__init__(self)
        self.train_data = CdDataset(Path(train_data_dir), train_index_list)
        self.test_data  = CdDataset(Path(test_data_dir),  test_index_list)
        self.train_indices = train_index_list
        self.test_indices = test_index_list
    
    def decode(self, x):
        return x
