import torchaudio
import torch
from torch.utils.data import Dataset
import numpy as np
import os

# musicnet没有分开的，只能各个乐器混合起来训练
# 但是midi是分乐器的，可以用于乐器分离的训练
class MusicNetData(Dataset):
    def __init__(self, folder: str | list[str], input = '.wav', output = '.npy'):
        """
        folder: 数据集文件夹, 支持字符串或字符串列表
        input: 输入数据后缀 只能是.wav
        output: 输出数据后缀 只能是.npy
        """
        self.input = input
        self.output = output

        # 支持 folder 为字符串或字符串列表
        if isinstance(folder, str):
            folder_list = [folder]
        else:
            folder_list = list(folder)

        self.data_paths = []
        for f in folder_list:
            abs_folder = os.path.abspath(f)
            self.data_paths.extend([
                os.path.join(abs_folder, file)[:-4]
                for file in os.listdir(abs_folder)
                if file.endswith('.wav') and os.path.isfile(os.path.join(abs_folder, file))
            ])

        self.data_num = len(self.data_paths)
    
    def __len__(self):
        return self.data_num

    @staticmethod
    def get_data_by_suffix(path):
        if path.endswith('.npy'):
            return torch.tensor(np.load(path), dtype=torch.float32)
        elif path.endswith('.wav'):
            return torchaudio.load(path, normalize=True)[0]  # [channel, time]
        else:
            raise ValueError("Unsupported file type")

    def __getitem__(self, idx):
        """
        idx: 0 ~ len(self) - 1
        return: (input, output)
        input: .wav: [channel, time]
        output: .npy: [mix, channel, time]
        """
        data_path = self.data_paths[idx]

        # 读取数据
        input_data = self.get_data_by_suffix(data_path + self.input)
        output_data = self.get_data_by_suffix(data_path + self.output)

        return input_data, output_data