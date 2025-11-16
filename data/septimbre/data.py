import torchaudio
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import itertools

DEFAULT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'small'))

class Instruments(Dataset):
    def __init__(self, folder = DEFAULT_FOLDER, mix = 2, input = '.cqt.npy', output = '.npy'):
        """
        folder: 数据集文件夹
        mix: 混合乐器数量
        input: 输入数据后缀
        output: 输出数据后缀
        """
        self.input = input
        self.output = output

        instrument_folders = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

        self.data_paths = []    # 二位列表，每个元素是一个乐器的所有无后缀路径
        for inst in instrument_folders:
            wav_files = [os.path.join(inst, f[:-4]) for f in os.listdir(inst) if f.endswith('.wav')]
            self.data_paths.append(wav_files)

        self.instrument_num = len(instrument_folders)
        self.data_per_inst = len(self.data_paths[0])
        self.mix = mix

    @property
    def mix(self):
        return self._mix

    @mix.setter
    def mix(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("mix must be an integer greater than or equal to 1")
        self._mix = value
        self.midi_combinations = self.data_per_inst ** value    # 一个数
        # 通过数学计算得到组合太难了，所以先列出所有可能的组合 是一个列表，每个元素是一个长为mix的元组
        self.inst_combinations = list(itertools.combinations(range(self.instrument_num), value))
        self.combinations = len(self.inst_combinations) * self.midi_combinations

    def __len__(self):
        return self.combinations

    @staticmethod
    def get_data_by_suffix(folder, suffix):
        if suffix.endswith('.npy'):
            return torch.tensor(np.load(folder), dtype=torch.float32)
        elif suffix.endswith('.wav'):
            return torchaudio.load(folder, normalize=True)[0]  # [channel, time]
        else:
            raise ValueError("Unsupported file type")

    def __getitem__(self, idx):
        """
        idx: 0 ~ len(self) - 1
        return: (input, output)
        input:  for .cqt.npy: [mix, 2, freq, time]
                for .wav: [mix, channel, time]
        output: for .npy: [mix, channel, time]
        """
        inst_comb_idx = idx // self.midi_combinations
        midi_comb_idx = idx % self.midi_combinations

        inst_comb = self.inst_combinations[inst_comb_idx]

        inputs = []
        outputs = []
        for i in range(self.mix):
            path = self.data_paths[inst_comb[i]][midi_comb_idx % self.data_per_inst]
            inputs.append(Instruments.get_data_by_suffix(path + self.input, self.input))
            outputs.append(Instruments.get_data_by_suffix(path + self.output, self.output))
            midi_comb_idx //= self.data_per_inst

        # 返回混合前的数据，混合由用户自行完成
        return torch.stack(inputs), torch.stack(outputs)
