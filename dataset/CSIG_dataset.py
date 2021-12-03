import os
import json
import numpy as np
import torch
# import torchaudio
from torch.utils.data import Dataset
from .preprocess import wavfile_to_examples
from tqdm import tqdm
from sklearn.model_selection import KFold


def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b


def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


# def read_audio(wav_path):
#     waveform, sample_rate = torchaudio.load(wav_path)
#     print("Shape of waveform:{}".format(waveform.size()))  # 音频大小
#     print("sample rate of waveform:{}".format(sample_rate))  # 采样率
#     specgram = torchaudio.transforms.MelSpectrogram()(waveform)
#     print("Shape of spectrogram:{}".format(specgram.size()))


# 对语音数据进行预处理
def preprocess(audio_paths, labels, root_path):
    print('Start preprocess audio......')
    process_features = []
    process_labels = []
    for i, audio_name in enumerate(tqdm(audio_paths)):
        audio_path = os.path.join(root_path, audio_name)
        wave_datas = wavfile_to_examples(audio_path)
        wave_datas = np.float32(wave_datas)
        for wave_data in wave_datas:
            process_labels.append(labels[i])
            process_features.append(np.expand_dims(wave_data, axis=0))
    print('Preprocess audio finished!!!')

    return process_features, process_labels


# 分割训练集和验证集
def split_data_by_k_fold(audio_paths, labels, phase='Train', n_splits=5, k=1):
    kf = KFold(n_splits=n_splits)
    for i, (train, valid) in enumerate(kf.split(X=audio_paths, y=labels)):
        if i == k:
            train_indexs, valid_indexs = train, valid
    if phase == 'Train':
        x = np.array(audio_paths)[train_indexs]
        y = np.array(labels)[train_indexs]
    else:
        x = np.array(audio_paths)[valid_indexs]
        y = np.array(labels)[valid_indexs]

    return x, y


class AudioDataset(Dataset):
    def __init__(self, root_path='/data1/huyj/CSIG_Audio/FMFCC_Audio_train/', data_type='Train',
                 is_one_hot=False, num_classes=2, seed=2021, k=-1):
        assert data_type in ['Train', 'Val']
        self.root_path = os.path.join(root_path, 'Train')

        self.data_type = data_type
        self.is_one_hot = is_one_hot

        self.num_classes = num_classes  # 1标识真；0标识假， 注意！！！！！！！！！！！！！
        if data_type != 'Test':
            json_file = os.path.join(root_path, 'train_label.json')
            with open(json_file, 'r') as f:
                audio_dict = json.load(f)
                self.audio_paths = list(audio_dict.keys())
                self.labels = [int(value) for value in audio_dict.values()]  # 把字符串转为数字

        print('Total audio:', len(self.audio_paths), len(self.labels))
        if k == -1 or k < 0 or k > 4:  # 80% train list for training dataset while 20% for validation dataset
            self.audio_paths, self.labels = shuffle_two_array(self.audio_paths, self.labels, seed=seed)
            split_index = int(len(self.audio_paths) * 0.8)
            if data_type == 'Train':
                self.audio_paths, self.labels = self.audio_paths[:split_index], self.labels[:split_index]
            else:  # val
                self.audio_paths, self.labels = self.audio_paths[split_index:], self.labels[split_index:]
        else:
            # TODO 比赛提交的代码没有这个，导致不同划分出现问题，不同折验证结果差别很大, 因此需要shuffle
            self.audio_paths, self.labels = shuffle_two_array(self.audio_paths, self.labels, seed=seed)  # TODA 赛后加
            self.audio_paths, self.labels = split_data_by_k_fold(self.audio_paths, self.labels, phase=data_type,
                                                                 n_splits=5, k=k)
        # 对数据进行预处理
        self.process_features, self.process_labels = preprocess(self.audio_paths, self.labels, self.root_path)
        print(data_type, f'fold:{k}', len(self.audio_paths), len(self.process_labels))

    def __len__(self):
        return len(self.process_labels)

    def __getitem__(self, index):
        wave_data = self.process_features[index]
        label = self.process_labels[index]

        if self.is_one_hot:
            label = one_hot(self.num_classes, label)

        return wave_data, label


def preprocess_submit(audio_names, root_path, data_type='Test',
                      npy_root='/pubdata/chenby/py_project/CSIG_audio/dataset/audio_npy',
                      load_npy=True):
    npy_path = os.path.join(npy_root, f'{data_type}_audio_info.npy')
    if (not load_npy) or (not os.path.exists(npy_path)):
        print('Start preprocess audio......')
        process_names = []
        process_indexes = []  # 每段语音的段下标
        audio_info = []
        for i, audio_name in enumerate(tqdm(audio_names)):
            audio_path = os.path.join(root_path, audio_name)
            wave_datas = wavfile_to_examples(audio_path)
            for j in range(len(wave_datas)):
                process_names.append(audio_name)
                process_indexes.append(j)
                audio_info.append([audio_name, j])
        print('Preprocess audio finished!!!')
        if load_npy:
            np.save(npy_path, np.array(audio_info))
    else:
        audio_info = np.load(npy_path)
        process_names, process_indexes = audio_info[:, 0], audio_info[:, 1]
        process_indexes = [int(index) for index in process_indexes]  # 把字符串转为数字
        print(f'Load audio info from {npy_path}')

    return process_names, process_indexes


class AudioDatasetSubmit(Dataset):
    def __init__(self, root_path='/pubdata/chenby/dataset/CSIG/FMFCC_Audio_test/',
                 npy_root='/pubdata/chenby/py_project/CSIG_audio/dataset/audio_npy',
                 data_type='Test', load_npy=True):
        assert data_type in ['Test']
        self.root_path = os.path.join(root_path, data_type)
        self.audio_names = sorted(os.listdir(self.root_path))
        print(f'Before preprocess length: {len(self.audio_names)}-{len(set(self.audio_names))}')
        self.audio_names, self.indexes = preprocess_submit(self.audio_names, self.root_path, data_type, npy_root, load_npy)
        print(f'After preprocess length: {len(self.audio_names)}-{len(set(self.audio_names))}')

        print('Total audio:', len(self.audio_names))

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):
        audio_index = self.indexes[index]

        audio_path = os.path.join(self.root_path, self.audio_names[index])
        wave_datas = wavfile_to_examples(audio_path)  # preprocess
        wave_datas = np.float32(wave_datas)
        wave_data = wave_datas[audio_index:audio_index+1]

        return wave_data, os.path.basename(audio_path)



