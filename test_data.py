from dataset import AudioDataset, wavfile_to_examples, AudioDatasetSubmit
import torch
import os
import numpy as np


def read_audio(wav_path):
    import torchaudio
    waveform, sample_rate = torchaudio.load(wav_path)
    print("Shape of waveform:{}".format(waveform.size()))  # 音频大小
    print("sample rate of waveform:{}".format(sample_rate))  # 采样率
    specgram = torchaudio.transforms.MelSpectrogram()(waveform)
    print("Shape of spectrogram:{}".format(specgram.size()))


def extract_vggish_input():
    audio_root = './dataset/wav_samples'
    audio_names = os.listdir(audio_root)
    for audio_name in audio_names:
        audio_path = os.path.join(audio_root, audio_name)
        wave_data = wavfile_to_examples(audio_path)
        print(np.array(wave_data).shape)
        npy_name = audio_name.split('.wav')[0] + '.npy'
        np.save(f'./dataset/wav_samples/{npy_name}', wave_data)


def test_data_loader():
    # root_path = '/pubdata/chenby/dataset/CSIG/FMFCC_Audio_train/'
    root_path = '/pubdata/chenby/dataset/CSIG/FMFCC_Audio_test/'

    # dataset = AudioDataset(root_path=root_path, is_one_hot=True, k=0, data_type='Val')
    dataset = AudioDatasetSubmit(root_path, load_npy=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    # eval_dataset = AudioDataset(root_path=train_path, data_type='Val')
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)
    for i, (x, y) in enumerate(train_loader):
        print(x.size(), len(y), x[0, 0, 0, :10])
        if i == 10:
            break


if __name__ == "__main__":
    test_data_loader()
    # extract_vggish_input()
    # read_audio(wav_path='/pubdata/chenby/dataset/CSIG/FMFCC_Audio_train/Train/20000002.wav')
