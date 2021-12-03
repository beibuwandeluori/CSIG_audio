import os
import numpy as np
import json

import torch
from torch.autograd import Variable
import torch.nn as nn

from dataset.preprocess.vggish_input import wavfile_to_examples
from network.model import AudioModel, AudioModelV2, AudioModelV3
from tqdm import tqdm


def clip_pred(val, threshold=0.01):
    if val < threshold:
        val = threshold
    elif val > (1 - threshold):
        val = 1 - threshold

    return val


def predict_audio(file, model_):
    wave_data = wavfile_to_examples(file)
    wave_data = np.float32(wave_data)
    wave_data = np.expand_dims(wave_data, axis=1)
    # wave_data = wavfile_to_examples(file)
    # wave_data = np.float32(wave_data)
    # b, h, w = wave_data.shape
    # add_axis = np.zeros((16 - b, h, w), dtype=np.float32)
    # wave_data = np.concatenate((wave_data, add_axis), axis=0)
    # wave_data = np.expand_dims(wave_data, axis=0)
    wave_data = torch.from_numpy(wave_data)
    wave_data = wave_data.type(torch.FloatTensor)
    prediction, output = predict_with_audio_model(wave_data, model_)
    pred = output[:, 1]
    return prediction, pred


def predict_with_audio_model(audio, model, post_function=nn.Softmax(dim=1)):
    """
    Predicts the label of an input audio.

    :param audio: numpy audio
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Model prediction
    model.eval()
    with torch.no_grad():
        audio = Variable(audio.cuda())
        output = model(audio)
        output = post_function(output)
        output = torch.mean(input=output, dim=0, keepdim=True)

    # Cast to desired
    _, prediction = torch.max(output, 1)  # argmax
    prediction = prediction.cpu().numpy()

    return prediction, output.cpu().numpy()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    root_test = '/pubdata/chenby/dataset/CSIG/FMFCC_Audio_test/Test'

    model_name = 'resnet34'  # NextVLAD efficientnet-b0
    # save_root = f'./output/results/vggisgh_{model_name}_LS'
    save_root = f'./output/results/{model_name}_e2'
    if save_root and not os.path.exists(save_root):
        os.makedirs(save_root)
    # 加载模型
    model_path = "/pubdata/chenby/py_project/CSIG_audio/output/weights/resnet34/audio2_acc0.9982.pth"
    # audio_model = AudioModel(model_name=model_name)
    audio_model = AudioModelV3(model_name=model_name)
    if model_path is not None:
        # model = torch.load(model_path)
        audio_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f'Load model in {model_path}')
    audio_model = audio_model.cuda()

    new_path = []
    audio_names = os.listdir(root_test)
    for audio_name in audio_names:
        new_path.append(os.path.join(root_test, audio_name))
    new_path = sorted(new_path)
    preds = np.array([])
    output = np.array([])
    for i, path in enumerate(tqdm(new_path)):
        prediction, pred = predict_audio(path, audio_model)
        # print(pred)
        preds = np.concatenate([preds, prediction])
        output = np.concatenate([output, pred])
    result = {}

    for i, (key, value) in enumerate(zip(new_path, output)):
        audio_name = os.path.split(key)[-1]
        result[audio_name] = value
    # save predicted value in json file
    save_json_path = os.path.join(save_root, 'results.json')
    with open(save_json_path, 'w') as file_obj:
        json.dump(result, file_obj)
