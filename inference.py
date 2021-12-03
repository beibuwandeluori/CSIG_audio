import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from dataset.CSIG_dataset import AudioDataset, AudioDatasetSubmit
from network.model import AudioModel, AudioModelV2, AudioModelV3


def ensemble_json(json_paths, save_json_root):
    dicts = []
    for json_path in json_paths:
        with open(json_path, 'r') as load_f:
            print(f'Load in {json_path}!')
            dict_i = json.load(load_f)
            dicts.append(dict_i)
    results = {}
    len_dicts = len(dicts)
    for key, value in dicts[0].items():
        results[key] = value
        for dicts_i in dicts[1:]:
            results[key] += dicts_i[key]
        results[key] /= len_dicts

    json_path = f'{save_json_root}/results.json'
    with open(json_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def eval_model(eval_loader, weights=None):
    eval_process = tqdm(eval_loader)
    outputs = {}
    with torch.no_grad():
        for img, audio_names in eval_process:
            img = Variable(img.cuda(device_id))
            if not is_ensemble:
                y_pred = audio_model(img)
                y_pred = nn.Softmax(dim=1)(y_pred)
            else:
                if weights is None:
                    weights = [1.0/len(models) for _ in range(len(models))]

                for i, model in enumerate(models):
                    if i == 0:
                        y_pred = weights[i] * nn.Softmax(dim=1)(model(img))
                    else:
                        y_pred += weights[i] * nn.Softmax(dim=1)(model(img))

            y_pred = y_pred[:, 1].data.cpu().numpy()
            for i, audio_name in enumerate(audio_names):
                if outputs.get(audio_name) is None:
                    outputs[audio_name] = [y_pred[i]]
                else:
                    outputs[audio_name].append(y_pred[i])
                if np.isnan(y_pred[i]):
                    print(audio_name, img[i], y_pred[i])

    print(len(outputs))
    results = {}
    for key, value in outputs.items():
        results[key] = sum(value) / len(value)
        # score = "%.4f" % (sum(value) / len(value))
        # results[key] = score

    json_path = f'{json_root}/results.json'
    with open(json_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def load_model(model_paths, models, device_id):
    for i in range(len(models)):
        models[i].load_state_dict(torch.load(model_paths[i], map_location='cpu'))
        print('Model {} found in {}'.format(i, model_paths[i]))
        models[i] = models[i].cuda(device_id)
        models[i].eval()

    return models


if __name__ == '__main__':
    json_ensemble = False
    if json_ensemble:
        json_root = f'./output/results/ensemble_01_t'
        if json_root and not os.path.exists(json_root):
            os.makedirs(json_root)
        ensemble_json(json_paths=[
            './output/results/efficientnet-b2_96_k2e6_hop48_t/results.json',
            './output/results/efficientnet-b2_96_k2e9_t/results.json'],
                      save_json_root=json_root)
    else:
        root_path = '/data/audioForRuningModels/'
        npy_root = './dataset/audio_npy/'

        batch_size = 128
        device_id = 0
        load_npy = False
        model_name = 'efficientnet-b2'  # efficientnet-b3 tf_efficientnet_b3_ns
        json_root = f'./output/results/{model_name}_96_k2e6_hop48_t'
        # json_root = f'./output/results/{model_name}_96_k2e9_t'
        if json_root and not os.path.exists(json_root):
            os.makedirs(json_root)

        weights = None
        is_ensemble = False
        if not is_ensemble:
            # model_path = None
            model_path = './output/weights/efficientnet-b2_96_hop48_k2/audio6_acc0.9832.pth'
            # model_path = './output/weights/efficientnet-b2_96_k2/audio9_acc0.9752.pth'
            # Load model
            audio_model = AudioModelV3(model_name=model_name)
            audio_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print('Model found in {}'.format(model_path))
            audio_model = audio_model.cuda(device_id)
            audio_model.eval()
        else:
            # weights = [0.2, 0.3, 0.5]
            models = load_model(model_paths=['/pubdata/chenby/py_project/CSIG_audio/output/weights/efficientnet-b2_96_k2/audio9_acc0.9752.pth',
                                             '/pubdata/chenby/py_project/CSIG_audio/output/weights/efficientnet-b4_96_k2/audio9_acc0.9817.pth',
                                             ],
                                models=[AudioModelV3(model_name='efficientnet-b2'),
                                        AudioModelV3(model_name='efficientnet-b4'),
                                        ],
                                device_id=device_id)

        start = time.time()
        xdl_test = AudioDatasetSubmit(root_path=root_path, npy_root=npy_root, load_npy=load_npy)
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        eval_model(test_loader, weights)
        print('Total time:', time.time() - start)








