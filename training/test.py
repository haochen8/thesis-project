"""
eval pretained model.
"""
import csv
import json
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
import re
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from pathlib import Path
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
parser.add_argument('--export_artifacts_dir', type=str, default=None,
                    help='Optional directory for per-dataset prediction CSVs and metric JSONs.')
parser.add_argument('--dataset_json_folder', type=str, default=None,
                    help='Optional override for config["dataset_json_folder"].')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def serialize_image_ref(image_ref):
    if isinstance(image_ref, (list, tuple)):
        return json.dumps(image_ref, default=str)
    return str(image_ref)


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    label_lists = []
    sample_records = []
    dataset_images = data_loader.dataset.data_dict['image']
    sample_index = 0
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        batch_labels = data_dict['label'].cpu().detach().numpy().reshape(-1)
        batch_probs = predictions['prob'].cpu().detach().numpy().reshape(-1)
        batch_preds = (batch_probs > 0.5).astype(int)
        batch_images = dataset_images[sample_index:sample_index + len(batch_labels)]

        label_lists += list(batch_labels)
        prediction_lists += list(batch_probs)
        for local_index, (image_ref, prob, label_value, pred_value) in enumerate(zip(batch_images, batch_probs, batch_labels, batch_preds)):
            sample_records.append({
                'sample_index': sample_index + local_index,
                'image': serialize_image_ref(image_ref),
                'label': int(label_value),
                'prob': float(prob),
                'pred': int(pred_value),
            })
        sample_index += len(batch_labels)
    
    return np.array(prediction_lists), np.array(label_lists), sample_records


def write_dataset_artifacts(export_root, dataset_name, metric_one_dataset, sample_records):
    export_root = Path(export_root)
    dataset_dir = export_root / sanitize_name(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = dataset_dir / "predictions.csv"
    with predictions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "image", "label", "prob", "pred"])
        writer.writeheader()
        for row in sample_records:
            writer.writerow(row)

    metrics_path = dataset_dir / "metrics.json"
    metrics_payload = {
        "dataset": dataset_name,
        "num_samples": len(sample_records),
        "metrics": {
            k: to_jsonable(v)
            for k, v in metric_one_dataset.items()
            if k not in ("pred", "label")
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    return predictions_path, metrics_path
    
def test_epoch(model, test_data_loaders, export_artifacts_dir=None):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps, sample_records = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset

        if export_artifacts_dir:
            predictions_path, metrics_path = write_dataset_artifacts(
                export_artifacts_dir,
                key,
                metric_one_dataset,
                sample_records,
            )
            tqdm.write(f"predictions_export: {predictions_path}")
            tqdm.write(f"metrics_export: {metrics_path}")
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            if k in ['pred', 'label']:
                continue
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    print(f"===> Using device: {device}")
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    if args.dataset_json_folder:
        dataset_json_folder = os.path.abspath(os.path.expanduser(args.dataset_json_folder))
        config['dataset_json_folder'] = dataset_json_folder
        config2['dataset_json_folder'] = dataset_json_folder
    export_artifacts_dir = None
    if args.export_artifacts_dir:
        export_artifacts_dir = Path(os.path.abspath(os.path.expanduser(args.export_artifacts_dir)))
        export_artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders, export_artifacts_dir=export_artifacts_dir)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
