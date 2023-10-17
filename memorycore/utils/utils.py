
import random

import numpy as np
import pandas as pd
import torch
import yaml
from easydict import EasyDict
from sklearn.metrics import roc_auc_score

def get_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    return config

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calc_auroc(scores_list,scores_map_list,gt_list,gt_mask_list):
    scores = np.asarray(scores_list)
    # Normalization
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # calculate image-level ROC AUC score
    #scores = scores.reshape(scores.shape[0], -1).max(axis=1)

    gt_list = np.asarray(gt_list)
    img_roc_auc = roc_auc_score(gt_list, scores)

    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list)
    scores_map = np.asarray(scores_map_list)
    gt_mask = (gt_mask > 0.5).astype(np.int_)
    pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores_map.flatten())

    return img_roc_auc*100,pixel_rocauc*100
def save_excel():

    ans = [[], [], [], []]

    with open("result.log", "r") as lines:
        for line in lines:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
            #  print(data_line)
            category = data_line[0].split(":")[0]
            ans[0].append(category)
            image_auroc = format(float(data_line[3].split(",")[0]),".1f")
            ans[1].append(float(image_auroc))
            pixel_auroc = format(float(data_line[5].split(",")[0]),".1f")
            ans[2].append(float(pixel_auroc))
    ans[0].append("avg")
    ans[1].append(float(format(sum(ans[1]) / len(ans[1]),".1f")))
    ans[2].append(float(format(sum(ans[2]) / len(ans[2]),".1f")))
    df = pd.DataFrame({"category": ans[0],
                       "image_auroc": ans[1],
                       "pixel_aurco": ans[2]
                       })

    df = df.set_index('category')
    print(df)
    df.to_excel('result.xlsx')
    print("image avg: {}, pixel avg: {}".format(ans[1][-1],ans[2][-1]))