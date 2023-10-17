import os
import time

import torch

from tqdm import tqdm

from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from models.memorycore import MemoryCore
from utils.utils import get_config, seed_everything, calc_auroc, save_excel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]
def main(classname):

    config_path = r"config.yaml"
    config = get_config(config_path)
    config.dataset.classname = classname
    if config.project.seed:
        seed_everything(config.project.seed)

    memorycore = MemoryCore(config).to(device)

    train_dataset = FSAD_Dataset_train(
                        dataset_path=config.dataset.path,
                        class_name=config.dataset.classname,
                        resize=config.dataset.image_size,
                        shot=config.model.shot,
                        batch=config.dataset.train_batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = FSAD_Dataset_test(
                        dataset_path=config.dataset.path,
                        class_name=config.dataset.classname,
                        resize=config.dataset.image_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # init memory bank
    origin_images = train_dataset.get_shot_images()
    for image in origin_images:
        memorycore(image.to(device))

    for i in range(config.model.loop):
        for images in tqdm(train_loader):
            images = torch.cat((images), 0)
            memorycore(images.to(device))
    memorycore.train_end()
    memorycore.eval()
    print("embedding finished")

    gt_list = []
    gt_mask_list = []
    scores_list = []
    scores_map_list = []
    for image,label,mask in tqdm(test_loader):
        anomaly_maps, anomaly_score = memorycore(image.to(device))
        gt_list.append(label.cpu().numpy())
        gt_mask_list.append(mask.cpu().numpy())
        scores_list.append(anomaly_score.cpu().numpy())
        scores_map_list.append(anomaly_maps.cpu().numpy())
       # print("score = {}".format(anomaly_score))

    img_auc,pixel_auc = calc_auroc(scores_list,scores_map_list,gt_list,gt_mask_list)

    print("class {} image auroc = {}, pixel auroc = {}".format(config.dataset.classname,img_auc,pixel_auc))
    with open("result.log", "a") as f:
        f.write("{}: ".format(config.dataset.classname))
        f.write("image_auroc = {:.3f},pixel_aurco = {:.3f}".format(img_auc, pixel_auc))
        f.write("\n")
def train_test(infer_all = True):
    if infer_all:
        if (os.path.isfile("result.log")):
            # os.remove() function to remove the file
            os.remove("result.log")
        for classname in CLASS_NAMES:
            main(classname)
            for i in range(10):
                torch.cuda.empty_cache()
        save_excel()
    else:
        main("hazelnut")
if __name__ == '__main__':

    train_test(infer_all=True)
