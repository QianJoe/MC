import glob
import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from augments.class_aug import Class_Augments

CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]


class FSAD_Dataset_train(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 resize=256,
                 shot=2,
                 batch=32
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.resize = resize
        self.shot = shot
        self.batch = batch
        self.samples = self.load_dataset_folder()
        self.transform = A.Compose(
            [
                A.Resize(height=resize, width=resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    def load_dataset_folder(self):
        img_dir = os.path.join(self.dataset_path, self.class_name, "train", 'good',"*")
        train_imgs = glob.glob(img_dir)
        return train_imgs[:self.shot]

    def get_shot_images(self):
        images = []
        for path in self.samples:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(self.transform(image=image)["image"].unsqueeze(0))
        return images

    def random_transform(self,image):

        class_augments = Class_Augments(self.resize)
        transform = getattr(class_augments, self.class_name + "_aug")()
        #transform = self.get_transform()
        return transform(image=image)["image"]
    def __getitem__(self, idx):

        path = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = []
        for i in range(self.batch):
            images.append(self.random_transform(image))
       #images.append(self.transform(image=image)["image"])
        return images

    def __len__(self):
        return len(self.samples)


class FSAD_Dataset_test(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 resize=256
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.resize = resize
        self.samples = self.load_dataset_folder()
        # set transforms
        self.transform = A.Compose(
            [
                A.Resize(height=resize, width=resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def load_dataset_folder(self):
        img_dir = os.path.join(self.dataset_path, self.class_name, "test", '*', "*")
        test_imgs = glob.glob(img_dir)
        return test_imgs

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if path.split("/")[-2] == 'good':
            mask = np.zeros(shape=image.shape[:2])
            label = 0
        else:
            mask_path = path.replace('test', 'ground_truth').replace('.png','_mask.png')
            label = 1
            mask = cv2.imread(mask_path, flags=0) / 255.0
        pre_processed = self.transform(image=image,mask=mask)
        image = pre_processed["image"]
        mask = pre_processed["mask"]
        return image,label,mask

    def __len__(self):
        return len(self.samples)