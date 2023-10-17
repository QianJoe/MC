
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Class_Augments():

    def __init__(self,resize=256):
        self.resize = resize
    def bottle_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def cable_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def capsule_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def carpet_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def grid_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5)
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def hazelnut_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def leather_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def metal_nut_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def pill_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def screw_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def tile_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def toothbrush_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5)
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def transistor_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def wood_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
    def zipper_aug(self):
        transform = A.Compose(
            [
                A.Resize(height=self.resize, width=self.resize, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.OneOf([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(p=0.5),
                ], p=1),
                ToTensorV2(),
            ]
        )
        return transform
