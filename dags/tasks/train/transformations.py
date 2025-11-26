
import albumentations as A

def build_transforms():
    return A.Compose([
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Optional: small rotation
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

