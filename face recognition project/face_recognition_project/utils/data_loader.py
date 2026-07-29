import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMG_SIZE = 224


def get_transforms(img_size=IMG_SIZE, grayscale=False):
    if grayscale:
        train = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        test = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        train = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return train, test


TRAIN_TRANSFORMS, TEST_TRANSFORMS = get_transforms()


class LFWDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, grayscale=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        mode = "L" if self.grayscale else "RGB"
        img = Image.open(self.image_paths[idx]).convert(mode)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


def load_lfw_dataset(root_dir="dataset/lfw-deepfunneled",
                     min_faces=25,
                     max_classes=None,
                     test_size=0.2,
                     batch_size=32,
                     img_size=IMG_SIZE,
                     grayscale=False,
                     num_workers=0):

    train_tf, test_tf = get_transforms(img_size, grayscale=grayscale)

    person_images = []

    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        images = sorted(
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        if len(images) >= min_faces:
            person_images.append((person, images))

    person_images.sort(key=lambda item: len(item[1]), reverse=True)
    if max_classes is not None:
        person_images = person_images[:max_classes]

    class_names = [person for person, _ in person_images]

    train_imgs, train_lbls = [], []
    val_imgs, val_lbls = [], []

    for label, (_, images) in enumerate(person_images):
        if len(images) == 1:
            train_imgs.extend(images)
            train_lbls.extend([label])
            continue

        tr_imgs, va_imgs = train_test_split(
            images,
            test_size=test_size,
            random_state=42,
        )
        train_imgs.extend(tr_imgs)
        train_lbls.extend([label] * len(tr_imgs))
        val_imgs.extend(va_imgs)
        val_lbls.extend([label] * len(va_imgs))

    print(f"Classes: {len(class_names)}")
    print(f"Images : {len(train_imgs) + len(val_imgs)}")
    print(f"Train  : {len(train_imgs)}  |  Val: {len(val_imgs)}")

    train_dataset = LFWDataset(train_imgs, train_lbls, train_tf, grayscale=grayscale)
    val_dataset = LFWDataset(val_imgs, val_lbls, test_tf, grayscale=grayscale)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_names
