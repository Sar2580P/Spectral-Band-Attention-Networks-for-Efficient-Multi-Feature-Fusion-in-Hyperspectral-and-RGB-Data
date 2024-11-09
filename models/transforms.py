from torchvision import transforms



hsi_img_transforms = transforms.Compose([
   transforms.ToTensor(),
   transforms.RandomHorizontalFlip(p=0.7),
   transforms.RandomVerticalFlip(p=0.7),
   transforms.RandomAffine(degrees=10, translate=(0.1, 0.2), scale=(0.8, 1.25)),

])
val_hsi_transforms = transforms.Compose([
   transforms.ToTensor(),
])

rgb_transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.25)),
      transforms.ToTensor(),
   ])

val_rgb_transforms = transforms.Compose([
      transforms.ToTensor(),
   ])