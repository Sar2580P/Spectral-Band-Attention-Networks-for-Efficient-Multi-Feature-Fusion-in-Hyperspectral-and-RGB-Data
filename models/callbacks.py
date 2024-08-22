from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from torchvision import transforms

from  pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.0001,
   patience=10,
   verbose=True,
   mode='min'
)

theme = RichProgressBarTheme(metrics='green', time='yellow', progress_bar_finished='#8c53e0' ,progress_bar='#c99e38')
rich_progress_bar = RichProgressBar(theme=theme)

rich_model_summary = RichModelSummary(max_depth=5)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=2,
    verbose=True,
    mode='min',
 )
rgb_transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.25)),
      transforms.ToTensor(),
   ])

val_rgb_transforms = transforms.Compose([
      transforms.ToTensor(),
   ])

hsi_img_transforms = transforms.Compose([
   transforms.ToTensor(),
   transforms.RandomHorizontalFlip(p=0.7),
   transforms.RandomVerticalFlip(p=0.7),
   transforms.RandomAffine(degrees=5, translate=(0.1, 0.2), scale=(0.8, 1.25)),

])
val_hsi_transforms = transforms.Compose([
   transforms.ToTensor(),
])
