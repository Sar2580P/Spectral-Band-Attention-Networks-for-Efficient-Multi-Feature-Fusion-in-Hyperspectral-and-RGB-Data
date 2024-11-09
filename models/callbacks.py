from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from  pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from processing.utils import read_yaml
from pytorch_lightning.callbacks import LearningRateMonitor

config = read_yaml('models/rgb/config.yaml')

es_config = config['EarlyStopping']
early_stop_callback = EarlyStopping(
   monitor=es_config['monitor'],
   min_delta=es_config['min_delta'],
   patience=es_config['patience'],
   verbose=es_config['verbose'],
   mode=es_config['mode']
)

theme = RichProgressBarTheme(metrics='green', time='yellow', progress_bar_finished='#8c53e0' ,progress_bar='#c99e38')
rich_progress_bar = RichProgressBar(theme=theme)

rich_model_summary = RichModelSummary(max_depth=5)

ckpt_config = config['ModelCheckpoint']
checkpoint_callback = ModelCheckpoint(
    monitor=ckpt_config['monitor'],
    save_top_k=ckpt_config['save_top_k'],
    verbose=ckpt_config['verbose'],
    mode=ckpt_config['mode'],
 )


lr_monitor = LearningRateMonitor(logging_interval='step')
