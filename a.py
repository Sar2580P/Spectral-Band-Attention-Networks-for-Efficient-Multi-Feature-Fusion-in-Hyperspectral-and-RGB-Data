import torchvision
from torchview import draw_graph
from pytorch_lightning import LightningModule
import torch.nn as nn
from models.modules import FC

def plot_model( config , model):
  model_graph = draw_graph(model, input_size=(config['BATCH_SIZE'] , config['C'] , config['H'] , config['W']), graph_dir ='TB', expand_nested=True,
                            graph_name=config['model_name'],save_graph=True,filename=config['model_name'],
                            directory=config['dir'], depth = 5)
  model_graph.visual_graph

class MobileNet():
  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
  def __init__(self, config ):
    self.config = config
    self.model = self.get_model()
    # print(self.model)
    plot_model(config={'BATCH_SIZE':32 , 'C' : 3 , 'H' : 40 , 'W' : 32 , 'model_name' : f"convnext" , 'dir' : 'pics'} , model=self.model)
    # self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 20}]



  def get_model(self):
    self.mobileNet = torchvision.models.convnext_tiny(weights = 'IMAGENET1K_V1')

    # self.base_model = nn.Sequential(*list(self.mobileNet.children())[:-1])
    # self.head = nn.Sequential(
    #   *list(self.mobileNet.children())[-1][:-1] ,
    #             FC(0.2 , 1280 ,512),
    #             FC(0.14 ,512, 256),
    #                     )
    return self.mobileNet

  def forward(self, x):
    return self.model(x)


model = MobileNet(config = {})