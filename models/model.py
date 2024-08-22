import torchvision
import torch.nn as nn
from modules import *
import torch
from torchview import draw_graph
from pytorch_lightning import LightningModule


def plot_model( config , model):
  model_graph = draw_graph(model, input_size=(config['BATCH_SIZE'] , config['C'] , config['H'] , config['W']), graph_dir ='TB', expand_nested=True,
                            graph_name=config['model_name'],save_graph=True,filename=config['model_name'],
                            directory=config['dir'], depth = 7)
  model_graph.visual_graph

#___________________________________________________________________________________________________________________
class MobileNet():
  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
  def __init__(self, config ):
    self.config = config
    self.model = self.get_model()
    # print(self.model)
    plot_model(self.config , self.model)
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 20}]



  def get_model(self):
    self.mobileNet = torchvision.models.mobilenet_v3_large(weights = 'MobileNet_V3_Large_Weights.DEFAULT', progress = True)

    self.base_model = nn.Sequential(*list(self.mobileNet.children())[:-1])
    self.head = nn.Sequential(
      *list(self.mobileNet.children())[-1][:-1] ,
                FC(0.2 , 1280 ,512),
                FC(0.14 ,512, 256),
                        )
    return nn.Sequential(
                self.base_model ,
                nn.Flatten(1),
                self.head ,
                FC(0, 256, self.config['num_classes']) ,
                        )

  def forward(self, x):
    return self.model(x)

#___________________________________________________________________________________________________________________

class RGB_Resnet():
  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
  def __init__(self, config ):
    self.config = config

    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 20}]
    # plot_model(self.config , self.model)


  def get_model(self):
    if self.config['variant'] == 50:
      self.resnet = torchvision.models.resnet50(weights = 'DEFAULT', progress = True)
    else:
      self.resnet = torchvision.models.resnet34(weights = 'DEFAULT', progress = True)

    self.base_model = nn.Sequential(*list(self.resnet.children())[:-1])
    self.head = nn.Sequential(
                nn.Flatten(1) ,
                FC(0.2 , self.config['flatten_dim'], 256),
                FC(0.0 ,256, self.config['num_classes']),
                        )
    return nn.Sequential(
                self.base_model ,
                self.head ,
                        )

  def forward(self, x):
    x = self.model(x)
    # return self.head(x)
    return x
#___________________________________________________________________________________________________________________


class GoogleNet():
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    self.config = config


    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 100}]
    # plot_model(self.config , self.model)
  def get_model(self):
    self.head = nn.Sequential(
                nn.Flatten(1) ,
                FC(0.2 , 1024 ,512),

    )
    self.gnet = torchvision.models.googlenet( weights='DEFAULT', progress = True)    # 'DEFAULT'  : 'IMAGENET1K_V1'
    # for param in self.gnet.parameters():
    #   param.requires_grad = False
    self.base_model = nn.Sequential(*list(self.gnet.children())[:-2])
    return nn.Sequential(
                  self.base_model ,
                  self.head,
                  FC(0, 512, self.config['num_classes'])
                        )

  def forward(self, x):
    x =  self.model(x)
    # print(x.shape)
    return x

#___________________________________________________________________________________________________________________
class DenseNetRGB():
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    self.config = config


    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 5}]
    # plot_model(self.config , self.model)
  def get_model(self):
    self.head = nn.Sequential(
                nn.Flatten(1) ,
                FC(0.2 , 1024 ,144),

    )
    self.dnet = torchvision.models.densenet121( weights='DEFAULT', progress = True)    # 'DEFAULT'  : 'IMAGENET1K_V1'

    self.base_model = nn.Sequential(
                                    *list(self.dnet.children())[:-1],
                                    nn.AdaptiveAvgPool2d(1),
                                    )
    return nn.Sequential(
                  self.base_model ,
                  self.head,
                  FC(0, 144, self.config['num_classes'])
                        )

  def forward(self, x):
    x =  self.model(x)
    # print(x.shape)
    return x
#___________________________________________________________________________________________________________________

class HSIModel(nn.Module):
  def __init__(self , config, n_res_blocks = 12):
    super(HSIModel, self).__init__()
    self.config = config
    self.in_channels = self.config['in_channels']
    self.n_res_blocks = n_res_blocks
    self.squeeze_channels = 512
    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 1}]

    plot_model(self.config , self.model)

  def get_model(self):
    self.head = nn.Sequential(
                  nn.Flatten(1) ,
                  FC(0.2 , 512, 256),
                  FC(0.15, 256, 128),
    )
    li = [ResidualBlock(256) for i in range(self.n_res_blocks)]
    self.base_model = nn.Sequential(
        # BandAttentionBlock(self.in_channels),
        nn.BatchNorm2d(self.in_channels),
        SqueezeBlock(self.in_channels, 100),
        SqueezeBlock(100, 512),
        nn.Dropout(p=0.2) ,
        XceptionBlock(512, 256),
        XceptionBlock(256, 256),
        nn.Dropout(p=0.2) ,
        # XceptionBlock(256, 128),
        nn.Sequential(*li),
        # XceptionBlock(128, 256),
        # SeparableConvBlock(128, 256),

        SeparableConvBlock(256,512),
        nn.MaxPool2d(kernel_size = (3,3) ,stride = (2,2)) ,
        nn.Dropout(p=0.15) ,
        nn.AdaptiveAvgPool2d((1,1)) ,
    )
    return nn.Sequential(
                  self.base_model,
                  self.head,
                  FC(0, 128, self.config['num_classes']),
                        )

  def forward(self, x):

    return self.model(x)
#___________________________________________________________________________________________________________________

class DenseNet(nn.Module):
    def __init__(self,densenet_variant,in_channels,num_classes, compression_factor, k , config):

        super(DenseNet,self).__init__()

        self.densenet_variant = densenet_variant
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.compression_factor = compression_factor
        self.k = k
        self.model = self.get_model()
        self.config = config
        self.layer_lr = [{'params' : self.model.parameters() , 'lr' : self.config['lr'] * 1}]
        # plot_model(self.config , self.model)



    def get_model(self):
        seq_1 =  nn.Sequential(
                      nn.Conv2d(in_channels=self.in_channels ,out_channels=64 ,kernel_size=7 ,stride=2 ,padding=3 ,bias = False) ,
                      nn.BatchNorm2d(num_features=64) ,
                      nn.ReLU() ,
                      nn.MaxPool2d(kernel_size=2, stride=2) ,
                    )
        #----------------------------------------------------------------------------------------------------------------------------
        # adding 3 DenseBlocks and 3 Transition Layers
        self.deep_nn = nn.ModuleList()
        dense_block_inchannels = self.in_channels

        for num in range(len(self.densenet_variant))[:-1]:

            self.deep_nn.add_module( f"DenseBlock_{num+1}" , DenseBlock( self.densenet_variant[num] , dense_block_inchannels ,  k = self.k)  )
            dense_block_inchannels  = int(dense_block_inchannels + self.k * self.densenet_variant[num])

            self.deep_nn.add_module( f"TransitionLayer_{num+1}" , TransitionLayer( dense_block_inchannels, self.compression_factor ) )
            dense_block_inchannels = int(dense_block_inchannels * self.compression_factor)

        # adding the 4th and final DenseBlock
        self.deep_nn.add_module( f"DenseBlock_{num+2}" , DenseBlock( self.densenet_variant[-1] , dense_block_inchannels  , k = self.k) )
        self.dense_block_inchannels  = int(dense_block_inchannels + self.k * self.densenet_variant[-1])
        #----------------------------------------------------------------------------------------------------------------------------

        seq_2 = nn.Sequential(
                          *self.deep_nn ,
                          # nn.BatchNorm2/d(num_features=self.dense_block_inchannels)  ,
                          nn.ReLU() ,
                          # # Average Pool
                          nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(1) ,
                          # # fully connected layer
                          nn.Linear(self.dense_block_inchannels, self.num_classes)
                )

        return nn.Sequential(
                  # seq_1,
                  seq_2
                        )

    def forward(self,x):
        """
        deep_nn is the module_list container which has all the dense blocks and transition blocks
        """

        return self.model(x)



class Varietal4_Classification(nn.Module):
    def __init__(self, checkpoint_path, num_classes=4):
        super(Varietal4_Classification, self).__init__()

        # Load the Lightning checkpoint
        lightning_module = LightningModule.load_from_checkpoint(checkpoint_path)

        # Extract the model from the LightningModule
        self.model = lightning_module.model

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last linear layer in nn.Sequential
        if isinstance(self.model, nn.Sequential):
            # Find the index of the last Linear layer
            layers = list(self.model.children())
            last_layer = layers[-1]

            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                # Replace it with a new Linear layer
                layers[-1] = nn.Linear(in_features, num_classes)

            # Rebuild the sequential model
            self.model = nn.Sequential(*layers)
        else:
            raise ValueError("Model must be an instance of nn.Sequential")

    def forward(self, x):
        return self.model(x)

# mnet = MobileNet(config={'num_classes': 96, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 224 , 'W' : 224 , 'model_name' : 'MobileNet' , 'dir' : './'})

# gnet = GoogleNet(config={'num_classes': 96, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 247 , 'W' : 120 , 'model_name' : 'GoogleNet' , 'dir' : './'})
# dnet = DenseNetRGB(config={'num_classes': 98, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 247 , 'W' : 120 , 'model_name' : 'DenseNet' , 'dir' : './'})

# resnet = RGB_Resnet(config={'num_classes': 98, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 247 , 'W' : 120 ,
#                             'model_name' : 'Resnet' , 'dir' : './', 'variant' : 101})


# # dnet = DenseNet(densenet_variant = [12, 18, 24, 6] , in_channels=168, num_classes=98 , compression_factor=0.3 , k = 32 , config={'num_classes': 98, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 168 , 'H' : 40 , 'W' : 24 , 'model_name' : 'DenseNet' , 'dir' : './'})

# x = torch.rand((32, 3 ,40,70))

# y = resnet.forward(x)
# print('\n\n\n\n\n', y.shape)