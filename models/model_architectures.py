import torchvision
import torch.nn as nn
from models.modules import ( Composite, FC,
                            ResidualBlock, SqueezeBlock, XceptionBlock,
                            SeparableConvBlock, DenseBlock, TransitionLayer ,
                            get_activation_function)
import torch
from torchview import draw_graph
from pytorch_lightning import LightningModule

def plot_model( config , model):
  model_graph = draw_graph(model, input_size=(config['BATCH_SIZE'] , config['C'] , config['H'] , config['W']), graph_dir ='TB', expand_nested=True,
                            graph_name=config['model_name'],save_graph=True,filename=config['model_name'],
                            directory='pics', depth = 3)
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

class RGB_Resnet(nn.Module):
  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
  def __init__(self, config ):
    super(RGB_Resnet, self).__init__()
    self.config = config
    self.model_name = config['model_name']+'-'+str(config['resnet_variant'])
    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 20}]
    # plot_model(self.config , self.model)


  def get_model(self):
    if self.config['resnet_variant'] == 50:
      self.resnet = torchvision.models.resnet50(weights = 'DEFAULT', progress = True)
      flatten_dim = 2048
    elif self.config['resnet_variant'] == 34:
      self.resnet = torchvision.models.resnet34(weights = 'DEFAULT', progress = True)
      flatten_dim = 512
    self.base_model = nn.Sequential(*list(self.resnet.children())[:-1])
    self.head = nn.Sequential(
                nn.Flatten(1) ,
                FC(0.2 , flatten_dim, 256),
                FC(0.0 ,256, self.config['num_classes']),
                        )
    return nn.Sequential(
                self.base_model ,
                self.head ,
                        )

  def forward(self, x):
    x = self.model(x)
    return x
#___________________________________________________________________________________________________________________


class GoogleNet(nn.Module):
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    super(GoogleNet, self).__init__()
    self.config = config
    self.model_name = config['model_name']

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
class DenseNetRGB(nn.Module):
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    super(DenseNetRGB, self).__init__()
    self.config = config
    self.model_name = config['model_name']+'-'+str(config['densenet_variant'])

    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 5}]
    # plot_model(self.config , self.model)
  def get_model(self):
    if self.config['densenet_variant'] == 121:
      self.dnet = torchvision.models.densenet121( weights='DEFAULT', progress = True)    # 'DEFAULT'  : 'IMAGENET1K_V1'
      flatten_dim = 1024
    else:
      self.dnet = torchvision.models.densenet169( weights='DEFAULT', progress = True)
      flatten_dim = 1664
    self.base_model = nn.Sequential(
                                    *list(self.dnet.children())[:-1],
                                    nn.AdaptiveAvgPool2d(1),
                                    )
    self.head = nn.Sequential(
                nn.Flatten(1) ,
                FC(0.2 , flatten_dim ,256),
              )
    return nn.Sequential(
                  self.base_model ,
                  self.head,
                  FC(0, 256, self.config['num_classes'])
                        )

  def forward(self, x):
    x =  self.model(x)
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
    def __init__(self, config, plot_model_arch = True):

        super(DenseNet,self).__init__()

        self.densenet_variant = config['densenet_variant']
        self.in_channels = config['C']
        self.num_classes = config['num_classes']
        self.compression_factor = config['compression_factor']
        self.k = config['k']
        self.config = config
        self.activation_func =config['activation_func']
        self.model = self.get_model()

        self.layer_lr = [{'params' : self.model.parameters()}]
        self.model_name = config['model_name']+'-'+'-'.join([str(i) for i in self.densenet_variant])

        if plot_model_arch:
          plot_model(config={'BATCH_SIZE':32 , 'C' : self.in_channels , 'H' : 40 , 'W' : 24 , 'model_name' : f"hsi_{self.densenet_variant}" , 'dir' : 'pics'} , model=self.model)

    def get_model(self):
        # adding 3 DenseBlocks and 3 Transition Layers
        self.deep_nn = nn.ModuleList()
        dense_block_inchannels = self.in_channels

        for num in range(len(self.densenet_variant))[:-1]:

            self.deep_nn.add_module( f"DenseBlock_{num+1}" , DenseBlock( self.densenet_variant[num] , dense_block_inchannels ,  k = self.k , activation = self.activation_func)  )
            dense_block_inchannels  = int(dense_block_inchannels + self.k * self.densenet_variant[num])

            self.deep_nn.add_module( f"TransitionLayer_{num+1}" , TransitionLayer( dense_block_inchannels, self.compression_factor) )
            dense_block_inchannels = int(dense_block_inchannels * self.compression_factor)

        # adding the 4th and final DenseBlock
        self.deep_nn.add_module( f"DenseBlock_{num+2}" , DenseBlock( self.densenet_variant[-1] , dense_block_inchannels  , k = self.k , activation = self.activation_func) )
        self.dense_block_inchannels  = int(dense_block_inchannels + self.k * self.densenet_variant[-1])
        #----------------------------------------------------------------------------------------------------------------------------

        self.seq_2 = nn.Sequential(
                          *self.deep_nn ,
                           get_activation_function(self.activation_func),
                          # # Average Pool
                          nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(1) ,
                          # # fully connected layer
                          nn.Linear(self.dense_block_inchannels, self.num_classes)
                )
        return nn.Sequential(self.seq_2)

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

class DeepEnsembleModel(nn.Module):
    def __init__(self, config):
        super(DeepEnsembleModel, self).__init__()
        self.config = config
        activation_fn = get_activation_function(config['activation_func'])
        input_dim = config['base_models_ct'] * config['num_classes']
        self.model_name = config['model_name']
        dropout_rate = config['dropout_rate']



        # Define layers using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024),
            nn.BatchNorm1d(1024),                # Batch normalization for stable learning
            activation_fn,
            nn.Dropout(dropout_rate),            # Dropout for regularization

            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            activation_fn,
            nn.Dropout(dropout_rate),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            activation_fn,
            nn.Dropout(dropout_rate),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            activation_fn,
            nn.Dropout(dropout_rate),

            nn.Linear(in_features=128, out_features=config['num_classes'])
        )
        self.layer_lr = [{'params' : self.model.parameters()}]

    def forward(self, x):
        return self.model(x)


# mnet = MobileNet(config={'num_classes': 96, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 224 , 'W' : 224 , 'model_name' : 'MobileNet' , 'dir' : './'})

# gnet = GoogleNet(config={'num_classes': 96, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 247 , 'W' : 120 , 'model_name' : 'GoogleNet' , 'dir' : './'})
# dnet = DenseNetRGB(config={'num_classes': 98, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 247 , 'W' : 120 , 'model_name' : 'DenseNet' , 'dir' : './', 'densenet_variant' : 169})

# resnet = RGB_Resnet(config={'num_classes': 98, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 3 , 'H' : 247 , 'W' : 120 ,
#                             'model_name' : 'Resnet' , 'dir' : './', 'resnet_variant' : 34})


# dnet = DenseNet(densenet_variant = [12, 18, 24, 6] , in_channels=168, num_classes=98 , compression_factor=0.25, k = 32 , config={'num_classes': 98, 'lr' : 0.001 , 'BATCH_SIZE' : 32 , 'C' : 168 , 'H' : 40 , 'W' : 24 , 'model_name' : 'DenseNet' , 'dir' : './'})

# config = {
#     'num_classes': 98,
#     'lr' : 0.001,
#     'BATCH_SIZE' : 32,
#     'C' : 168,
#     'H' : 40,
#     'W' : 24,
#     'model_name' : 'HSIModel',
#     'dir' : './',
#     'in_channels' : 168 ,
#     "k" : 64 ,
#     "compression_factor" : 0.25,
#     'apply_BAM' : False,
#     'densenet_variant' : "123"
# }

# dnet =DenseNet(densenet_variant = [12, 18, 24, 6] , config = config).to('cuda')
# x = torch.rand((32, 168 ,40,24)).to('cuda')

# y = dnet.forward(x)
# print('\n\n\n\n\n', y.shape)