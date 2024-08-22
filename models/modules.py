import torch.nn as nn
import torch 

class FC(nn.Module):
    def __init__(self, drop ,in_size, out_size):
      super(FC ,self).__init__()
      self.drop , self.in_size , self.out_size = drop , in_size , out_size
      self.model = self.get_model()
    def get_model(self):
      return nn.Sequential(
          nn.Linear(self.in_size, self.out_size) , 
          nn.PReLU(),
          nn.Dropout(self.drop),
      )
    def forward(self, x):
      return self.model(x)
  
#___________________________________________________________________________________________________________________

class Composite(nn.Module):
  def __init__(self , in_channels, out_channels, kernel_size=(3,3),padding = 'same',  stride = 1):
    super(Composite, self).__init__()
    self.in_channels , self.out_channels = in_channels , out_channels
    self.kernel_size , self.padding , self.stride = kernel_size , padding , stride
    self.model = self.get_model()
    
  def get_model(self):
    return nn.Sequential(
        nn.BatchNorm2d(self.in_channels ),  
        nn.PReLU(),
        nn.Conv2d(in_channels = self.in_channels  , out_channels = self.out_channels , 
                  kernel_size = self.kernel_size ,padding = self.padding)
    )
  def forward(self, x):
    return self.model(x)
#___________________________________________________________________________________________________________________

class BandAttentionBlock(nn.Module):
  def __init__(self, in_channels, r=2):
      super(BandAttentionBlock ,self).__init__()
      self.conv2d_a = Composite(in_channels = in_channels , out_channels = 16)
      self.conv2d_b = Composite(in_channels = 16 , out_channels = 32)
      self.conv2d_c = Composite(in_channels = 32 , out_channels = 32)
      self.conv2d_d = Composite(in_channels = 32 , out_channels = 32)
      self.conv2d_e = Composite(in_channels = 32 , out_channels = 32)
      self.max_pool = nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2))
      self.gap = nn.AdaptiveAvgPool2d((1,1))        
      self.conv1d_a = nn.Conv1d(in_channels = 1 , out_channels = in_channels//r , kernel_size = 32)
      self.conv1d_b = nn.Conv1d(in_channels = 1 , out_channels = in_channels , kernel_size = in_channels//r )
      self.sigmoid = nn.Sigmoid()

      self.att_model = self.get_model()

  def get_model(self):
    return nn.Sequential(
        self.conv2d_a ,
        self.max_pool, 
        self.conv2d_b ,
        self.conv2d_c ,
        self.max_pool, 
        self.conv2d_d ,
        self.conv2d_e ,
        self.gap ,
    )
  def forward(self, x):
    vector = self.att_model(x)   # (B,32,1,1)
    vector = vector.squeeze(3)  # (B,32,1)
    vector = vector.permute(0,2,1)  # (B,1,32)
    vector = self.conv1d_a(vector)  # (B, 84 , 1),  when --> r=2
    vector = vector.permute(0,2,1)  # (B,1,84)
    vector = self.conv1d_b(vector)  # (B, 168 , 1)
    vector = vector.squeeze(2)  # (B,168)
    channel_weights = self.sigmoid(vector)  # (B,168)    
    # Multiply the image and vector along the channel dimension.
    # vector: (C,)          x: (B, C, H, W)
    output = x * channel_weights.unsqueeze(2).unsqueeze(3)  # (B, C, H, W)

    return output
#___________________________________________________________________________________________________________________

class SeparableConvBlock(nn.Module):
  def __init__(self, in_channel, out_channels, kernel_size = (3,3)):
    super(SeparableConvBlock, self).__init__()
    self.in_channels , self.out_channels = in_channel , out_channels
    self.kernel_size = kernel_size
    self.seperable_conv = self.get_model()

  def get_model(self):
    return nn.Sequential(
        nn.Conv2d(in_channels = self.in_channels , out_channels = self.out_channels , kernel_size = (1,1)), 
        nn.Conv2d(in_channels = self.out_channels , out_channels = self.out_channels , kernel_size = self.kernel_size , 
                  padding = 'same' , groups = self.out_channels), 
        nn.PReLU(), 
        nn.BatchNorm2d(self.out_channels)
        
    )
  def forward(self, x):
    # print('inside forward of seperable conv block ')
    return self.seperable_conv(x)
#___________________________________________________________________________________________________________________

class SqueezeBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(SqueezeBlock, self).__init__()
    self.in_channels, self.out_channels =  in_channels, out_channels
    self.model = self.get_model()

  def get_model(self):
    return nn.Sequential(
        nn.BatchNorm2d(self.in_channels), 
        nn.Conv2d(in_channels = self.in_channels , out_channels = self.out_channels , kernel_size = (1,1))
    )
  def forward(self, x):
   return self.model(x)
#___________________________________________________________________________________________________________________

class XceptionBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(XceptionBlock, self).__init__()
    self.in_channels , self.out_channels = in_channels , out_channels
    self.xception_model = self.get_model()
    self.conv1_1 = nn.Conv2d(in_channels = self.in_channels , out_channels = self.out_channels , kernel_size = (1,1) , stride = (2,2))

  def get_model(self):
    return nn.Sequential(
        SeparableConvBlock(in_channel = self.in_channels , out_channels = self.out_channels) ,
        nn.PReLU() ,
        SeparableConvBlock(in_channel = self.out_channels , out_channels = self.out_channels) ,
        nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2)) ,
    )
  def forward(self, x):
    a = self.conv1_1(x)
    b = self.xception_model(x)
    return a+b   # side branch + main branch
  
#___________________________________________________________________________________________________________________

class ResidualBlock(nn.Module):
  def __init__(self, in_channels , n = 3):
    super(ResidualBlock, self).__init__()
    # n : no. of seperable conv blocks in a residual block
    
    self.n = n 
    self.in_channels  = in_channels
    self.model = self.get_model()

  def get_model(self):

    self.sep_conv_blocks = [SeparableConvBlock(in_channel = self.in_channels , out_channels = self.in_channels) for i in range(self.n)]
    return  nn.Sequential(*self.sep_conv_blocks)

  def forward(self, x):
    return x + self.model(x)    # side branch + main branch
  
#__________________________________________ DENSE_NET _________________________________________________________________________

class BottleNeck(nn.Module):
  def __init__(self, in_channels,  k = 32 ,compression_factor = 0.5):
    super(BottleNeck, self).__init__()
    self.in_channels = in_channels
    self.k = k
    self.model = self.get_model()
  
  def get_model(self):
    return nn.Sequential(
        Composite(in_channels = self.in_channels ,kernel_size = (1,1) , out_channels = 4*self.k, padding=0) ,
        Composite(in_channels = 4*self.k ,kernel_size = (3,3) , out_channels = self.k ,padding = 1) ,
    )
  
  def forward(self, x):
    # print('inside bottleneck layer' , 'in : ',  x.shape)
    x = torch.cat([x, self.model(x)] , 1)
    # print('outsie BottleNeck : ' , x.shape)
    return x
  
#___________________________________________________________________________________________________________________
class DenseBlock(nn.Module):
    def __init__(self , layer_num , in_channels, k):
        """
        Looping through total number of layers in the denseblock. 
        Adding k number of channels in each loop as each layer generates tensor with k channels.
        
        Args:
            layer_num (int) : total number of dense layers in the dense block
            in_channels (int) : input number of channels 
        """

        super(DenseBlock,self).__init__()
        self.layer_num = layer_num
        self.k = k
        self.in_channels = in_channels
        self.model = self.get_model()

    def get_model(self):
        
        self.deep_nn = nn.ModuleList()
        for num in range(self.layer_num):
            self.deep_nn.add_module(f"DenseLayer_{num}",BottleNeck(self.in_channels + self.k*num))
        return nn.Sequential(*self.deep_nn)


    def forward(self,x):
      # print('inside dense block' , '     in : ',  x.shape , self.in_channels , 'layer_num --->'  , self.layer_num)
      y =  self.model(x)
      # print('outside DenseBlock : ' ,y.shape , '<--------')
      return y
#___________________________________________________________________________________________________________________

class TransitionLayer(nn.Module):
    def __init__(self,in_channels,compression_factor):
        """
        - 1x1 conv used to change output channels using the compression_factor (default = 0.5).
        - avgpool used to downsample the feature map resolution 
        
        Args:
            compression_factor (float) : output_channels/input_channels
            in_channels (int) : input number of channels 
        """

        super(TransitionLayer,self).__init__()
        self.in_channels = in_channels
        self.compression_factor = compression_factor
        self.model = self.get_model()

    def get_model(self):
        return nn.Sequential(nn.BatchNorm2d(self.in_channels) ,
                            nn.Conv2d(in_channels = self.in_channels , out_channels = int(self.in_channels * self.compression_factor) ,
                                      kernel_size = 1 ,stride = 1 ,padding = 0, bias=False ) , 
                            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )
        

    def forward(self,x):
      # print('inside transition layer' , '    in : ',  x.shape)
      # print('outside Transition : ' ,x.shape)
        return self.model(x)





  