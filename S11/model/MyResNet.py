import torch
import torch.nn as nn
import torch.nn.functional as F

class MyResNet(nn.Module):

  def __init__(self):
    super(MyResNet,self).__init__()

    self.prepLayer=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               ) 
    self.layer1=  nn.Sequential(self.cmbr(64,128),self.resblk(128,128))
    self.layer2 = self.cmbr(128,256)
    self.layer3=  nn.Sequential(self.cmbr(256,512),self.resblk(512,512))
    self.maxpool = nn.MaxPool2d(4,4)
    
    self.fc = nn.Conv2d(in_channels=512,out_channels=10,kernel_size=(1,1),bias=False,padding=0,stride=1)

  def cmbr(self,in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),bias=False,padding=1,stride=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
  def resblk(self,in_channels,out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1,stride=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1,stride=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),)
    
  def forward(self,x):
    # PrepLayer
      x = self.prepLayer(x)
      #layer 1
      x= self.layer1(x)
      #layer 2
      x = self.layer2(x)
      #layer 3
      x = self.layer3(x)
      x = self.maxpool(x)
      x = self.fc(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)    