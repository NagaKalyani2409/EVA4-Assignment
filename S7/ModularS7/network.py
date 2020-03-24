from imports import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Convolution Block #1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2, dilation=2, bias=False)      
        #input :32x32x3    k=3x3  kernel_size=32(3x3x3x32)  output=32x32x32      RF:5    
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2, dropout=0.1,  bias=False)     
        #input:32x32x32   k=3x3 , kernel_size=64(3x3x32x64)     output=32x32x64      RF:9 

        #MaxPool #1
        self.pool = nn.MaxPool2d(2, 2)                                                                                  
        #input:32x32x64   k=2x2 kernel_size=2 output=16x16x64      RF:10

        #ConvolutionBlock #2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64*3, kernel_size=3, padding=1, bias=False, groups=64)     
      
        #input:16x16x64   k=3x3  kernel_size=32((3x3x1)64x3)  output=16x16x(64*3)     RF:14       
       
        self.conv4 = nn.Conv2d(in_channels=64*3, out_channels=128, kernel_size=1, bias=False)   
        #input:16x16x(64*3)   k=1x1 kernel_size=128(1x1x32x128)  output=16x16x128     RF:14

        #maxpool #2
        self.pool = nn.MaxPool2d(2, 2) 
        #input:16x16x128   k=2x2 , output=8x8x128      RF:16

        #convolution Block#3
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, dropout=0.1, bias=False)                 
        #input:8x8x128   k=1x1 ,kernel_size=32(3x3x1x32)   output=8x8x32     RF:16

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, dropout=0.1, padding=1, bias=False)                  
        #input:8x8x32    k=3x3 kernel_size=64(3x3x32x64)  output=8x8x64     RF:24

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dropout=0.2, padding=1, bias=False)                
         #input:8x8x64   k=3x3 ,kerenel_size=128(3x3x64x128) output=8x8x128    RF:32

        #maxpool #2
        self.pool = nn.MaxPool2d(2, 2)                                                                                 
         #input:8x8x128   k=2x2 ,kerenel_size=2   output=4x4x128    RF:36

        
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False)                 
         #input:4x4x128  k=3x3 kernel_size=64    output:4x4x64    RF:42
        self.gap = nn.AvgPool2d(kernel_size=4)            
         #input:4x4x64   k=4 , kernel_size=4,      output:1x1x64    RF:46
        self.final = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, bias=False) 
         #input:1x1x64   k=1, kernel_size=10    o:1x1x10    RF:60

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.pool(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.pool(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        x=self.pool(x)
        x=self.conv8(x)
        x=self.gap(x)
        x=self.final(x)
        x = x.view(-1, 10)                           
        return F.log_softmax(x)


        