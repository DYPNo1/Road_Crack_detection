import torch.nn as nn
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class SegmentNet(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super(SegmentNet, self).__init__()

        self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True)
                        )

        self.layer2 = nn.Sequential(
        					nn.MaxPool2d(2)
        				)       	
       	
       	self.layer3 = nn.Sequential(
       						nn.Conv2d(64, 128, 3, stride=1, padding=1),
       						nn.BatchNorm2d(128),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(128, 128, 3, stride=1, padding=1),
       						nn.BatchNorm2d(128),
       						nn.ReLU(inplace=True)
       					)

       	self.layer4 = nn.Sequential(
        					nn.MaxPool2d(2)
        				)

        self.layer5 = nn.Sequential(
       						nn.Conv2d(128, 256, 3, stride=1, padding=1),
       						nn.BatchNorm2d(256),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(256, 256, 3, stride=1, padding=1),
       						nn.BatchNorm2d(256),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(256, 256, 3, stride=1, padding=1),
       						nn.BatchNorm2d(256),
       						nn.ReLU(inplace=True)
       					) 

       	self.layer6 = nn.Sequential(
        					nn.MaxPool2d(2)
        				) 

        self.layer7 = nn.Sequential(
       						nn.Conv2d(256, 512, 3, stride=1, padding=1),
       						nn.BatchNorm2d(512),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(512, 512, 3, stride=1, padding=1),
       						nn.BatchNorm2d(512),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(512, 512, 3, stride=1, padding=1),
       						nn.BatchNorm2d(512),
       						nn.ReLU(inplace=True)
       					)

       	self.layer8 = nn.Sequential(
        					nn.MaxPool2d(2)
        				)				  

       	self.layer9 = nn.Sequential(
                            nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
                        )

        self.layer10 = nn.Sequential(
       						nn.Conv2d(256, 512, 3, stride=1, padding=1),
       						nn.BatchNorm2d(512),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(512, 512, 3, stride=1, padding=1),
       						nn.BatchNorm2d(512),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(512, 512, 3, stride=1, padding=1),
       						nn.BatchNorm2d(512),
       						nn.ReLU(inplace=True)
       					)

        self.layer11 = nn.Sequential(
                            nn.ConvTranspose2d(512,256, 2, stride=2, padding=0)
                        )

        self.layer12 = nn.Sequential(
       						nn.Conv2d(256, 256, 3, stride=1, padding=1),
       						nn.BatchNorm2d(256),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(256, 256, 3, stride=1, padding=1),
       						nn.BatchNorm2d(256),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(256, 256, 3, stride=1, padding=1),
       						nn.BatchNorm2d(256),
       						nn.ReLU(inplace=True)
       					)

        self.layer13 = nn.Sequential(
                            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
                        )

        self.layer14 = nn.Sequential(
       						nn.Conv2d(128, 128, 3, stride=1, padding=1),
       						nn.BatchNorm2d(128),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(128, 128, 3, stride=1, padding=1),
       						nn.BatchNorm2d(128),
       						nn.ReLU(inplace=True)
       					)

        self.layer15 = nn.Sequential(
                            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
                        )

        self.layer16 = nn.Sequential(
       						nn.Conv2d(64, 64, 3, stride=1, padding=1),
       						nn.BatchNorm2d(64),
       						nn.ReLU(inplace=True),
       						nn.Conv2d(64, 64, 3, stride=1, padding=1),
       						nn.BatchNorm2d(64),
       						nn.ReLU(inplace=True)
       					)

        self.layer17 = nn.Sequential(
       						nn.Conv2d(1024, 1, 3, stride=1, padding=1),
       						nn.BatchNorm2d(1),
       						nn.ConvTranspose2d(1, 1, 8, stride=8, padding=0),
       						nn.BatchNorm2d(1)
       					)

        self.layer18 = nn.Sequential(
       						nn.Conv2d(512, 1, 3, stride=1, padding=1),
       						nn.BatchNorm2d(1),
       						nn.ConvTranspose2d(1, 1, 4, stride=4, padding=0),
       						nn.BatchNorm2d(1)
       					)

        self.layer19 = nn.Sequential(
       						nn.Conv2d(256, 1, 3, stride=1, padding=1),
       						nn.BatchNorm2d(1),
       						nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0),
       						nn.BatchNorm2d(1)
       					)

        self.layer20 = nn.Sequential(
       						nn.Conv2d(128, 1, 3, stride=1, padding=1),
       						nn.BatchNorm2d(1),
       						nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0),
       						nn.BatchNorm2d(1)
       					)

        self.layer21 = nn.Sequential(
        					nn.Conv2d(4, 1, 3, stride=1, padding=1),
        					nn.BatchNorm2d(1)
        				)

        self.layer22 = nn.Sequential(
        					nn.Sigmoid()
        				)

        self.layer23 = nn.Sequential(
        					nn.Sigmoid()
        				)

        self.layer24 = nn.Sequential(
        					nn.Sigmoid()
        				)

        self.layer25 = nn.Sequential(
        					nn.Sigmoid()
        				)

        self.layer26 = nn.Sequential(
        					nn.Sigmoid()
        				)

        if init_weights == True:
            pass



    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        x12 = self.layer12(x11)
        x13 = self.layer13(x12)
        x14 = self.layer14(x13)
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)

        x17 = self.layer17(torch.cat((x7, x10), 1))
        x18 = self.layer18(torch.cat((x5, x12), 1))
        x19 = self.layer19(torch.cat((x3, x14), 1))
        x20 = self.layer20(torch.cat((x1, x16), 1))
        x21 = self.layer21(torch.cat((x17, x18, x19, x20), 1))

        x22 = self.layer22(x17)
        x23 = self.layer23(x18)
        x24 = self.layer24(x19)
        x25 = self.layer25(x20)
        x26 = self.layer26(x21)

        return {"seg": x26, "scale4":x22, "scale3":x23, "scale2":x24, "scale1":x25}


if  __name__=='__main__':
    
    snet = SegmentNet()
    img  =  torch.randn(4, 3, 480, 320)

    snet.eval()

    snet = snet.cuda()
    img = img.cuda()

    ret = snet(img)
    s = ret["seg"]
    
    pass



