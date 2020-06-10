import torch
import torch.nn as nn
import numpy as np
import math
import time


def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)








class Bottleneck(nn.Module):
    

    def __init__(self, inplanes, planes, hyper, stride=1):
        super(Bottleneck, self).__init__()

        self.parallel = hyper['num_parallel']
        self.no_uniform = hyper['no_uniform']
        self.alpha_scale = hyper['alpha_scale']

        self.dim3x3 = planes
        expansion = 4

        
        
        self.out_planes = planes * expansion

        self.downsample0 = None
        if(stride!=1 or inplanes != planes * expansion):
            self.downsample0 = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion)
            )
        
        self.conv1 = nn.Conv2d(inplanes, self.dim3x3*self.parallel, groups=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim3x3*self.parallel)


        
        self.conv2 = nn.Conv2d(self.dim3x3*self.parallel, self.dim3x3*self.parallel, groups=self.parallel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.dim3x3*self.parallel)


        
        self.conv3 = nn.Conv2d(self.dim3x3*self.parallel, planes * expansion * self.parallel, groups=self.parallel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion * self.parallel)


        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

        if(self.no_uniform):
            alphas = np.array([1/np.sqrt(self.parallel+1-i) for i in range(1, self.parallel+1)])
            a_sum = np.sum(alphas)
            scale = np.sqrt(self.parallel)/a_sum
            alphas = alphas * scale
            assert np.abs(np.sum(alphas)-np.sqrt(self.parallel))<1e-3
            self.alphas = alphas.reshape(-1, 1, 1, 1, 1)



    def forward(self, x):
        identity = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)
        

        out = torch.stack(torch.split(out, self.out_planes, dim=1))
        out = torch.sum(out, dim=0)
        if(not self.no_uniform): # uniform scaling
            out *= (self.alpha_scale / np.sqrt(self.parallel))
        else: # non-uniform scaling
            out *= (self.alpha_scale * self.alphas)

        if(self.downsample0!=None):
            identity = self.downsample0(identity)        

        out += identity
        out = self.relu(out)

        return out

class presnet(nn.Module):

    def __init__(self, block, layers, hyper_params, num_classes=10):
        super(presnet, self).__init__()

        self.expansion = 4
        self.hyper_params = hyper_params
        
        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], hyper_params)
        self.layer2 = self._make_layer(block, 128, layers[1], hyper_params, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], hyper_params, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256*self.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.modules(): 
            if isinstance(m, Bottleneck): # scale up the default initialization of group convolution
                m.conv1.weight.data *= np.sqrt(m.parallel)
                m.conv2.weight.data *= np.sqrt(m.parallel)
                m.conv3.weight.data *= np.sqrt(m.parallel)

                
        
    def _make_layer(self, block, planes, blocks, hyper_params, stride=1):
        layers = []

        layers.append(block(self.inplanes, planes, hyper_params, stride=stride))


        self.inplanes = planes * self.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, hyper_params))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnext(hyper_params, num_classes):
    model = presnet(Bottleneck, [3, 3, 3], hyper_params, num_classes)
    return model   
