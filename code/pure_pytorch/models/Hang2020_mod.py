#Hang et al. 2020 https://arxiv.org/pdf/2005.11977.pdf
from math import ceil
import numpy
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch
from torchvision import models
    
class conv_module(nn.Module):
    def __init__(self, in_channels, resnet_blocks=4, pretrained=False):
        # resnet_blocks is number of ResNet blocks to use. Minimum is 1, maximum is 4.
        super().__init__()
        self.resnet_model = models.resnet18(pretrained=pretrained)
        
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # delete last two layers: avgpool and classifier
        self.resnet_model = nn.Sequential(*list(self.resnet_model.children())[:-2])
        # limit number of ResNet blocks according to resnet_blocks parameter
        self.resnet_model = nn.Sequential(*list(self.resnet_model.children())[:4+resnet_blocks])
        
        if pretrained:
            #Download weights for resnet18
            url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            state = torch.utils.model_zoo.load_url(url)
            
            # Adapt conv1.weight layer to 150 input channels
            conv1_weight = state['conv1.weight']
            conv1_dtype = conv1_weight.dtype
            conv1_weight = conv1_weight.float()
            repeat = int(ceil(in_channels / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :] # copy first 3 channels repeat-times
            conv1_weight *= (3 / float(in_channels))
            conv1_weight = conv1_weight.to(conv1_dtype)
            self.resnet_model[0].weight.data = conv1_weight
            
            # after all these operations check if weights are correct in one layer
            assert torch.equal(self.resnet_model[1].weight, state['bn1.weight'])
            
            
    def forward(self, x, pool=False):
        x = self.resnet_model(x)
        return x
    
class Classifier(nn.Module):
    """A small module to seperate the classifier head, which depends on the number of classes.
    This makes it easier to pretain on other data
    """
    def __init__(self, in_features, classes):
        super(Classifier,self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=classes)
#         print(in_features)
        
    def forward(self, features):
#         print(features.size())
        
        scores = self.fc1(features)
        
        return scores
    
def global_spectral_pool(x):
    """Helper function to keep the same dimensions after pooling to avoid resizing each time"""
    global_pool = torch.mean(x,dim=(2,3))
    global_pool = global_pool.unsqueeze(-1)
    
    return global_pool


class spectral_attention(nn.Module):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    The feature maps should be pooled to remove spatial dimensions before reading in the module
    Args:
        in_channels: number of feature maps of the current image
    """
    def __init__(self, filters):
        super(spectral_attention, self).__init__()        
        # Weak Attention with adaptive kernel size based on size of incoming feature map
        if filters == 64:
            kernel_size = 3
        elif filters == 128:
            kernel_size = 5
        elif filters == 256 or filters == 512:
            kernel_size = 7
        else:
            raise ValueError(
                "Unknown incoming kernel size {} for attention layers".format(kernel_size))
        
        self.attention_conv1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same")
        self.attention_conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same")
        
    def forward(self, x):
        """Calculate attention and class scores for batch"""
        #Global pooling and add dimensions to keep the same shape
        pooled_features = global_spectral_pool(x)
        
        #Attention layers
        attention = self.attention_conv1(pooled_features)
        attention = F.relu(attention)
        attention = self.attention_conv2(attention)
        attention = F.sigmoid(attention)
        
        #Add dummy dimension to make the shapes the same
#         print(f'before unsqueeze attention size: {attention.size()}, x size: {x.size()} ')
        attention = attention.unsqueeze(-1)
        attention = torch.mul(x, attention)
        
        # Classification Head
        pooled_attention_features = global_spectral_pool(attention)
#         print(f'after global_spectral_pool pooled_attention_features: {pooled_attention_features.size()}')

        pooled_attention_features = torch.flatten(pooled_attention_features, start_dim=1)
#         print(f'after flattening pooled_attention_features: {pooled_attention_features.size()}')
        
        return attention, pooled_attention_features
    
    
class spectral_network(nn.Module):
    """
        Learn spectral features with alternating convolutional and attention pooling layers
    """
    def __init__(self, bands, classes, pretrained=False):
        super(spectral_network, self).__init__()
        
        #First submodel is 32 filters
        self.conv1 = conv_module(in_channels=bands, resnet_blocks=1, pretrained=pretrained)
        self.attention_1 = spectral_attention(filters=64)
        self.classifier1 = Classifier(classes=classes, in_features=64)
    
        self.conv2 = conv_module(in_channels=64, resnet_blocks=2, pretrained=pretrained)
        self.attention_2 = spectral_attention(filters=128)
        self.classifier2 = Classifier(classes=classes, in_features=128)
    
        self.conv3 = conv_module(in_channels=128, resnet_blocks=3, pretrained=pretrained)
        self.attention_3 = spectral_attention(filters=256)
        self.classifier3 = Classifier(classes=classes, in_features=256)
    
    def forward(self, x):
        """The forward method is written for training the joint scores of the three attention layers"""
        x = self.conv1(x)
#         print(x.size())
        x, attention = self.attention_1(x)
#         print(x.size(), attention.size())
        scores1 = self.classifier1(attention)
        
        x = self.conv2(x, pool = True)
#         print(x.size(), attention.size())
        x, attention = self.attention_2(x)
#         print(x.size(), attention.size())
        scores2 = self.classifier2(attention)
        
        x = self.conv3(x, pool = True)        
        x, attention = self.attention_3(x)
        scores3 = self.classifier3(attention)
        
        return [scores1,scores2,scores3]

class spatial_attention(nn.Module):
    """
    Learn cross band spatial features with a set of convolutions and spectral pooling attention layers
    """
    def __init__(self, filters):
        super(spatial_attention,self).__init__()
        self.channel_pool = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=1)
        
        # Weak Attention with adaptive kernel size based on size of incoming feature map
        if filters == 64:
            kernel_size = 7
        elif filters == 128:
            kernel_size = 5
        elif filters == 256:
            kernel_size = 3
        else:
            raise ValueError(
                "Unknown incoming kernel size {} for attention layers".format(kernel_size))
        
        self.attention_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same")
        self.attention_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same")
        
        #Add a classfication branch with max pool based on size of the layer
        if filters == 64:
            pool_size = (4, 4)
            in_features = 128
        elif filters == 128:
            in_features = 256
            pool_size = (2, 2)
        elif filters == 256:
            in_features = 512
            pool_size = (1, 1)
        else:
            raise ValueError("Unknown filter size for max pooling")
        
        self.class_pool = nn.MaxPool2d(pool_size)
        
    def forward(self, x):
        """Calculate attention and class scores for batch"""
        #Global pooling and add dimensions to keep the same shape
        pooled_features = self.channel_pool(x)
        pooled_features = F.relu(pooled_features)
        
        #Attention layers
        attention = self.attention_conv1(pooled_features)
        attention = F.relu(attention)
        attention = self.attention_conv2(attention)
        attention = F.sigmoid(attention)
        
        #Add dummy dimension to make the shapes the same
#         print(f'before unsqueeze attention size: {attention.size()}, x size: {x.size()} ')
        attention = torch.mul(x, attention)
#         print(f'after mul attention size: {attention.size()}')

        
        # Classification Head
        pooled_attention_features = self.class_pool(attention)
#         print(f'after class pool pooled_attention_features: {pooled_attention_features.size()}')

        pooled_attention_features = torch.flatten(pooled_attention_features, start_dim=1)
#         print(f'after flattening pooled_attention_features: {pooled_attention_features.size()}')


        return attention, pooled_attention_features
    
class spatial_network(nn.Module):
    """
        Learn spatial features with alternating convolutional and attention pooling layers
    """
    def __init__(self, bands, classes, pretrained=False):
        super(spatial_network, self).__init__()
        
        #First submodel is 32 filters
        self.conv1 = conv_module(in_channels=bands, resnet_blocks=1, pretrained=pretrained)
        self.attention_1 = spatial_attention(filters=64)
        self.classifier1 = Classifier(classes=classes, in_features=20736)
    
        self.conv2 = conv_module(in_channels=64, resnet_blocks=2, pretrained=pretrained)
        self.attention_2 = spatial_attention(filters=128)
        self.classifier2 = Classifier(classes=classes, in_features=3200)        
    
        self.conv3 = conv_module(in_channels=128, resnet_blocks=3, pretrained=pretrained)
        self.attention_3 = spatial_attention(filters=256)
        self.classifier3 = Classifier(classes=classes, in_features=256)        
    
    def forward(self, x):
        """The forward method is written for training the joint scores of the three attention layers"""
        x = self.conv1(x)
#         print(x.size())
        x, attention = self.attention_1(x)
#         print(x.size(), attention.size())
        scores1 = self.classifier1(attention)
        
        x = self.conv2(x, pool = True)
#         print(x.size())
        x, attention = self.attention_2(x)
#         print(x.size(), attention.size())
        scores2 = self.classifier2(attention)
        
        x = self.conv3(x, pool = True)   
#         print(x.size())
        x, attention = self.attention_3(x)
#         print(x.size(), attention.size())
        scores3 = self.classifier3(attention)
        
        return [scores1,scores2,scores3]

class Hang2020(nn.Module):
    def __init__(self, bands, classes, pretrained=False):
        super(Hang2020, self).__init__()    
        self.spectral_network = spectral_network(bands, classes, pretrained=pretrained)
        self.spatial_network = spatial_network(bands, classes, pretrained=pretrained)
        
        #Learnable weight
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)
        
    def forward(self, x):
        spectral_scores = self.spectral_network(x)
        spatial_scores = self.spatial_network(x)
        
        #Take the final attention scores
        spectral_classes = spectral_scores[-1]
        spatial_classes = spatial_scores[-1]
#         print(spectral_classes.size(), spatial_classes.size())
        
        #Weighted average
        self.weighted_average = torch.sigmoid(self.alpha)
        joint_score = spectral_classes * self.weighted_average + spatial_classes * (1-self.weighted_average)
        
        return joint_score
        
    
def load_from_backbone(state_dict, classes, bands):
    train_state_dict = torch.load(state_dict, map_location="cpu")
    dict_items = train_state_dict.items()
    model = spectral_network(classes=classes, bands=bands)
    dict_to_update = model.state_dict()
    
    #update weights from non-classifier layers
    pretrained_dict = {k: v for k, v in dict_items if not "classifier" in k}
    dict_to_update.update(pretrained_dict)
    model.load_state_dict(dict_to_update)
    
    return model