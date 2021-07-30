import torch
import torchvision
from .add_gcn import ADD_GCN

model_dict = {'ADD_GCN': ADD_GCN}

def get_model(num_classes, args):
    if args.pretrain_name == 'resnet101':
        pretrain = torchvision.models.resnet101(pretrained=True)
    elif args.pretrain_name == 'resnext101-swl':
        model_url = 'facebookresearch/semi-supervised-ImageNet1K-models'
        model_name = 'resnext101_32x4d_swsl'
    else:
        raise Exception('Pretrained models not supported')
        
        pretrain = torch.hub.load(model_url, model_name)

    model = model_dict[args.model_name](pretrain, num_classes)
    return model
