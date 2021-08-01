import torch
import torchvision
from .add_gcn import ADD_GCN
from .res2net_v1b import res2net101_v1b

model_dict = {'ADD_GCN': ADD_GCN}

def get_model(num_classes, args):
    if args.pretrain_name == 'resnet101':
        pretrain = torchvision.models.resnet101(pretrained=True)
    elif args.pretrain_name == 'resnext101-swl':
        model_url = 'facebookresearch/semi-supervised-ImageNet1K-models'
        model_name = 'resnext101_32x4d_swsl'

        pretrain = torch.hub.load(model_url, model_name)
    elif args.pretrain_name == 'res2net101-v1b':
        pretrain = res2net101_v1b(pretrained=True)
    else:
        raise Exception('Pretrained models not supported')

    model = model_dict[args.model_name](pretrain, num_classes)
    return model
