import logging

import torch

logger = logging.getLogger(__name__)

from ..tresnet import TResnetL, TResnetM, TResnetXL


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes,'remove_aa_jit': args.remove_aa_jit}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model

def load_tresnetm():
    url = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/tresnet_m_448.pth'
    weights_m = torch.hub.load_state_dict_from_url(url)
    model = TResnetM()
    model.load_state_dict(weights_m)

    return model
