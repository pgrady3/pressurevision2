import torch
from prediction.model.fpn_dann_logits_model import FPN_DANN_Logits
from prediction.model.masked_binary_cross_entropy import MaskedBCELoss
from prediction.model.soft_cross_entropy import SoftCrossEntropyLoss
from prediction.pred_util import resnet_preprocessor
from prediction.loader import ForceDataset

import ssl  # Hack to get around SSL certificate of the segmentation_models_pytorch being out of date
ssl._create_default_https_context = ssl._create_unverified_context


def build_model(config, device, phases):
    out_dict = dict()

    if config.USE_SOFT_CROSS_ENTROPY:
        weight = [float(config.FORCE_CLASSIFICATION_NONZERO_WEIGHT)] * config.NUM_FORCE_CLASSES
        weight[0] = 1
        out_dict['criterion'] = SoftCrossEntropyLoss(omega=config.SOFT_CROSS_ENTROPY_OMEGA, num_classes=config.NUM_FORCE_CLASSES, weight=torch.tensor(weight).cuda())
    else:
        # Normal cross-entropy, weighted in this case
        weight = [float(config.FORCE_CLASSIFICATION_NONZERO_WEIGHT)] * config.NUM_FORCE_CLASSES
        weight[0] = 1
        out_dict['criterion'] = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda())

    out_channels = config.NUM_FORCE_CLASSES

    out_dict['criterion_fingers'] = MaskedBCELoss()
    out_dict['criterion_dann'] = torch.nn.CrossEntropyLoss()

    print('Loss function:', out_dict['criterion'])
    num_weak_logits = 6
    if config.WEAK_LABEL_HIGH_LOW:
        num_weak_logits = 7

    if config.NETWORK_TYPE == 'fpn_dann_logits':
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'

        # create segmentation model with pretrained encoder
        model = FPN_DANN_Logits(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=out_channels,
            activation=None,    # raw logits
            in_channels=config.NETWORK_INPUT_CHANNELS,
            num_out_logits=num_weak_logits
        )

        preprocessing_fn = resnet_preprocessor
    else:
        raise ValueError('Unknown model')

    model = model.to(device)
    out_dict['model'] = model

    if 'train' in phases:
        out_dict['train_dataset'] = ForceDataset(config, config.TRAIN_FILTER,
                                                 skip_frames=config.DATALOADER_TRAIN_SKIP_FRAMES,
                                                 preprocessing_fn=preprocessing_fn,
                                                 phase='train')

    if 'target_domain' in phases:
        out_dict['train_weak_dataset'] = ForceDataset(config, config.TRAIN_WEAK_FILTER,
                                                 skip_frames=config.DATALOADER_TRAIN_SKIP_FRAMES,
                                                 preprocessing_fn=preprocessing_fn,
                                                 phase='train')

    if 'val' in phases:
        out_dict['val_dataset'] = ForceDataset(config, config.VAL_FILTER,
                                                skip_frames=config.DATALOADER_TEST_SKIP_FRAMES,
                                                preprocessing_fn=preprocessing_fn,
                                                phase='val')

    if 'val_weak' in phases:
        out_dict['val_weak_dataset'] = ForceDataset(config, config.VAL_WEAK_FILTER,
                                                 skip_frames=config.DATALOADER_TEST_SKIP_FRAMES,
                                                 preprocessing_fn=preprocessing_fn,
                                                 phase='val')

    return out_dict

