import argparse
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from prediction.model_builder import build_model
import torchmetrics
from prediction.pred_util import find_latest_checkpoint, classes_to_scalar, load_config
import pprint


class VolumetricIOU(torchmetrics.Metric):
    """
    This calculates the IoU summed over the entire dataset, then averaged. This means an image with no
    GT or pred force will contribute none to this metric.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape

        high = torch.maximum(preds, target)
        low = torch.minimum(preds, target)

        self.numerator += torch.sum(low)
        self.denominator += torch.sum(high)

    def compute(self):
        return self.numerator / self.denominator


class ContactIOU(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape
        assert preds.dtype == torch.long    # Make sure we're getting ints

        bool_pred = preds > 0
        bool_gt = target > 0

        self.numerator += torch.sum(bool_gt & bool_pred)
        self.denominator += torch.sum(bool_gt | bool_pred)

    def compute(self):
        return self.numerator / self.denominator


def reset_metrics(all_metrics):
    for key, metric in all_metrics.items():
        metric.reset()


def print_metrics(all_metrics):
    out_dict = dict()
    for key, metric in all_metrics.items():
        out_dict[key] = metric.compute().item()

    pprint.pprint(out_dict)


CONTACT_THRESH = 1.0


def run_metrics(all_metrics, pressure_gt, pressure_pred, config):
    # Takes CUDA BATCHES as input
    pressure_pred = pressure_pred.detach()  # just in case

    contact_pred = (pressure_pred > CONTACT_THRESH).long()
    contact_gt = (pressure_gt > CONTACT_THRESH).long()

    all_metrics['contact_iou'](contact_pred, contact_gt)
    all_metrics['contact_precision'](contact_pred, contact_gt)
    all_metrics['contact_recall'](contact_pred, contact_gt)

    all_metrics['mae'](pressure_pred, pressure_gt)
    all_metrics['vol_iou'](pressure_pred, pressure_gt)

    any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0
    any_contact_gt = torch.sum(contact_gt, dim=(1, 2)) > 0

    all_metrics['temporal_accuracy'](any_contact_pred, any_contact_gt)
    all_metrics['temporal_precision'](any_contact_pred, any_contact_gt)
    all_metrics['temporal_recall'](any_contact_pred, any_contact_gt)
    all_metrics['temporal_f1'](any_contact_pred, any_contact_gt)


def setup_metrics(device):
    all_metrics = dict()

    all_metrics['contact_iou'] = ContactIOU().to(device)
    all_metrics['contact_precision'] = torchmetrics.classification.Precision(task='binary').to(device)
    all_metrics['contact_recall'] = torchmetrics.classification.Recall(task='binary').to(device)
    all_metrics['mae'] = torchmetrics.MeanAbsoluteError().to(device)
    all_metrics['vol_iou'] = VolumetricIOU().to(device)
    all_metrics['temporal_accuracy'] = torchmetrics.Accuracy(task='binary').to(device)
    all_metrics['temporal_precision'] = torchmetrics.classification.Precision(task='binary').to(device)
    all_metrics['temporal_recall'] = torchmetrics.classification.Recall(task='binary').to(device)
    all_metrics['temporal_f1'] = torchmetrics.F1Score(task='binary', num_classes=2, average='macro', mdmc_average='samplewise').to(device)
    return all_metrics


def setup_weak_label_metrics(device):
    weak_metrics = dict()
    weak_metrics['thumb_acc'] = torchmetrics.Accuracy(task='binary').to(device)
    weak_metrics['index_acc'] = torchmetrics.Accuracy(task='binary').to(device)
    weak_metrics['middle_acc'] = torchmetrics.Accuracy(task='binary').to(device)
    weak_metrics['ring_acc'] = torchmetrics.Accuracy(task='binary').to(device)
    weak_metrics['pinky_acc'] = torchmetrics.Accuracy(task='binary').to(device)
    weak_metrics['all_finger_acc'] = torchmetrics.Accuracy(task='binary').to(device)
    weak_metrics['force_acc'] = torchmetrics.Accuracy(task='binary', ignore_index=-1).to(device)
    return weak_metrics


def run_weak_label_metrics(weak_metrics, weak_gt, weak_pred):
    # Takes CUDA BATCHES as input
    weak_pred = weak_pred.detach()

    weak_metrics['thumb_acc'](weak_pred[:, 0], weak_gt[:, 0])
    weak_metrics['index_acc'](weak_pred[:, 1], weak_gt[:, 1])
    weak_metrics['middle_acc'](weak_pred[:, 2], weak_gt[:, 2])
    weak_metrics['ring_acc'](weak_pred[:, 3], weak_gt[:, 3])
    weak_metrics['pinky_acc'](weak_pred[:, 4], weak_gt[:, 4])
    weak_metrics['all_finger_acc'](weak_pred[:, :5].amax(dim=1), weak_gt[:, :5].amax(dim=1))
    weak_metrics['force_acc'](weak_pred[:, 6], weak_gt[:, 6])


def evaluate_split(config, device, evaluate_weak=False, force_test_on_test=False):
    config.DATALOADER_TEST_SKIP_FRAMES = 64

    if force_test_on_test:
        print('Testing on actual test set, not validation set!')
        config.VAL_FILTER = config.TEST_FILTER
        config.VAL_WEAK_FILTER = config.TEST_WEAK_FILTER
        config.DATALOADER_TEST_SKIP_FRAMES = 1

    random.seed(5)  # Set the seed so the sequences will be randomized the same
    if evaluate_weak:
        model_dict = build_model(config, device, ['val_weak'])
    else:
        model_dict = build_model(config, device, ['val'])

    checkpoint_path = find_latest_checkpoint(config.CONFIG_NAME)

    try:
        best_model = torch.load(checkpoint_path)
        best_model.eval()
        # torch.save(best_model.state_dict(), 'state_dict_temp.pt')
    except:
        best_model = model_dict['model']
        best_model.load_state_dict(torch.load(checkpoint_path))
        best_model.eval()

    if evaluate_weak:
        val_loader = DataLoader(model_dict['val_weak_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    else:
        val_loader = DataLoader(model_dict['val_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    all_metrics = setup_metrics(device)
    weak_metrics = setup_weak_label_metrics(device)

    for idx, batch in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            image = batch['img']
            force_gt_scalar = batch['raw_force'].cuda()
            weak_labels_source_gt = batch['fingers'].cuda()

            model_output = best_model(image.cuda())
            force_pred_class = model_output[0]

            force_pred_class = torch.argmax(force_pred_class, dim=1)
            force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)

            if evaluate_weak:
                contact_pred = (force_pred_scalar > CONTACT_THRESH).long()
                any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0
                any_contact_gt = torch.amax(weak_labels_source_gt, dim=1) > 0
                all_metrics['temporal_accuracy'](any_contact_pred, any_contact_gt)
                all_metrics['temporal_precision'](any_contact_pred, any_contact_gt)
                all_metrics['temporal_recall'](any_contact_pred, any_contact_gt)
                all_metrics['temporal_f1'](any_contact_pred, any_contact_gt)
            else:
                run_metrics(all_metrics, force_gt_scalar, force_pred_scalar, config)

            fingers_pred = model_output[1]['bottleneck_logits']
            fingers_gt = batch['fingers'].cuda()
            run_weak_label_metrics(weak_metrics, fingers_gt, fingers_pred)

    print_metrics(all_metrics)
    print_metrics(weak_metrics)


def evaluate(config, device, force_test_on_test=False):
    print('Running evaluation on fully-labeled data')
    evaluate_split(config, device, evaluate_weak=False, force_test_on_test=force_test_on_test)
    print('Running evaluation on weakly-labeled data')
    evaluate_split(config, device, evaluate_weak=True, force_test_on_test=force_test_on_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--TEST_ON_TEST', action='store_true')
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate(config, device, force_test_on_test=args.TEST_ON_TEST)
