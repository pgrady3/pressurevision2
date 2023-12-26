import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
from prediction.model_builder import build_model
import prediction.evaluator as evaluator
from prediction.pred_util import classes_to_scalar, parse_config_args, find_latest_checkpoint
from recording.util import AverageMeter, mkdir
from tqdm import tqdm


def val_epoch(config, val_metrics):
    model.eval()
    evaluator.reset_metrics(val_metrics)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader)):
            image = data['img']
            image_gpu = image.cuda()
            force_gt_scalar = data['raw_force'].cuda()

            force_estimated, _ = model(image_gpu, alpha=0)

            force_pred_class = torch.argmax(force_estimated, dim=1)
            force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)

            evaluator.run_metrics(val_metrics, force_gt_scalar, force_pred_scalar, config)

    for key, metric in val_metrics.items():
        writer.add_scalar('val/' + key, metric.compute(), global_iter)

    writer.flush()
    print('Finished val epoch: {}. Avg temporal acc {:.4f} --------------------'.format(epoch, val_metrics['temporal_accuracy'].compute()))


def val_weak_epoch(config, val_metrics):
    model.eval()
    evaluator.reset_metrics(val_metrics)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_weak_loader)):
            image = data['img']
            image_gpu = image.cuda()
            weak_labels_source_gt = data['fingers'].cuda()

            force_estimated, _ = model(image_gpu, alpha=0)

            force_pred_class = torch.argmax(force_estimated, dim=1)
            force_pred_scalar = classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)

            contact_pred = (force_pred_scalar > evaluator.CONTACT_THRESH).long()
            any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0
            any_contact_gt = torch.amax(weak_labels_source_gt, dim=1) > 0
            val_metrics['temporal_accuracy'](any_contact_pred, any_contact_gt)
            val_metrics['temporal_precision'](any_contact_pred, any_contact_gt)
            val_metrics['temporal_recall'](any_contact_pred, any_contact_gt)
            val_metrics['temporal_f1'](any_contact_pred, any_contact_gt)

    writer.add_scalar('val_weak/temporal_accuracy', val_metrics['temporal_accuracy'].compute(), global_iter)
    writer.add_scalar('val_weak/temporal_precision', val_metrics['temporal_precision'].compute(), global_iter)
    writer.add_scalar('val_weak/temporal_recall', val_metrics['temporal_recall'].compute(), global_iter)
    writer.add_scalar('val_weak/temporal_f1', val_metrics['temporal_f1'].compute(), global_iter)

    writer.flush()
    print('Finished val weak epoch: {}. Avg temporal acc {:.4f} --------------------'.format(epoch, val_metrics['temporal_accuracy'].compute()))


def train_epoch(config):
    model.train()
    loss_image_meter = AverageMeter('Loss image', ':.4e')
    loss_domain_source_meter = AverageMeter('Loss domain source', ':.4e')
    loss_domain_target_meter = AverageMeter('Loss domain target', ':.4e')
    loss_logits_source_meter = AverageMeter('Loss logits source', ':.4e')
    loss_logits_target_meter = AverageMeter('Loss logits target', ':.4e')

    iterations = 0
    global global_iter

    train_iterator = iter(train_loader)
    train_weak_iterator = iter(train_weak_loader)

    p = epoch / config.MAX_EPOCHS   # [0-1), how far along with training are we
    alpha = 2. / (1 + np.exp(-10 * p)) - 1  # The DANN gradient reversal weight. This is the strength of the adversarial loss
    # print('Alpha', alpha)

    with tqdm(total=config.TRAIN_ITERS_PER_EPOCH) as progress_bar:
        while iterations < config.TRAIN_ITERS_PER_EPOCH:

            try:    # Use the DataLoader in a special way to allow it to hand fixed-size epochs
                data_source = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                data_source = next(train_iterator)

            try:
                data_target = next(train_weak_iterator)
            except StopIteration:
                train_weak_iterator = iter(train_weak_loader)
                data_target = next(train_weak_iterator)

            image_source_gpu = data_source['img'].cuda()
            image_target_gpu = data_target['img'].cuda()
            force_gt_source_gpu = data_source['force'].cuda()
            batch_size_source = data_source['img'].shape[0]
            batch_size_target = data_target['img'].shape[0]
            weak_labels_source_gt = data_source['fingers'].cuda()
            weak_labels_target_gt = data_target['fingers'].cuda()

            domain_source_gt = torch.zeros(batch_size_source, dtype=torch.long, device=image_source_gpu.device)
            domain_target_gt = torch.ones(batch_size_target, dtype=torch.long, device=image_target_gpu.device)

            force_estimated_source, dict_source_est = model(image_source_gpu, alpha=alpha)
            force_estimated_target, dict_target_est = model(image_target_gpu, alpha=alpha)

            loss_pressure = criterion(force_estimated_source, force_gt_source_gpu)
            loss_domain_source = criterion_dann(dict_source_est['domain_logits'], domain_source_gt) * config.LAMBDA_DOMAIN
            loss_domain_target = criterion_dann(dict_target_est['domain_logits'], domain_target_gt) * config.LAMBDA_DOMAIN
            loss_logits_source = criterion_fingers(dict_source_est['bottleneck_logits'], weak_labels_source_gt) * config.LAMBDA_FINGERS_SOURCE
            loss_logits_target = criterion_fingers(dict_target_est['bottleneck_logits'], weak_labels_target_gt) * config.LAMBDA_FINGERS_TARGET

            loss = loss_pressure + loss_domain_source + loss_domain_target + loss_logits_source + loss_logits_target

            loss_image_meter.update(loss_pressure.item(), batch_size_source)
            loss_domain_source_meter.update(loss_domain_source.item(), batch_size_source)
            loss_domain_target_meter.update(loss_domain_target.item(), batch_size_source)
            loss_logits_source_meter.update(loss_logits_source.item(), batch_size_source)
            loss_logits_target_meter.update(loss_logits_target.item(), batch_size_source)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iterations += batch_size_source
            global_iter += batch_size_source
            progress_bar.update(batch_size_source)     # Incremental update
            progress_bar.set_postfix(loss=str(loss_image_meter))

    writer.add_scalar('training/loss_image', loss_image_meter.avg, global_iter)
    writer.add_scalar('training/loss_domain_source', loss_domain_source_meter.avg, global_iter)
    writer.add_scalar('training/loss_domain_target', loss_domain_target_meter.avg, global_iter)
    writer.add_scalar('training/loss_logits_source', loss_logits_source_meter.avg, global_iter)
    writer.add_scalar('training/loss_logits_target', loss_logits_target_meter.avg, global_iter)
    writer.add_scalar('training/alpha', alpha, global_iter)
    writer.add_scalar('training/lr', scheduler.get_last_lr()[0], global_iter)
    print('Finished training epoch: {}. Avg loss {:.4f} --------------------'.format(epoch, loss_image_meter.avg))
    writer.flush()


if __name__ == "__main__":
    config = parse_config_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = build_model(config, device, ['train', 'target_domain', 'val', 'val_weak'])
    criterion = model_dict['criterion']
    criterion_dann = model_dict['criterion_dann']
    criterion_fingers = model_dict['criterion_fingers']
    model = model_dict['model']

    if hasattr(config, 'USE_CHECKPOINT'):
        checkpoint_path = find_latest_checkpoint(config.USE_CHECKPOINT)
        print('LOADING CHECKPOINT FROM:', checkpoint_path)
        model = torch.load(checkpoint_path)

    train_loader = DataLoader(model_dict['train_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    train_weak_loader = DataLoader(model_dict['train_weak_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(model_dict['val_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    val_weak_loader = DataLoader(model_dict['val_weak_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config.LEARNING_RATE_INITIAL)])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.LEARNING_RATE_SCHEDULER_STEP,
                                                     gamma=config.LEARNING_RATE_SCHEDULER_GAMMA)

    desc = config.CONFIG_NAME + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter('runs/' + desc)
    global_iter = 0

    val_metrics = evaluator.setup_metrics(device)

    for epoch in range(config.MAX_EPOCHS):
        train_epoch(config)
        val_epoch(config, val_metrics)
        val_weak_epoch(config, val_metrics)
        save_path = 'data/model/{}_{}.pth'.format(config.CONFIG_NAME, epoch)
        mkdir(save_path, cut_filename=True)
        torch.save(model, save_path)
        scheduler.step()
        print('\n')

    evaluator.evaluate(config, device, force_test_on_test=True)
