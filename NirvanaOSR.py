import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision.transforms as tf
import numpy as np

from modules.dchs import NirvanaOpenset_loss
from Networks.models import classifier32, ViTB16, ViTB32
from Networks.resnet import resnet50, resnet18, resnet34, resnet101, resnet152
from osr_dataloader import (
    Random300K_Images, BloodMNIST_OSR, OCTMnist_OSR,
    DermaMNIST_OSR, ASC_OSR, breakhis_OSR, DTD_OE, Imagenette_OE
)
from utils import Logger, save_networks, load_networks
from core import test_ddfm_b9, train_Nirvana_oe, train_Nirvana_oe_reg
from split import splits_2020 as splits
from distance_metrics import simplex_distance_metrics_from_loader

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='bloodmnist',
                    choices=['bloodmnist', 'octmnist', 'dermamnist', 'asc', 'breakhis_40'],
                    help="Dataset selection")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./logs_results', help='Directory to save results')

# Optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--l1-weight', type=float, default=0.0, help='L1 regularization weight')
parser.add_argument('--l2-weight', type=float, default=1e-4, help='L2 regularization weight (weight decay)')
parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'rmsprop'],
                    help='Optimizer type to use (sgd or rmsprop)')

# model
parser.add_argument('--noisy-ratio', type=float, default=0.0, help="noisy ratio for ablation study")
parser.add_argument('--m-min', type=float, default=38.0, help="margin for hinge")
parser.add_argument('--m-max', type=float, default=71.0, help="margin for hinge")
parser.add_argument('--Expand', default=100, type=int, metavar='N', help='Expand factor of centers')
parser.add_argument('--outlier-weight', type=float, default=1.0, help='Weight for outlier triplet loss component')
parser.add_argument('--inter-weight', type=float, default=1.0, help='Weight for interclass triplet loss component')
parser.add_argument('--model', type=str, default='classifier32',
                    help='resnet50, classifier32, resnet18, resnet34, resnet101, resnet152, vit_b16, vit_b32')
parser.add_argument('--loss', type=str, default='NirvanaOpenset')
parser.add_argument('--pretrained-model', type=str, default=None, help='Path to your fine-tuned model')

# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log_random30k_noisy_rampfalse')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--oe', action='store_true', help="Outlier Exposure", default=True)
parser.add_argument('--oe-path', type=str,
                    default='./data',
                    help='Path to 300K random images for outlier exposure')
parser.add_argument('--oe-dataset', type=str, default='300k',
                    choices=['300k', 'dtd', 'imagenette'],
                    help='Select outlier exposure dataset. Options: 300k, dtd, imagenette')
parser.add_argument('--use-attn', action='store_true', default=False,
                    help='Pass this flag to use Self-Attention and GAP in the ResNet. Defaults to False.')

parser.add_argument('--num-seeds', type=int, default=1,
                    help='How many seeds to run.')

# ASC dataset imbalance settings
parser.add_argument('--imbalance-ratio', type=int, default=None, choices=[2, 5, 10, 50, 100],
                    help='Imbalance ratio for ASC training data: None (balanced), 2 (very mild), 5 (mild), 10 (mild), 50 (moderate), 100 (severe)')
parser.add_argument('--imbalance-seed', type=int, default=42,
                    help='Random seed for reproducible imbalance creation (ASC only)')


def main_worker(options):
    best_acc_avg = 0.0
    results_best = dict()

    best_acc_avg_b9 = 0.0
    results_b9_best = dict()

    options['ramp_activate'] = False

    seed_everything(int(options['seed']))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu'])
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']:
        use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
    else:
        print("Currently using CPU")

    device = torch.device('cuda' if use_gpu else 'cpu')

    print("{} Preparation".format(options['dataset']))

    if options['dataset'] == 'bloodmnist':
        options['img_size'] = 224
        Data = BloodMNIST_OSR(
            known=options['known'],
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size'],
            image_size=options['img_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader

    elif options['dataset'] == 'octmnist':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        options['img_size'] = 224
        Data = OCTMnist_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size'],
            image_size=options['img_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader

    elif options['dataset'] == 'dermamnist':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        options['img_size'] = 224
        Data = DermaMNIST_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size'],
            image_size=options['img_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader

    elif options['dataset'] == 'asc':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        Data = ASC_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size'],
            imbalance_ratio=options.get('imbalance_ratio'),
            random_state=options.get('imbalance_seed', 42)
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader
        options['img_size'] = 224

    elif options['dataset'] == 'breakhis_40':
        split_dict = splits[options['dataset']][options['item']]
        known = split_dict['known']
        unknown = split_dict['unknown']
        Data = breakhis_OSR(
            known=known,
            unknown=unknown,
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader
        options['img_size'] = 224

    else:
        raise ValueError('No dataset chosen or dataset not supported in this script.')
    print("Outlier exposure mode is on.")
    trainloader_oe = None

    try:
        background_path = os.path.join(
            os.path.dirname(options['dataroot']),
            '300K_random_images',
            '300K_random_images.npy'
        )

        if options.get('oe_path') is not None and os.path.exists(options['oe_path']):
            background_path = options['oe_path']

        if not os.path.exists(background_path):
            raise FileNotFoundError(f"Background dataset not found at {background_path}")

        img_size = 224
        options['img_size'] = 224
        
        oe_transform = tf.Compose([
            tf.Resize((img_size, img_size)),
            tf.RandomCrop(img_size, padding=20),
            tf.RandomHorizontalFlip(),
            tf.ToTensor()
        ])

        oe_dataset_type = options.get('oe_dataset', '300k').lower()
        print(f"Loading outlier exposure dataset: {oe_dataset_type}")

        if oe_dataset_type == 'dtd':
            oe_data = DTD_OE(
                root=options['dataroot'],
                transform=oe_transform,
                download=True
            )
        elif oe_dataset_type == 'imagenette':
            oe_data = Imagenette_OE(
                root=options['dataroot'],
                transform=oe_transform,
                download=True
            )
        else:
            oe_data = Random300K_Images(
                file_path=background_path,
                transform=oe_transform,
                extendable=options['noisy_ratio']
            )

        print(f"Loaded background dataset with {len(oe_data)} images")

        g_oe = torch.Generator()
        g_oe.manual_seed(int(options['seed']))

        trainloader_oe = torch.utils.data.DataLoader(
            oe_data,
            batch_size=options['batch_size'],
            shuffle=True,
            num_workers=4,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g_oe,
            pin_memory=True
        )
        print(f"Background loader created with {len(trainloader_oe)} batches")

        if options.get('noisy_ratio'):
            oe_data.data = oe_data.data[:10000]
            oe_data.data.extend(list(Data.noisy_data))
            print("#of background images {}".format(len(oe_data)))
            options['ramp_activate'] = True

    except Exception as e:
        print(f"Warning: Failed to load background dataset: {str(e)}")
        trainloader_oe = None

    options['num_classes'] = Data.num_classes

    print("Creating model: {}".format(options['model']))
    
    # Check for incompatible flags with ViT models
    if options['model'] in ['vit_b16', 'vit_b32'] and options.get('use_attn', False):
        raise ValueError(
            f"Error: --use-attn flag is not compatible with {options['model']} models. "
            "Vision Transformers already use self-attention mechanisms. "
            "Please omit the --use-attn flag when using ViT models."
        )
    
    if options['model'] == 'classifier32':
        net = classifier32(num_classes=options['num_classes'])
        feat_dim = 128
    elif options['model'] == 'resnet50':
        net = resnet50(pretrained=True, num_classes=options['num_classes'], use_attn=options.get('use_attn', False), img_size=options.get('img_size', 224))
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet18':
        net = resnet18(pretrained=True, num_classes=options['num_classes'], use_attn=options.get('use_attn', False), img_size=options.get('img_size', 224))
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet34':
        net = resnet34(pretrained=True, num_classes=options['num_classes'], use_attn=options.get('use_attn', False), img_size=options.get('img_size', 224))
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet101':
        net = resnet101(pretrained=True, num_classes=options['num_classes'], use_attn=options.get('use_attn', False), img_size=options.get('img_size', 224))
        feat_dim = net.feat_dim
    elif options['model'] == 'resnet152':
        net = resnet152(pretrained=True, num_classes=options['num_classes'], use_attn=options.get('use_attn', False), img_size=options.get('img_size', 224))
        feat_dim = net.feat_dim
    elif options['model'] == 'vit_b16':
        net = ViTB16(num_classes=options['num_classes'], pretrained=True, img_size=options.get('img_size', 224))
        feat_dim = net.feat_dim
    elif options['model'] == 'vit_b32':
        net = ViTB32(num_classes=options['num_classes'], pretrained=True, img_size=options.get('img_size', 224))
        feat_dim = net.feat_dim
    else:
        raise ValueError('Model not supported in this file.')

    # Loss
    options.update({'feat_dim': feat_dim, 'use_gpu': use_gpu})

    criterion = NirvanaOpenset_loss(
        num_classes=options['num_classes'],
        feat_dim=options['feat_dim'],
        precalc_centers=True,
        m_min=options['m_min'],
        m_max=options['m_max'],
        Expand=options['Expand'],
        inter_weight=options['inter_weight'],
        outlier_weight=options['outlier_weight']
    )

    class_counts = torch.zeros(options['num_classes'], dtype=torch.long)
    for _, labels in trainloader:
        class_counts += torch.bincount(labels, minlength=options['num_classes'])
    criterion.set_class_counts(class_counts)

    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()

    dir_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'], options['m_min'], options['m_max'], options['oe'])
    model_path = os.path.join(options['outf'], 'models', options['dataset'], dir_name)
    os.makedirs(model_path, exist_ok=True)

    file_name = '{}_{}_{}_{}_{}_{}'.format(
        options['model'], options['loss'], options['item'],
        options['m_min'], options['m_max'], options['noisy_ratio']
    )

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results, results_b9 = test_ddfm_b9(net, criterion, testloader, outloader, epoch=0, **options)
        variance_stats = simplex_distance_metrics_from_loader(
            net, testloader, options['known'], device, centers=criterion.centers
        )
        variance_stats["intraclass_num_classes"] = len(options['known'])
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
            results['ACC'], results['AUROC'], results['OSCR']
        ))
        return results, results_b9, variance_stats  

    # Optimizer
    l2_weight = options['l2_weight']
    if options['optim'] == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=options['lr'],
            momentum=0.9,
            weight_decay=l2_weight
        )
        print(f"Using SGD optimizer with lr={options['lr']}, momentum=0.9, weight_decay={l2_weight}")
    elif options['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            net.parameters(),
            lr=options['lr'],
            alpha=0.95,
            eps=1e-6,
            weight_decay=l2_weight,
            momentum=0.9,
            centered=False
        )
        print(f"Using RMSprop optimizer with lr={options['lr']}, alpha=0.95, eps=1e-6, weight_decay={l2_weight}, momentum=0.9, centered=False")
    else:
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=options['lr'],
            momentum=0.9,
            weight_decay=l2_weight
        )
        print(f"Unknown optimizer '{options.get('optimizer', 'N/A')}', falling back to SGD")

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['max_epoch'] * len(trainloader))

    start_time = time.time()

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

        with torch.no_grad():
            criterion.class_counts.zero_()
            criterion.compute_margins()

        if options['oe'] and trainloader_oe is not None:
            train_Nirvana_oe_reg(
                net, criterion, optimizer, scheduler,
                trainloader, trainloader_oe,
                epoch=epoch, **options
            )
        else:
            train_Nirvana_oe_reg(
                net, criterion, optimizer, scheduler,
                trainloader,
                epoch=epoch, **options
            )

        if (options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0) or ((epoch + 1) == options['max_epoch']):
            print("==> Test", options['loss'])
            results, results_b9 = test_ddfm_b9(net, criterion, testloader, outloader, epoch=epoch, **options)

            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
                results['ACC'], results['AUROC'], results['OSCR']
            ))
            print("Normalized - Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
                results_b9['ACC'], results_b9['AUROC'], results_b9['OSCR']
            ))

            avg_acc = (results['AUROC'] + results['OSCR']) / 2.0
            if avg_acc >= best_acc_avg:
                best_acc_avg = avg_acc
                results_best = results
                print("Best Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
                    results_best['ACC'], results_best['AUROC'], results_best['OSCR']
                ))
                save_networks(net, model_path, file_name, ext='best', criterion=criterion)

            avg_acc_b9 = (results_b9['AUROC'] + results_b9['OSCR']) / 2.0
            if avg_acc_b9 >= best_acc_avg_b9:
                best_acc_avg_b9 = avg_acc_b9
                results_b9_best = results_b9
                print("Normalized - Best Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
                    results_b9_best['ACC'], results_b9_best['AUROC'], results_b9_best['OSCR']
                ))
                save_networks(net, model_path, file_name, ext='best_b9', criterion=criterion)

            save_networks(net, model_path, file_name, criterion=criterion)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    variance_net = net
    if results_best:
        try:
            variance_net, _ = load_networks(net, model_path, file_name, loss='best')
        except Exception as e:
            print(f"Warning: Failed to load best checkpoint for variance: {str(e)}")
            variance_net = net

    variance_stats = simplex_distance_metrics_from_loader(
        variance_net, testloader, options['known'], device, centers=criterion.centers
    )
    variance_stats["intraclass_num_classes"] = len(options['known'])
    return results_best, results_b9_best, variance_stats


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    print(f'Using device: {device}')

    options = vars(args)
    options['device'] = device
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])

    if options.get('num_seeds', 1) <= 1:
        SEEDS = [int(options['seed'])]  
    else:
        SEEDS = [1, 2, 3, 4, 5][:int(options['num_seeds'])]

    all_results = {}
    all_results_b9 = {}
    all_metrics_b9 = {}

    for seed in SEEDS:
        print(f"\n==============================")
        print(f" Running seed = {seed}")
        print(f"==============================")

        options['seed'] = int(seed)

        results = dict()
        results_b9 = dict()
        metrics_b9 = dict()

        for i in range(len(splits[options['dataset']])):
            split_dict = splits[options['dataset']][i]
            known = split_dict['known']
            unknown = split_dict['unknown']

            options.update({
                'item': i,
                'known': known,
                'unknown': unknown,
                'img_size': 224
            })

            dir_name = '{}_{}_{}_{}_{}_{}_seed{}'.format(
                options['model'], options['loss'], options['m_min'], options['m_max'],
                options['oe'], options['noisy_ratio'], seed
            )
            dir_path = os.path.join(options['outf'], 'results', dir_name)
            os.makedirs(dir_path, exist_ok=True)

            file_name = options['dataset'] + '.csv'
            file_name_b9 = options['dataset'] + '_b9.csv'
            file_name_metrics_b9 = options['dataset'] + '_metrics_b9.csv'

            res, res_b9, metrics = main_worker(options)

            res['unknown'] = unknown
            res['known'] = known
            res['seed'] = seed

            res_b9['unknown'] = unknown
            res_b9['known'] = known
            res_b9['seed'] = seed

            metric_res_b9 = dict(metrics)
            metric_res_b9['ACC'] = res_b9.get('ACC')
            metric_res_b9['AUROC'] = res_b9.get('AUROC')
            metric_res_b9['OSCR'] = res_b9.get('OSCR')
            metric_res_b9['unknown'] = unknown
            metric_res_b9['known'] = known
            metric_res_b9['seed'] = seed

            results[str(i)] = res
            pd.DataFrame(results).to_csv(os.path.join(dir_path, file_name))

            results_b9[str(i)] = res_b9
            pd.DataFrame(results_b9).to_csv(os.path.join(dir_path, file_name_b9))

            metrics_b9[str(i)] = metric_res_b9
            pd.DataFrame(metrics_b9).to_csv(os.path.join(dir_path, file_name_metrics_b9))

        all_results[str(seed)] = results
        all_results_b9[str(seed)] = results_b9
        all_metrics_b9[str(seed)] = metrics_b9

    def per_seed_means(all_res, key):
        """
        all_res[str(seed)][str(split)] -> dict(metric)
        returns: dict {seed(int): mean_over_splits}
        """
        seed_means = {}
        for seed, splits_dict in all_res.items():
            vals = []
            for sp, r in splits_dict.items():
                if key in r:
                    vals.append(float(r[key]))
            vals = np.array(vals, dtype=float)
            if len(vals) == 0:
                continue
            seed_means[int(seed)] = float(vals.mean())
        return seed_means

    def summarize_over_seeds(all_res, key):
        """
        returns:
          mean_over_seeds, std_over_seeds, seed_means_dict
        """
        seed_means = per_seed_means(all_res, key)
        vals = np.array(list(seed_means.values()), dtype=float)
        return float(vals.mean()), float(vals.std()), seed_means

    print("\n==============================")
    print(" Summary (mean over splits → mean ± std over seeds)")
    print("==============================")

    summary_rows_raw = []
    per_seed_rows_raw = []

    for k in ["ACC", "AUROC", "OSCR"]:
        mean_k, std_k, seed_means = summarize_over_seeds(all_results, k)

        print(f"\n[RAW] {k}")
        for s in sorted(seed_means.keys()):
            print(f"  Seed {s}: {seed_means[s]:.3f}")
            per_seed_rows_raw.append({
                "seed": s,
                "metric": k,
                "mean_over_splits": seed_means[s],
                "dataset": options["dataset"]
            })
        print(f"  ==> Final (over seeds): {mean_k:.3f} ± {std_k:.3f}")

        summary_rows_raw.append({
            "metric": k,
            "mean_over_seeds": mean_k,
            "std_over_seeds": std_k,
            "num_seeds": len(seed_means),
            "num_splits": len(splits[options['dataset']]),
            "dataset": options["dataset"]
        })

    summary_rows_b9 = []
    per_seed_rows_b9 = []

    for k in ["ACC", "AUROC", "OSCR"]:
        mean_k, std_k, seed_means = summarize_over_seeds(all_results_b9, k)

        print(f"\n[B9] {k}")
        for s in sorted(seed_means.keys()):
            print(f"  Seed {s}: {seed_means[s]:.3f}")
            per_seed_rows_b9.append({
                "seed": s,
                "metric": k,
                "mean_over_splits": seed_means[s],
                "dataset": options["dataset"]
            })
        print(f"  ==> Final (over seeds): {mean_k:.3f} ± {std_k:.3f}")

        summary_rows_b9.append({
            "metric": k,
            "mean_over_seeds": mean_k,
            "std_over_seeds": std_k,
            "num_seeds": len(seed_means),
            "num_splits": len(splits[options['dataset']]),
            "dataset": options["dataset"]
        })

    df_summary_raw = pd.DataFrame(summary_rows_raw)
    df_summary_b9  = pd.DataFrame(summary_rows_b9)

    df_per_seed_raw = pd.DataFrame(per_seed_rows_raw)
    df_per_seed_b9  = pd.DataFrame(per_seed_rows_b9)

    summary_dir_name = '{}_{}_{}_{}_{}_{}'.format(
        options['model'], options['loss'], options['m_min'], options['m_max'],
        options['oe'], options['noisy_ratio']
    )
    summary_dir = os.path.join(options['outf'], 'results', summary_dir_name)
    os.makedirs(summary_dir, exist_ok=True)

    summary_file_raw = f"{options['dataset']}_summary_mean_std_over_seeds.csv"
    summary_file_b9  = f"{options['dataset']}_summary_mean_std_over_seeds_b9.csv"

    df_summary_raw.to_csv(os.path.join(summary_dir, summary_file_raw), index=False)
    df_summary_b9.to_csv(os.path.join(summary_dir, summary_file_b9), index=False)

    per_seed_file_raw = f"{options['dataset']}_per_seed_mean_over_splits.csv"
    per_seed_file_b9  = f"{options['dataset']}_per_seed_mean_over_splits_b9.csv"

    df_per_seed_raw.to_csv(os.path.join(summary_dir, per_seed_file_raw), index=False)
    df_per_seed_b9.to_csv(os.path.join(summary_dir, per_seed_file_b9), index=False)

    print(f"\nSaved SEED-WISE summary CSVs:")
    print(f"  RAW summary -> {os.path.join(summary_dir, summary_file_raw)}")
    print(f"  B9  summary -> {os.path.join(summary_dir, summary_file_b9)}")
    print(f"  RAW per-seed -> {os.path.join(summary_dir, per_seed_file_raw)}")
    print(f"  B9  per-seed -> {os.path.join(summary_dir, per_seed_file_b9)}")
