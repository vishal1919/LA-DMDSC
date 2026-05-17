import os
import argparse
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import silhouette_score

from split import splits_2020 as splits
from osr_dataloader import (
    BloodMNIST_OSR, DermaMNIST_OSR,
    ASC_OSR, breakhis_OSR
)
from Networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from Networks.models import classifier32
from utils import load_networks


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(args, split_id):
    split_dict = splits[args.dataset][split_id]
    known = split_dict["known"]
    unknown = split_dict["unknown"]

    dataroot = os.path.join(args.dataroot, args.dataset)

    if args.dataset == "bloodmnist":
        Data = BloodMNIST_OSR(
            known=known,
            #unknown=unknown,
            dataroot=dataroot,
            use_gpu=not args.use_cpu,
            batch_size=args.batch_size
        )

    elif args.dataset == "dermamnist":
        Data = DermaMNIST_OSR(
            known=known,
            unknown=unknown,
            dataroot=dataroot,
            use_gpu=not args.use_cpu,
            batch_size=args.batch_size
        )

    elif args.dataset == "asc":
        Data = ASC_OSR(
            known=known,
            unknown=unknown,
            dataroot=dataroot,
            use_gpu=not args.use_cpu,
            batch_size=args.batch_size
        )

    elif args.dataset == "breakhis_40":
        Data = breakhis_OSR(
            known=known,
            unknown=unknown,
            dataroot=dataroot,
            use_gpu=not args.use_cpu,
            batch_size=args.batch_size
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return Data, known, unknown


def build_model(args, num_classes, use_attn):
    if args.model == "classifier32":
        return classifier32(num_classes=num_classes)

    kwargs = {
        "pretrained": False,
        "num_classes": num_classes,
        "use_attn": use_attn,
        "img_size": args.img_size
    }

    if args.model == "resnet18":
        return resnet18(**kwargs)
    elif args.model == "resnet34":
        return resnet34(**kwargs)
    elif args.model == "resnet50":
        return resnet50(**kwargs)
    elif args.model == "resnet101":
        return resnet101(**kwargs)
    elif args.model == "resnet152":
        return resnet152(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def make_file_name(args, split_id):
    if args.filename_mode == "single":
        return f"{args.model}_{args.loss}_{split_id}_{args.margin}_{args.noisy_ratio}"

    elif args.filename_mode == "minmax":
        return f"{args.model}_{args.loss}_{split_id}_{args.m_min}_{args.m_max}_{args.noisy_ratio}"

    else:
        raise ValueError("filename_mode must be either 'single' or 'minmax'")


def checkpoint_exists(model_dir, file_name):
    ckpt_path = os.path.join(model_dir, "checkpoints", file_name + "_.pth")
    return os.path.exists(ckpt_path), ckpt_path


def load_saved_model(net, args, split_id, use_attn, device):
    model_dir = args.attn_model_dir if use_attn else args.no_attn_model_dir
    attn_name = "WITH attention" if use_attn else "WITHOUT attention"

    file_name = make_file_name(args, split_id)

    exists, ckpt_path = checkpoint_exists(model_dir, file_name)

    print(f"\nLoading model {attn_name}")
    print("Model dir :", model_dir)
    print("File name :", file_name)
    print("Checkpoint:", ckpt_path)

    if not exists:
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{ckpt_path}\n\n"
            f"Check --filename-mode, --margin, --m-min, --m-max, and model folder."
        )

    net, _ = load_networks(
        net,
        model_dir,
        file_name,
        criterion=None
    )

    net = net.to(device)
    net.eval()
    return net


def extract_features(net, loader, device):
    net.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features, logits = net(images, return_feature=True)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


def calculate_silhouette(features, labels):
    """
    Raw Euclidean silhouette score.

    This is mathematically aligned with NirvanaOpenset_loss,
    because the loss uses Euclidean distance between features and centers.
    """

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= len(labels):
        return np.nan

    score = silhouette_score(
        features,
        labels,
        metric="euclidean"
    )

    return score


def evaluate_one_setting(args, split_id, use_attn, device):
    Data, known, unknown = build_dataset(args, split_id)

    net = build_model(
        args=args,
        num_classes=Data.num_classes,
        use_attn=use_attn
    )

    net = load_saved_model(
        net=net,
        args=args,
        split_id=split_id,
        use_attn=use_attn,
        device=device
    )

    features, labels = extract_features(net, Data.test_loader, device)

    raw_score = calculate_silhouette(features, labels)

    return {
        "split": split_id,
        "known": str(known),
        "unknown": str(unknown),
        "use_attention": use_attn,
        "silhouette_raw_euclidean": raw_score,
        "num_samples": len(labels),
        "num_classes": len(np.unique(labels))
    }


def main():
    parser = argparse.ArgumentParser(
        "Raw Euclidean Silhouette Measure for Known Classes"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="asc",
        choices=["bloodmnist", "dermamnist", "asc", "breakhis_40"]
    )

    parser.add_argument("--dataroot", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["classifier32", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    )

    parser.add_argument("--loss", type=str, default="NirvanaOpenset")

    parser.add_argument(
        "--filename-mode",
        type=str,
        default="minmax",
        choices=["single", "minmax"],
        help="single: model_loss_split_margin_noisy | minmax: model_loss_split_mmin_mmax_noisy"
    )

    parser.add_argument("--margin", type=float, default=35.0)
    parser.add_argument("--m-min", type=float, default=35.0)
    parser.add_argument("--m-max", type=float, default=55.0)
    parser.add_argument("--noisy-ratio", type=float, default=0.0)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use-cpu", action="store_true")

    parser.add_argument("--attn-model-dir", type=str, required=True)
    parser.add_argument("--no-attn-model-dir", type=str, required=True)

    parser.add_argument("--save-csv", type=str, default="./silhouette_results.csv")

    args = parser.parse_args()

    seed_everything(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )

    print("Using device:", device)
    print("Dataset:", args.dataset)
    print("Model:", args.model)
    print("Filename mode:", args.filename_mode)
    print("Silhouette metric: Raw Euclidean")

    all_rows = []

    for split_id in range(len(splits[args.dataset])):
        print(f"\n========== Split {split_id} ==========")

        row_no_attn = evaluate_one_setting(
            args=args,
            split_id=split_id,
            use_attn=False,
            device=device
        )

        all_rows.append(row_no_attn)

        row_attn = evaluate_one_setting(
            args=args,
            split_id=split_id,
            use_attn=True,
            device=device
        )

        all_rows.append(row_attn)

        print("\nWithout Attention:")
        print("Raw Euclidean Silhouette:", row_no_attn["silhouette_raw_euclidean"])

        print("\nWith Attention:")
        print("Raw Euclidean Silhouette:", row_attn["silhouette_raw_euclidean"])

    df = pd.DataFrame(all_rows)

    summary = df.groupby("use_attention")[[
        "silhouette_raw_euclidean"
    ]].agg(["mean", "std"])

    print("\n========== Final Summary ==========")
    print(summary)

    save_dir = os.path.dirname(args.save_csv)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    df.to_csv(args.save_csv, index=False)

    summary_csv = args.save_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_csv)

    print("\nSaved split-wise results:", args.save_csv)
    print("Saved summary:", summary_csv)


if __name__ == "__main__":
    main()
