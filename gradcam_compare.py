import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from split import splits_2020 as splits
from osr_dataloader import (
    BloodMNIST_OSR,
    DermaMNIST_OSR,
    ASC_OSR,
    breakhis_OSR,
)
from Networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from modules.dchs import NirvanaOpenset_loss


DATASET_DISPLAY_NAMES = {
    "bloodmnist": "BloodMNIST",
    "dermamnist": "DermaMNIST",
    "asc": "ASC",
    "breakhis_40": "BreaKHis_40",
}


# ==========================================================
# Dataset
# ==========================================================
def build_data(dataset_name, dataroot_base, item, batch_size, img_size):
    split = splits[dataset_name][item]
    known, unknown = split["known"], split["unknown"]

    dataroot = os.path.join(dataroot_base, dataset_name)

    if dataset_name == "bloodmnist":
        data = BloodMNIST_OSR(
            known=known,
            dataroot=dataroot,
            use_gpu=torch.cuda.is_available(),
            batch_size=batch_size,
            image_size=img_size,
        )

    elif dataset_name == "dermamnist":
        data = DermaMNIST_OSR(
            known=known,
            unknown=unknown,
            dataroot=dataroot,
            use_gpu=torch.cuda.is_available(),
            batch_size=batch_size,
            image_size=img_size,
        )

    elif dataset_name == "asc":
        data = ASC_OSR(
            known=known,
            unknown=unknown,
            dataroot=dataroot,
            use_gpu=torch.cuda.is_available(),
            batch_size=batch_size,
        )

    elif dataset_name == "breakhis_40":
        data = breakhis_OSR(
            known=known,
            unknown=unknown,
            dataroot=dataroot,
            use_gpu=torch.cuda.is_available(),
            batch_size=batch_size,
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return data, known, unknown


# ==========================================================
# Model
# ==========================================================
def build_model(
    model_name,
    num_classes,
    use_attn,
    img_size,
    attn_layer="none",
):
    """
    Requires corrected Networks/resnet.py supporting:
        attn_layer = "none", "layer3", "layer4", "layer3+layer4"
    """

    if not use_attn:
        attn_layer = "none"

    if model_name == "resnet18":
        return resnet18(
            pretrained=False,
            num_classes=num_classes,
            use_attn=use_attn,
            img_size=img_size,
            attn_layer=attn_layer,
        )

    if model_name == "resnet34":
        return resnet34(
            pretrained=False,
            num_classes=num_classes,
            use_attn=use_attn,
            img_size=img_size,
            attn_layer=attn_layer,
        )

    if model_name == "resnet50":
        return resnet50(
            pretrained=False,
            num_classes=num_classes,
            use_attn=use_attn,
            img_size=img_size,
            attn_layer=attn_layer,
        )

    if model_name == "resnet101":
        return resnet101(
            pretrained=False,
            num_classes=num_classes,
            use_attn=use_attn,
            img_size=img_size,
            attn_layer=attn_layer,
        )

    if model_name == "resnet152":
        return resnet152(
            pretrained=False,
            num_classes=num_classes,
            use_attn=use_attn,
            img_size=img_size,
            attn_layer=attn_layer,
        )

    raise ValueError("Use resnet18/resnet34/resnet50/resnet101/resnet152")


def build_criterion(num_classes, feat_dim, args):
    return NirvanaOpenset_loss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        precalc_centers=True,
        m_min=args.m_min,
        m_max=args.m_max,
        Expand=args.Expand,
    )


# ==========================================================
# Checkpoint loading
# ==========================================================
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "net" in ckpt:
            return ckpt["net"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]

    return ckpt


def clean_state_dict_keys(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        cleaned[new_key] = value

    return cleaned


def load_checkpoint(net, criterion, ckpt_path, device, model_name="model"):
    if os.path.isdir(ckpt_path):
        raise IsADirectoryError(
            f"Checkpoint path is a directory, not .pth file:\n{ckpt_path}"
        )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found:\n{ckpt_path}")

    print(f"\nLoading {model_name} checkpoint:")
    print(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(ckpt)
    state_dict = clean_state_dict_keys(state_dict)

    try:
        net.load_state_dict(state_dict, strict=True)
        print(f"{model_name} loaded successfully with strict=True")

    except RuntimeError as error:
        print("\nSTRICT LOADING FAILED")
        print(error)

        raise RuntimeError(
            "\nCheckpoint and model architecture do not match.\n"
            "Do not use strict=False for Grad-CAM because it can generate invalid maps.\n"
            "Fix one of these:\n"
            "1. Use the correct checkpoint for this dataset.\n"
            "2. Use the same attention architecture as training.\n"
            "3. Check whether checkpoint was trained with layer3, layer4, or layer3+layer4 attention.\n"
        )

    if isinstance(ckpt, dict):
        if "criterion" in ckpt and ckpt["criterion"] is not None:
            try:
                criterion.load_state_dict(ckpt["criterion"], strict=True)
                print("Criterion loaded with strict=True")
            except Exception as e:
                print(f"Criterion strict loading skipped: {e}")

        if "centers" in ckpt:
            with torch.no_grad():
                criterion.centers.copy_(ckpt["centers"].to(criterion.centers.device))
                print("Centers loaded from checkpoint.")

    return net, criterion


# ==========================================================
# Parsing helpers
# ==========================================================
def parse_list_argument(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_layer_map(value, valid_layers, map_name):
    """
    Example:
        bloodmnist:layer4,dermamnist:layer3,asc:layer4,breakhis_40:layer3
        bloodmnist:self_attn,dermamnist:self_attn,asc:self_attn,breakhis_40:self_attn
    """

    layer_map = {}

    if value is None or value.strip() == "":
        return layer_map

    pairs = [v.strip() for v in value.split(",") if v.strip()]

    for pair in pairs:
        if ":" not in pair:
            raise ValueError(
                f"Invalid {map_name} format: {pair}\n"
                "Correct format: dataset:layer or dataset:layer3+layer4"
            )

        dataset_name, layers = pair.split(":", 1)
        dataset_name = dataset_name.strip()

        selected_layers = [x.strip() for x in layers.split("+") if x.strip()]

        if len(selected_layers) == 0:
            raise ValueError(f"No layer selected for dataset: {dataset_name}")

        for layer in selected_layers:
            if layer not in valid_layers:
                raise ValueError(
                    f"Invalid layer '{layer}' for {dataset_name}. "
                    f"Allowed layers: {valid_layers}"
                )

        layer_map[dataset_name] = selected_layers

    return layer_map


def parse_attention_layers(value):
    """
    Controls actual WITH-attention model architecture.

    Mapping:
        layer3        -> checkpoint keys self_attn.*
        layer4        -> checkpoint keys self_attn.*
        layer3+layer4 -> checkpoint keys self_attn3.* and self_attn4.*
    """

    attn_map = {}

    if value is None or value.strip() == "":
        return attn_map

    pairs = [v.strip() for v in value.split(",") if v.strip()]
    valid_archs = ["layer3", "layer4", "layer3+layer4", "none"]

    for pair in pairs:
        if ":" not in pair:
            raise ValueError(
                f"Invalid attention-layer format: {pair}\n"
                "Correct format: dataset:layer3 or dataset:layer4 or dataset:layer3+layer4"
            )

        dataset_name, arch = pair.split(":", 1)
        dataset_name = dataset_name.strip()
        arch = arch.strip()

        if arch not in valid_archs:
            raise ValueError(
                f"Invalid attention architecture '{arch}' for {dataset_name}. "
                f"Allowed: {valid_archs}"
            )

        attn_map[dataset_name] = arch

    return attn_map


# ==========================================================
# Grad-CAM target layer
# ==========================================================
def get_target_layer(net, layer_name):
    if layer_name == "layer3":
        return net.layer3[-1]

    if layer_name == "layer4":
        return net.layer4[-1]

    if layer_name == "self_attn":
        if not hasattr(net, "self_attn"):
            raise ValueError(
                "This model does not have self_attn. "
                "Use attention architecture layer3 or layer4."
            )
        return net.self_attn

    if layer_name == "self_attn3":
        if not hasattr(net, "self_attn3"):
            raise ValueError(
                "This model does not have self_attn3. "
                "Use attention architecture layer3+layer4."
            )
        return net.self_attn3

    if layer_name == "self_attn4":
        if not hasattr(net, "self_attn4"):
            raise ValueError(
                "This model does not have self_attn4. "
                "Use attention architecture layer3+layer4."
            )
        return net.self_attn4

    raise ValueError(
        "target_layer must be layer3, layer4, self_attn, self_attn3, or self_attn4"
    )


def print_attention_debug(model, dataset_name):
    print(f"\nAttention debug for {dataset_name}:")
    if hasattr(model, "attn_layer"):
        print("model.attn_layer:", model.attn_layer)

    if hasattr(model, "self_attn"):
        print("self_attn query weight:", tuple(model.self_attn.query_conv.weight.shape))

    if hasattr(model, "self_attn3"):
        print("self_attn3 query weight:", tuple(model.self_attn3.query_conv.weight.shape))

    if hasattr(model, "self_attn4"):
        print("self_attn4 query weight:", tuple(model.self_attn4.query_conv.weight.shape))


# ==========================================================
# Grad-CAM core
# ==========================================================
class DistanceGradCAM:
    def __init__(self, model, criterion, target_layers):
        self.model = model
        self.criterion = criterion

        if not isinstance(target_layers, list):
            target_layers = [target_layers]

        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.handles = []

        for idx, layer in enumerate(self.target_layers):
            self.handles.append(
                layer.register_forward_hook(self._make_forward_hook(idx))
            )
            self.handles.append(
                layer.register_full_backward_hook(self._make_backward_hook(idx))
            )

    def _make_forward_hook(self, idx):
        def forward_hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[idx] = output

        return forward_hook

    def _make_backward_hook(self, idx):
        def backward_hook(module, grad_input, grad_output):
            self.gradients[idx] = grad_output[0]

        return backward_hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def _single_cam(self, acts, grads, out_size):
        acts = acts[0]
        grads = grads[0]

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def __call__(self, image_tensor):
        self.model.zero_grad(set_to_none=True)
        self.activations = {}
        self.gradients = {}

        features, _ = self.model(image_tensor, True)

        centers = self.criterion.centers.to(features.device).float()
        features = features.float()

        # Distance-based DSC/simplex classifier score:
        # score = - || f(x) - s_c ||^2
        squared_dists = torch.sum(
            (features.unsqueeze(1) - centers.unsqueeze(0)) ** 2,
            dim=2,
        )

        pred = torch.argmin(squared_dists, dim=1)
        score = -squared_dists[0, pred[0]]

        score.backward(retain_graph=True)

        out_size = image_tensor.shape[-2:]
        cams = []

        for idx in range(len(self.target_layers)):
            if idx not in self.activations:
                raise RuntimeError(f"Activation missing for target layer index {idx}")

            if idx not in self.gradients:
                raise RuntimeError(f"Gradient missing for target layer index {idx}")

            cam = self._single_cam(
                self.activations[idx],
                self.gradients[idx],
                out_size,
            )

            cams.append(cam)

        final_cam = torch.stack(cams, dim=0).mean(dim=0)
        final_cam = final_cam - final_cam.min()
        final_cam = final_cam / (final_cam.max() + 1e-8)

        return final_cam.detach().cpu().numpy()


# ==========================================================
# Visualization utilities
# ==========================================================
def denormalize(img):
    img = img.detach().cpu()

    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    img = img * std + mean
    img = img.clamp(0, 1)

    return img


def overlay_cam(img, cam, alpha=0.38):
    img_np = denormalize(img).permute(1, 2, 0).numpy()

    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0).float()
    cam_t = F.interpolate(
        cam_t,
        size=img_np.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    cam_np = cam_t.squeeze().numpy()
    cam_np = cam_np - cam_np.min()
    cam_np = cam_np / (cam_np.max() + 1e-8)

    heatmap = plt.get_cmap("jet")(cam_np)[:, :, :3]

    overlay = (1.0 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def get_one_sample(data_loader, device, sample_index=0):
    count = 0

    for images, labels in data_loader:
        for i in range(images.size(0)):
            if count == sample_index:
                label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                return images[i : i + 1].to(device), images[i].detach().cpu(), label

            count += 1

    raise RuntimeError("sample_index is larger than available test samples.")


# ==========================================================
# Per-dataset processing
# ==========================================================
def process_one_dataset(
    dataset_name,
    item,
    checkpoint_without,
    checkpoint_with,
    args,
    device,
):
    print("\n" + "=" * 70)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 70)

    if dataset_name not in args.gradcam_layer_map_without:
        raise ValueError(
            f"No DMDSC Grad-CAM layer selected for dataset '{dataset_name}'. "
            "Please provide it using --gradcam-layers-without."
        )

    if dataset_name not in args.gradcam_layer_map_with:
        raise ValueError(
            f"No LA-DMDSC Grad-CAM layer selected for dataset '{dataset_name}'. "
            "Please provide it using --gradcam-layers-with."
        )

    if dataset_name not in args.attention_layer_map:
        raise ValueError(
            f"No attention architecture selected for dataset '{dataset_name}'. "
            "Please provide it using --attention-layers."
        )

    selected_layers_without = args.gradcam_layer_map_without[dataset_name]
    selected_layers_with = args.gradcam_layer_map_with[dataset_name]
    selected_attention_arch = args.attention_layer_map[dataset_name]

    print(f"Selected DMDSC Grad-CAM layer(s): {selected_layers_without}")
    print(f"Selected LA-DMDSC Grad-CAM layer(s): {selected_layers_with}")
    print(f"Selected attention architecture: {selected_attention_arch}")

    data, known, unknown = build_data(
        dataset_name=dataset_name,
        dataroot_base=args.dataroot,
        item=item,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )

    num_classes = data.num_classes

    print(f"Known classes: {known}")
    print(f"Unknown classes: {unknown}")

    # ---------------- WITHOUT ATTENTION ----------------
    print("\nLoading model WITHOUT attention")

    model_without = build_model(
        model_name=args.model,
        num_classes=num_classes,
        use_attn=False,
        img_size=args.img_size,
        attn_layer="none",
    )

    criterion_without = build_criterion(
        num_classes=num_classes,
        feat_dim=model_without.feat_dim,
        args=args,
    )

    model_without, criterion_without = load_checkpoint(
        net=model_without,
        criterion=criterion_without,
        ckpt_path=checkpoint_without,
        device=device,
        model_name="WITHOUT-attention model",
    )

    model_without = model_without.to(device).eval()
    criterion_without = criterion_without.to(device).eval()

    # ---------------- WITH ATTENTION ----------------
    print("\nLoading model WITH attention")

    model_with = build_model(
        model_name=args.model,
        num_classes=num_classes,
        use_attn=True,
        img_size=args.img_size,
        attn_layer=selected_attention_arch,
    )

    criterion_with = build_criterion(
        num_classes=num_classes,
        feat_dim=model_with.feat_dim,
        args=args,
    )

    print_attention_debug(model_with, dataset_name)

    model_with, criterion_with = load_checkpoint(
        net=model_with,
        criterion=criterion_with,
        ckpt_path=checkpoint_with,
        device=device,
        model_name="WITH-attention model",
    )

    model_with = model_with.to(device).eval()
    criterion_with = criterion_with.to(device).eval()

    target_layers_without = [
        get_target_layer(model_without, layer_name)
        for layer_name in selected_layers_without
    ]

    target_layers_with = [
        get_target_layer(model_with, layer_name)
        for layer_name in selected_layers_with
    ]

    cam_without_extractor = DistanceGradCAM(
        model=model_without,
        criterion=criterion_without,
        target_layers=target_layers_without,
    )

    cam_with_extractor = DistanceGradCAM(
        model=model_with,
        criterion=criterion_with,
        target_layers=target_layers_with,
    )

    x_device, x_cpu, label = get_one_sample(
        data_loader=data.test_loader,
        device=device,
        sample_index=args.sample_index,
    )

    print(f"Selected sample index: {args.sample_index}")
    print(f"Sample label: {label}")

    x_without = x_device.clone().detach().requires_grad_(True)
    cam_without = cam_without_extractor(x_without)

    x_with = x_device.clone().detach().requires_grad_(True)
    cam_with = cam_with_extractor(x_with)

    cam_without_extractor.remove_hooks()
    cam_with_extractor.remove_hooks()

    return {
        "dataset": dataset_name,
        "display_name": DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name),
        "image": x_cpu,
        "cam_without": cam_without,
        "cam_with": cam_with,
        "selected_layers_without": "+".join(selected_layers_without),
        "selected_layers_with": "+".join(selected_layers_with),
        "attention_arch": selected_attention_arch,
        "label": label,
    }


# ==========================================================
# Save final figure
# ==========================================================
def save_multi_dataset_grid(results, out_path):
    n = len(results)

    fig, axes = plt.subplots(
        6,
        n,
        figsize=(3.0 * n, 10.2),
        gridspec_kw={
            "height_ratios": [1, 0.13, 1, 0.13, 1, 0.13]
        },
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, result in enumerate(results):
        image = result["image"]
        cam_without = result["cam_without"]
        cam_with = result["cam_with"]

        original = denormalize(image).permute(1, 2, 0).numpy()

        overlay_without = overlay_cam(image, cam_without)
        overlay_with = overlay_cam(image, cam_with)

        axes[0, i].imshow(original)
        axes[0, i].axis("off")
        axes[0, i].set_title(
            result["display_name"],
            fontsize=18,
            pad=14,
        )

        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].text(
                n * 0.52,
                0.5,
                "(a) Sample Images",
                ha="center",
                va="center",
                fontsize=18,
                transform=axes[1, i].transAxes,
            )

        axes[2, i].imshow(overlay_without)
        axes[2, i].axis("off")

        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].text(
                n * 0.52,
                0.5,
                "(b) Without Attention ",
                ha="center",
                va="center",
                fontsize=18,
                transform=axes[3, i].transAxes,
            )

        axes[4, i].imshow(overlay_with)
        axes[4, i].axis("off")

        axes[5, i].axis("off")
        if i == 0:
            axes[5, i].text(
                n * 0.52,
                0.5,
                "(c) With Attention",
                ha="center",
                va="center",
                fontsize=18,
                transform=axes[5, i].transAxes,
            )

    plt.subplots_adjust(
        left=0.02,
        right=0.995,
        top=0.94,
        bottom=0.03,
        wspace=0.03,
        hspace=0.04,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.savefig(
        out_path,
        dpi=900,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close()


# ==========================================================
# Main
# ==========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default="bloodmnist,dermamnist,asc,breakhis_40",
        help="Comma-separated dataset names.",
    )

    parser.add_argument(
        "--items",
        type=str,
        default="0,0,0,0",
        help="Comma-separated split ids corresponding to datasets.",
    )

    parser.add_argument(
        "--checkpoints-without-attn",
        type=str,
        required=True,
        help="Comma-separated WITHOUT-attention checkpoint paths.",
    )

    parser.add_argument(
        "--checkpoints-with-attn",
        type=str,
        required=True,
        help="Comma-separated WITH-attention checkpoint paths.",
    )

    parser.add_argument(
        "--attention-layers",
        type=str,
        default="bloodmnist:layer4,dermamnist:layer4,asc:layer4,breakhis_40:layer4",
        help=(
            "Dataset-wise WITH-attention model architecture. Example: "
            "bloodmnist:layer4,dermamnist:layer4,asc:layer4,breakhis_40:layer4"
        ),
    )

    parser.add_argument(
        "--gradcam-layers-without",
        type=str,
        default="bloodmnist:layer4,dermamnist:layer4,asc:layer4,breakhis_40:layer4",
        help=(
            "Dataset-wise Grad-CAM hook layers for DMDSC. Example: "
            "bloodmnist:layer4,dermamnist:layer4,asc:layer4,breakhis_40:layer4"
        ),
    )

    parser.add_argument(
        "--gradcam-layers-with",
        type=str,
        default="bloodmnist:self_attn,dermamnist:self_attn,asc:self_attn,breakhis_40:self_attn",
        help=(
            "Dataset-wise Grad-CAM hook layers for LA-DMDSC. Example: "
            "bloodmnist:self_attn,dermamnist:self_attn,asc:self_attn,breakhis_40:self_attn"
        ),
    )

    parser.add_argument("--dataroot", type=str, default="./data")
    parser.add_argument("--model", type=str, default="resnet50")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--sample-index", type=int, default=0)

    parser.add_argument("--m-min", type=float, default=35.0)
    parser.add_argument("--m-max", type=float, default=55.0)
    parser.add_argument("--Expand", type=int, default=100)

    parser.add_argument(
        "--save-path",
        type=str,
        default="./gradcam_multi_dataset_comparison.png",
    )

    args = parser.parse_args()

    gradcam_valid_layers = ["layer3", "layer4", "self_attn", "self_attn3", "self_attn4"]

    args.gradcam_layer_map_without = parse_layer_map(
        args.gradcam_layers_without,
        gradcam_valid_layers,
        "gradcam-layers-without",
    )

    args.gradcam_layer_map_with = parse_layer_map(
        args.gradcam_layers_with,
        gradcam_valid_layers,
        "gradcam-layers-with",
    )

    args.attention_layer_map = parse_attention_layers(args.attention_layers)

    dataset_list = parse_list_argument(args.datasets)
    item_list = [int(x) for x in parse_list_argument(args.items)]
    checkpoint_without_list = parse_list_argument(args.checkpoints_without_attn)
    checkpoint_with_list = parse_list_argument(args.checkpoints_with_attn)

    if not (
        len(dataset_list)
        == len(item_list)
        == len(checkpoint_without_list)
        == len(checkpoint_with_list)
    ):
        raise ValueError(
            "datasets, items, checkpoints-without-attn, and checkpoints-with-attn must have the same length."
        )

    for dataset_name in dataset_list:
        if dataset_name not in args.gradcam_layer_map_without:
            raise ValueError(
                f"DMDSC Grad-CAM layer choice missing for dataset '{dataset_name}'. "
                "Please add it in --gradcam-layers-without."
            )

        if dataset_name not in args.gradcam_layer_map_with:
            raise ValueError(
                f"LA-DMDSC Grad-CAM layer choice missing for dataset '{dataset_name}'. "
                "Please add it in --gradcam-layers-with."
            )

        if dataset_name not in args.attention_layer_map:
            raise ValueError(
                f"Attention architecture choice missing for dataset '{dataset_name}'. "
                "Please add it in --attention-layers."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nDataset-wise attention architecture setting:")
    for dataset_name in dataset_list:
        print(f"{dataset_name}: {args.attention_layer_map[dataset_name]}")

    print("\nDataset-wise DMDSC Grad-CAM hook layer setting:")
    for dataset_name in dataset_list:
        print(f"{dataset_name}: {args.gradcam_layer_map_without[dataset_name]}")

    print("\nDataset-wise LA-DMDSC Grad-CAM hook layer setting:")
    for dataset_name in dataset_list:
        print(f"{dataset_name}: {args.gradcam_layer_map_with[dataset_name]}")

    results = []

    for dataset_name, item, ckpt_wo, ckpt_w in zip(
        dataset_list,
        item_list,
        checkpoint_without_list,
        checkpoint_with_list,
    ):
        result = process_one_dataset(
            dataset_name=dataset_name,
            item=item,
            checkpoint_without=ckpt_wo,
            checkpoint_with=ckpt_w,
            args=args,
            device=device,
        )

        results.append(result)

    save_multi_dataset_grid(results, args.save_path)

    print(f"\nSaved multi-dataset Grad-CAM figure at: {args.save_path}")


if __name__ == "__main__":
    main()
