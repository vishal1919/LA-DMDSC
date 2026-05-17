import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
import numpy as np


class SimpleBackMix(torch.nn.Module):
    """
    Simple BackMix-style batch augmentation.
    Applies patch replacement BEFORE forward/loss.

    Notes:
    - Keeps original labels unchanged.
    - Intended for training only.
    - In OE training, use only on in-distribution (known) samples.
    """

    def __init__(self, p=0.5, min_ratio=0.15, max_ratio=0.25, same_class_only=False):
        super().__init__()
        self.p = float(p)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.same_class_only = bool(same_class_only)

        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {self.p}")
        if not (0.0 < self.min_ratio <= 1.0):
            raise ValueError(f"min_ratio must be in (0,1], got {self.min_ratio}")
        if not (0.0 < self.max_ratio <= 1.0):
            raise ValueError(f"max_ratio must be in (0,1], got {self.max_ratio}")
        if self.min_ratio > self.max_ratio:
            raise ValueError(
                f"min_ratio ({self.min_ratio}) must be <= max_ratio ({self.max_ratio})"
            )

    @torch.no_grad()
    def forward(self, images, labels):
        """
        Args:
            images: Tensor [B, C, H, W]
            labels: Tensor [B]

        Returns:
            mixed_images, labels
        """
        if (not self.training) or (random.random() > self.p):
            return images, labels

        if images is None or labels is None:
            return images, labels

        if images.dim() != 4:
            return images, labels

        B, C, H, W = images.shape
        if B < 2:
            return images, labels

        device = images.device
        mixed = images.clone()

        if self.same_class_only:
            donor_idx = []
            labels_cpu = labels.detach().cpu()
            for i in range(B):
                same_cls = torch.where(labels_cpu == labels_cpu[i])[0].tolist()
                same_cls = [x for x in same_cls if x != i]
                donor_idx.append(random.choice(same_cls) if len(same_cls) > 0 else i)
            donor_idx = torch.tensor(donor_idx, dtype=torch.long, device=device)
        else:
            donor_idx = torch.randperm(B, device=device)

        donor = images[donor_idx]

        for i in range(B):
            ratio = random.uniform(self.min_ratio, self.max_ratio)
            patch_h = max(1, int(H * ratio))
            patch_w = max(1, int(W * ratio))

            if patch_h >= H:
                patch_h = H - 1
            if patch_w >= W:
                patch_w = W - 1
            if patch_h <= 0 or patch_w <= 0:
                continue

            y1 = random.randint(0, H - patch_h)
            x1 = random.randint(0, W - patch_w)

            dy1 = random.randint(0, H - patch_h)
            dx1 = random.randint(0, W - patch_w)

            mixed[i, :, y1:y1 + patch_h, x1:x1 + patch_w] = \
                donor[i, :, dy1:dy1 + patch_h, dx1:dx1 + patch_w]

        return mixed, labels


def _build_backmix_from_options(options):
    """
    Builds the augmentation module from options dict.
    Safe defaults are used if keys are missing.
    """
    use_backmix = options.get('use_backmix', False)
    if not use_backmix:
        return None

    aug = SimpleBackMix(
        p=options.get('backmix_p', 0.5),
        min_ratio=options.get('backmix_min_ratio', 0.15),
        max_ratio=options.get('backmix_max_ratio', 0.25),
        same_class_only=options.get('backmix_same_class_only', False),
    )

    return aug


def _maybe_apply_backmix(backmix_aug, images, labels):
    if backmix_aug is None:
        return images, labels
    backmix_aug.train()
    return backmix_aug(images, labels)


def train_Nirvana_oe(net, criterion, optimizer, scheduler, trainloader, trainloader_oe, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0

    backmix_aug = _build_backmix_from_options(options)

    trainloader_oe.dataset.offset = np.random.randint(len(trainloader_oe.dataset))

    for batch_idx, (in_set, out_set) in enumerate(zip(trainloader, trainloader_oe)):
        inputs_in = in_set[0]
        inputs_oe = out_set[0]
        targets_in = in_set[1]

        if options['use_gpu']:
            inputs_in = inputs_in.cuda(non_blocking=True)
            inputs_oe = inputs_oe.cuda(non_blocking=True)
            targets_in = targets_in.cuda(non_blocking=True)

        # Apply BackMix only on known in-distribution samples before forward/loss
        inputs_in, targets_in = _maybe_apply_backmix(backmix_aug, inputs_in, targets_in)

        inputs_inout = torch.cat((inputs_in, inputs_oe), 0)

        optimizer.zero_grad()

        x, y = net(inputs_inout, True)
        intraclass_loss, triplet_loss, outlier_triplet_loss = criterion(
            targets_in,
            x[:len(inputs_in)],
            x[len(inputs_in):],
            ramp=options['ramp_activate']
        )

        total_loss = intraclass_loss + triplet_loss + outlier_triplet_loss
        total_loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.update(total_loss.item(), targets_in.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print(
                "LR: {} - Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                    optimizer.param_groups[0]["lr"],
                    batch_idx + 1,
                    len(trainloader),
                    losses.val,
                    losses.avg
                )
            )

        loss_all += losses.avg

    print("Epoch loss: {}".format(loss_all))
    return loss_all


def train_Nirvana_oe_reg(net, criterion, optimizer, scheduler, trainloader, trainloader_oe, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0

    backmix_aug = _build_backmix_from_options(options)

    trainloader_oe.dataset.offset = np.random.randint(len(trainloader_oe.dataset))

    for batch_idx, (in_set, out_set) in enumerate(zip(trainloader, trainloader_oe)):
        inputs_in = in_set[0]
        inputs_oe = out_set[0]
        targets_in = in_set[1]

        if options['use_gpu']:
            inputs_in = inputs_in.cuda(non_blocking=True)
            inputs_oe = inputs_oe.cuda(non_blocking=True)
            targets_in = targets_in.cuda(non_blocking=True)

        # Apply BackMix only on known in-distribution samples before forward/loss
        inputs_in, targets_in = _maybe_apply_backmix(backmix_aug, inputs_in, targets_in)

        inputs_inout = torch.cat((inputs_in, inputs_oe), 0)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            x, y = net(inputs_inout, True)
            intraclass_loss, triplet_loss, outlier_triplet_loss = criterion(
                targets_in,
                x[:len(inputs_in)],
                x[len(inputs_in):],
                ramp=options['ramp_activate'],
                update_counts=True,
            )

            total_loss = intraclass_loss + triplet_loss + outlier_triplet_loss

            if options.get('l1_weight', 0) > 0:
                l1_penalty = sum(p.abs().sum() for p in net.parameters())
                total_loss = total_loss + options['l1_weight'] * l1_penalty

            total_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        losses.update(total_loss.item(), targets_in.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print(
                "LR: {} - Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                    optimizer.param_groups[0]["lr"],
                    batch_idx + 1,
                    len(trainloader),
                    losses.val,
                    losses.avg
                )
            )
            with torch.no_grad():
                head = min(5, criterion.num_classes)
                print("m_per_class (head):", criterion.m_per_class[:head].detach().cpu().numpy())
                print("class_counts (head):", criterion.class_counts[:head].detach().cpu().numpy())

        loss_all += losses.avg

    print("Epoch loss: {}".format(loss_all))
    return loss_all


def train_ddfm_oe(net, criterion, optimizer, optimizer_center, scheduler, trainloader, trainloader_oe, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0

    backmix_aug = _build_backmix_from_options(options)

    trainloader_oe.dataset.offset = np.random.randint(len(trainloader_oe.dataset))

    for batch_idx, (in_set, out_set) in enumerate(zip(trainloader, trainloader_oe)):
        inputs_in = in_set[0]
        inputs_oe = out_set[0]
        targets_in = in_set[1]

        if options['use_gpu']:
            inputs_in = inputs_in.cuda(non_blocking=True)
            inputs_oe = inputs_oe.cuda(non_blocking=True)
            targets_in = targets_in.cuda(non_blocking=True)

        # Apply BackMix only on known in-distribution samples before forward/loss
        inputs_in, targets_in = _maybe_apply_backmix(backmix_aug, inputs_in, targets_in)

        inputs_inout = torch.cat((inputs_in, inputs_oe), 0)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            x, y = net(inputs_inout, True)
            intraclass_loss, triplet_loss, outlier_triplet_loss = criterion(
                targets_in,
                x[:len(inputs_in)],
                x[len(inputs_in):],
                ramp=options['ramp_activate']
            )

            total_loss = intraclass_loss + triplet_loss + outlier_triplet_loss
            total_loss.backward()

            optimizer.step()
            optimizer_center.step()

            if scheduler is not None:
                scheduler.step()

        losses.update(total_loss.item(), targets_in.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print(
                "LR: {} - Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                    optimizer.param_groups[0]["lr"],
                    batch_idx + 1,
                    len(trainloader),
                    losses.val,
                    losses.avg
                )
            )

        loss_all += losses.avg

    print("Epoch loss: {}".format(loss_all))
    return loss_all


def train(net, criterion, optimizer, scheduler, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0

    backmix_aug = _build_backmix_from_options(options)

    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # Apply BackMix before forward/loss
        data, labels = _maybe_apply_backmix(backmix_aug, data, labels)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            x, y = net(data, True)
            loss = criterion(y, labels)

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print(
                "Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                    batch_idx + 1,
                    len(trainloader),
                    losses.val,
                    losses.avg
                )
            )

        loss_all += losses.avg

    print("Epoch loss: {}".format(loss_all))
    return loss_all
