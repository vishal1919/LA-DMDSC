import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NirvanaOpenset_loss(nn.Module):
    """
    Adds dynamic margin adaptation based on class frequency.
    
    IMPORTANT: After initialization, call set_class_counts() with actual
    training data distribution BEFORE training to ensure proper margins from epoch 0.
        
        # Compute actual class distribution from training data
        train_labels = torch.cat([labels for _, labels in trainloader])
        class_counts = torch.bincount(train_labels, minlength=10)
        criterion.set_class_counts(class_counts)
        
        # Now margins are properly initialized before training starts
    """
    def __init__(
        self,
        num_classes=10,
        feat_dim=128,
        precalc_centers=None,
        m_min=38.0,
        m_max=71.0,
        Expand=100,
        outlier_weight=1.0,
        inter_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_centers = num_classes
        self.feat_dim = int(feat_dim)  # Ensure integer type
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.E = Expand
        self.outlier_weight = outlier_weight
        self.inter_weight = inter_weight

        if precalc_centers:
            precalculated_centers = FindCenters(self.feat_dim, self.E)[:self.num_classes, :]
        with torch.no_grad():
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim, requires_grad=False)
            )
            if precalc_centers:
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print("Centers loaded.")

        self.register_buffer("class_counts", torch.zeros(self.num_classes, dtype=torch.float32))
        self.register_buffer("m_per_class", torch.full((self.num_classes,), float(self.m_min)))
        self.register_buffer("_counts_initialized", torch.tensor(False))

    def set_class_counts(self, counts: torch.Tensor):
        """
        Set absolute counts per class (length = num_classes).
        Call this once from your train dataset histogram for exact DM margins.
        
        This should be called BEFORE training starts to ensure proper margin
        initialization from epoch 0.
        """
        counts = counts.to(dtype=torch.float32, device=self.class_counts.device)
        counts = torch.clamp(counts, min=0.0)
        self.class_counts.copy_(counts)
        self._counts_initialized.fill_(True)
        self.compute_margins()
        print(f"[NirvanaOpenset_loss] Class counts updated: {counts.tolist()}")
        print(f"[NirvanaOpenset_loss] Dynamic margins: {self.m_per_class.tolist()}")
        print(f"[NirvanaOpenset_loss] Margin range: [{self.m_per_class.min():.2f}, {self.m_per_class.max():.2f}]")

    def update_counts_online(self, labels: torch.Tensor):
        """
        Optional: online accumulation during training.
        """
        binc = torch.bincount(labels, minlength=self.num_classes).to(self.class_counts)
        self.class_counts.add_(binc)
        self.compute_margins()

    def compute_margins(self):
        N = torch.sum(self.class_counts)
        if N <= 0:
            self.m_per_class.fill_(self.m_min)
            return
        frac = self.class_counts / N
        self.m_per_class.copy_(self.m_min + (self.m_max - self.m_min) * (1.0 - frac))
        
    def forward(self, labels, x, x_out, ramp=False, update_counts=True):
        """
        x: (B, D) inlier features
        labels: (B,)
        x_out: (B_out, D) or None
        update_counts: if True, accumulates counts online each step
        """
        # Warn if class counts were never properly initialized
        if not self._counts_initialized.item() and self.class_counts.sum() == 0:
            import warnings
            warnings.warn(
                "Dynamic margins are NOT initialized! "
                "Call criterion.set_class_counts(counts) with actual training data distribution "
                "BEFORE training for proper margin adaptation. "
                "Currently using uniform margins for all classes.",
                UserWarning
            )
            self._counts_initialized.fill_(True)
        
        if labels.max() >= self.num_classes or labels.min() < 0:
            raise ValueError("Labels out of valid range for NirvanaOpenset_loss.")

        if update_counts:
            self.update_counts_online(labels.detach())

        dtype, device = x.dtype, x.device
        centers = self.centers.to(dtype=dtype, device=device)
        B = x.size(0)

        # ----- distances (inlier) -----
        x2 = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(B, self.num_classes)
        c2 = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, B).t()
        dist_in = (x2 + c2).to(dtype=dtype)
        dist_in.addmm_(x, centers.t(), beta=1, alpha=-2)  # [B, C]

        # intra-class
        intra = dist_in.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        intraclass_loss = intra.sum() / (B * self.feat_dim * 2.0)
        
        m_batch = self.m_per_class[labels].to(dtype=dtype, device=device)  # [B]

        # inter-class
        centers_dist_inter = intra.unsqueeze(1) - dist_in  # [B, C]
        mask = ~F.one_hot(labels.long(), num_classes=self.num_classes).bool()
        interclass_loss_triplet = (
            ((m_batch.unsqueeze(1) + centers_dist_inter).clamp(min=0.0) * mask).sum()
            / (self.num_centers * B * 2.0)
        )
        interclass_loss_triplet = self.inter_weight * interclass_loss_triplet

        # ----- outlier exposure 
        if x_out is not None:
            x_out = x_out.to(dtype=dtype, device=device)
            B_out = x_out.size(0)

            xo2 = torch.pow(x_out, 2).sum(dim=1, keepdim=True).expand(B_out, self.num_classes)
            c2o = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, B_out).t()
            dist_out = (xo2 + c2o).to(dtype=dtype)
            dist_out.addmm_(x_out, centers.t(), beta=1, alpha=-2)  # [B_out, C]

            # pair each outlier row with inlier labels' columns
            outlier_cols = dist_out.index_select(1, labels.long())  # [B_out, B]
            
            m_pair = m_batch.unsqueeze(0).expand(B_out, B)

            if ramp:
                hinge = (m_pair + (intra.unsqueeze(0) - outlier_cols)).clamp(min=0.0).clamp(max=60.0)
            else:
                hinge = (m_pair + (intra.unsqueeze(0) - outlier_cols)).clamp(min=0.0)

            outlier_triplet_multi_loss = hinge.sum() / (B * B_out * 2.0)
            weighted_outlier_loss = self.outlier_weight * outlier_triplet_multi_loss

            return intraclass_loss, interclass_loss_triplet, weighted_outlier_loss
        else:
            return intraclass_loss, interclass_loss_triplet, None

def FindCenters(k, E=1):
    """
    Calculates "k+1" equidistant points in R^{k}.
    Args:
        k (int) dimension of the space
        E (float) expand factor
    Returns:
        Centers (np.array) equidistant positions in R^{k}, shape (k+1 x k)
    """

    Centers = np.empty((k+1, k), dtype=np.float32)
    CC = np.empty((k,k), dtype=np.float32)
    Unit_Vector = np.identity(k)
    c = -((1+np.sqrt(k+1))/np.power(k, 3/2))
    CC.fill(c)
    d = np.sqrt((k+1)/k)
    DU = d*Unit_Vector
    Centers[0,:].fill(1/np.sqrt(k))
    Centers[1:,:] = CC + DU

    # Calculate and Check Distances
    Distances = np.empty((k+1,k), dtype=np.float32)
    for k, rows in enumerate(Centers):
        Distances[k,:] = np.linalg.norm(rows - np.delete(Centers, k, axis=0), axis=1)
    # print("Distances:",Distances)
    assert np.allclose(np.random.choice(Distances.flatten(), size=1), Distances, rtol=1e-05, atol=1e-08, equal_nan=False), "Distances are not equal"
    return Centers*E




def get_l2_pred(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes

        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits = 1/(1+distmat)
            return pred, logits
        else:
            return pred

def get_l2_pred_b9(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes

        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits_b9 = 1/(1+F.normalize(distmat,p=2))
            logits = 1/(1+distmat)
            return pred, logits, logits_b9
        else:
            return pred

def accuracy_l2(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes

        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)

        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size)


def accuracy_l2_nosubcenter(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes

        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        distmat = torch.cdist(features, serialized_centers, p=2)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)
        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size)

def get_l2_pred_nosubcenter(features,centers, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes

        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        distmat = torch.cdist(features, serialized_centers, p=2)
        pred = distmat.argmin(1)

        return pred

def cosine_similarity(features, centers, target):
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        pred = torch.empty(batch_size, device=features.device)
        for i in range(batch_size):
            pred[i] = nn.functional.cosine_similarity(features[i].reshape(1,-1), serialized_centers).argmax()
    return pred

def euc_cos(features,centers, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes

        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        disteuc = torch.cdist(features, serialized_centers, p=2)
        # pred = distmat.argmin(1)
        distcos = torch.empty(batch_size, num_classes, device=features.device)
        for i in range(batch_size):
            distcos[i] = nn.functional.cosine_similarity(features[i].reshape(1,-1), serialized_centers)
    return ((1/(2+distcos))*disteuc).argmin(1)
