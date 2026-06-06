import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sys
import os

medmnist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/MedMNIST')
sys.path.insert(0, medmnist_path)
from medmnist.dataset import BloodMNIST, DermaMNIST
from medmnist import BloodMNIST, DermaMNIST

class Random300K_Images(torch.utils.data.Dataset):
    def __init__(self, file_path='./data', transform=None, extendable=0):
        self.transform = transform
        self.extendable = extendable
        self.offset = 0
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = np.load(file_path)
        if extendable > 0:
            self.data = np.repeat(self.data, extendable + 1, axis=0)
            
        if transform is None:
            # Default to 224x224 to match NirvanaOSR 224x224 runs
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __getitem__(self, index):
        index = (index + self.offset) % len(self)
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    
    def __len__(self):
        return len(self.data)


class FilteredDataset(Dataset):
    """Helper class to filter and remap labels"""
    def __init__(self, dataset, mask, known_classes):
        self.dataset = dataset
        self.indices = np.where(mask)[0]
        self.target_map = {label: idx for idx, label in enumerate(known_classes)}
        
    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        return img, self.target_map[int(label)]

    def __len__(self):
        return len(self.indices)

class BloodMNIST_OSR(object):
    """Open Set Recognition wrapper for BloodMNIST dataset"""
    def __init__(self, known, dataroot='./data', use_gpu=True,
                 num_workers=12, batch_size=128, image_size=224):
        self.num_classes = len(known)
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
            
        self.unknown = list(set(range(0, 8)) - set(self.known))
        self.image_size = image_size

        print('BloodMNIST_OSR Known classes:', self.known)
        print('BloodMNIST_OSR Unknown classes:', self.unknown)
        print(f'BloodMNIST_OSR using native resolution size={self.image_size}')

        # Define transforms
        target_size = (self.image_size, self.image_size)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomCrop(self.image_size, padding=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load dataset
        train_dataset = BloodMNIST(
            split='train', transform=transform,
            download=False, root=dataroot,
            size=self.image_size
        )
        test_dataset = BloodMNIST(
            split='test', transform=test_transform,
            download=False, root=dataroot,
            size=self.image_size
        )

        train_labels = train_dataset.labels.squeeze()
        print("BloodMNIST_OSR Training set class distribution:", np.bincount(train_labels))

        # Filter dataset
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        # Create data loaders
        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known), 
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'BloodMNIST_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'BloodMNIST_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'BloodMNIST_OSR Test samples (unknown): {len(self.out_loader.dataset)}')


class DermaMNIST_OSR(object):
    """Open Set Recognition wrapper for DermaMNIST dataset"""
    def __init__(self, known,unknown = None, dataroot='./data', use_gpu=True, num_workers=4, batch_size=128, image_size=224):
        self.num_classes = len(known)
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
            
        if unknown is not None:
            self.unknown = unknown
        else:
            self.unknown = list(set(range(0, 7)) - set(self.known))
        self.image_size = image_size

        print('DermaMNIST_OSR Known classes:', self.known)
        print('DermaMNIST_OSR Unknown classes:', self.unknown)
        print(f'DermaMNIST_OSR using native resolution size={self.image_size}')

        # Use 224x224 for Derma to match other 224 runs
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomCrop(self.image_size, padding=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load dataset
        train_dataset = DermaMNIST(split='train', transform=transform, download=True, root=dataroot, size=self.image_size)
        test_dataset = DermaMNIST(split='test', transform=test_transform, download=True, root=dataroot, size=self.image_size)

        train_labels = train_dataset.labels.squeeze()
        print("DermaMNIST_OSR Training set class distribution:", np.bincount(train_labels))

        # Filter dataset
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        # Create data loaders
        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known), 
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'DermaMNIST_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'DermaMNIST_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'DermaMNIST_OSR Test samples (unknown): {len(self.out_loader.dataset)}')
    
class ASC_OSR(object):
    """Open Set Recognition wrapper for Augmented Skin Conditions dataset"""
    def __init__(self, known, unknown=None, dataroot='./data', use_gpu=True, num_workers=4, batch_size=128,
                 imbalance_ratio=None, random_state=42):
        """
        Parameters:
        -----------
        known : list
            List of known class indices
        unknown : list, optional
            List of unknown class indices
        dataroot : str
            Path to data directory
        use_gpu : bool
            Whether to use GPU
        num_workers : int
            Number of workers for DataLoader
        batch_size : int
            Batch size for DataLoader
        imbalance_ratio : int or None
            None for balanced (default)
            2 for very mild imbalance (IR=2)
            5 for mild imbalance (IR=5)
            10 for mild imbalance (IR=10)
            50 for moderate imbalance (IR=50)
            100 for severe imbalance (IR=100)
        random_state : int
            Random seed for reproducible imbalance creation
        """
        self.num_classes = len(known)
        self.imbalance_ratio = imbalance_ratio
        self.random_state = random_state
        
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
            
        if unknown is not None:
            self.unknown = unknown
        else:
            self.unknown = list(set(range(0, 6)) - set(self.known))  # Fixed: Changed from 8 to 6 classes

        print('ASC_OSR Known classes:', self.known)
        print('ASC_OSR Unknown classes:', self.unknown)
        
        if imbalance_ratio is not None:
            print(f'ASC_OSR Imbalance mode: IR={imbalance_ratio} (seed={random_state})')

        # Define transforms for 224x224 images
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset_path = os.path.join(dataroot, 'skin_conditions_dataset.npz')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Skin conditions dataset not found at {dataset_path}")
        
        data = np.load(dataset_path)
        
        # Apply imbalance to training data if specified
        train_images = data['train_images']
        train_labels = data['train_labels']
        
        if imbalance_ratio is not None:
            print(f"\n🔄 Applying IR={imbalance_ratio} imbalance to ASC training set...")
            print_imbalance_summary(train_labels, dataset_name="ASC Train (Original - Balanced)")
            
            # Get imbalanced indices
            imbalanced_indices = create_imbalanced_indices(
                train_labels,
                imbalance_ratio=imbalance_ratio,
                random_state=random_state
            )
            
            # Apply to training data
            train_images = train_images[imbalanced_indices]
            train_labels = train_labels[imbalanced_indices]
            
            print_imbalance_summary(train_labels, dataset_name=f"ASC Train (IR={imbalance_ratio})")
        
        # Create custom dataset classes
        train_dataset = ASCLoader(
            train_images, train_labels, transform=transform
        )
        test_dataset = ASCLoader(
            data['test_images'], data['test_labels'], transform=test_transform
        )

        train_labels_squeeze = train_dataset.labels.squeeze()
        
        if imbalance_ratio is None:
            print("ASC_OSR Training set class distribution (Balanced):", np.bincount(train_labels_squeeze))
        else:
            print(f"ASC_OSR Training set class distribution (IR={imbalance_ratio}):", np.bincount(train_labels_squeeze))

        # Filter datasets
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        # Create data loaders
        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known), 
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'ASC_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'ASC_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'ASC_OSR Test samples (unknown): {len(self.out_loader.dataset)}')


class ASCLoader(Dataset):
    """Custom dataset loader for .npz format"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, int(label)
    
    def __len__(self):
        return len(self.images)

class breakhis_OSR(object):
    def __init__(self, known, unknown=None, dataroot='./data', use_gpu=True, num_workers=4, batch_size=128):
        self.num_classes = len(known)
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
            
        if unknown is not None:
            self.unknown = unknown
        else:
            self.unknown = list(set(range(0, 8)) - set(self.known))  # Fixed: Changed from 8 to 6 classes

        print('breakhis_OSR Known classes:', self.known)
        print('breakhis_OSR Unknown classes:', self.unknown)

        # Define transforms for 224x224 images
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset_path = os.path.join(dataroot, 'breakhis_40.npz')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"breakhis dataset not found at {dataset_path}")
        
        data = np.load(dataset_path)
        
        # Create custom dataset classes
        train_dataset = breakhis_40Loader(
            data['train_images'], data['train_labels'], transform=transform
        )
        test_dataset = breakhis_40Loader(
            data['test_images'], data['test_labels'], transform=test_transform
        )

        train_labels = train_dataset.labels.squeeze()
        print("breakhis_OSR Training set class distribution:", np.bincount(train_labels))

        # Filter datasets
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        # Create data loaders
        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known), 
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'breakhis_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'breakhis_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'breakhis_OSR Test samples (unknown): {len(self.out_loader.dataset)}')

class breakhis_40Loader(Dataset):
    """Custom dataset loader for .npz format"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, int(label)
    
    def __len__(self):
        return len(self.images)
