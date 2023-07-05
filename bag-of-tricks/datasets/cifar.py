import json
import os
import random
import sys
import pickle
import torch
import torchvision
import copy

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torchvision.datasets import cifar
from datasets.randaugment import RandAugmentMC
from torch.utils.data import Dataset, DataLoader
from datasets.augmentation import Augmentation, CutoutDefault
from datasets.augmentation_archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10
from utils.utils_datasets import check_integrity, download_url, noisify, noisify_instance, noisify_pairflip, noisify_multiclass_symmetric


class CIFAR10(Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    cls_num = 10

    def __init__(self, root='./data/', noise_path=None, random_seed=0, noise_type=None,
                 noise_rate=0.1, train=True, transform=None, target_transform=None, download=False):

        self.num_per_cls_dict = None
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'cifar10'
        self.noise_type = noise_type
        idx_each_class_noisy = [[] for i in range(self.cls_num)]
        self.noise_path = noise_path

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            if noise_type in ['clean', 'clean_label']:
                print("clean labels")
                self.train_noisy_labels = copy.deepcopy(self.train_labels)
                self.actual_noise_rate = 0.0
            elif noise_type == 'instance':
                self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data,
                                                                                    self.train_labels,
                                                                                    noise_rate=noise_rate,
                                                                                    random_state=random_seed)
                print('over all noise rate is ', self.actual_noise_rate)
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.cls_num)]
                self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                print(f'The instance noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
            # elif noise_type in ['pairflip', 'symmetric']: 
            elif noise_type in ['pairflip', 'symmetric', 'asymmetric']: 
                print(f"noise type is {noise_type}")
                # print("symmetric or pairflip ")
                self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                            train_labels=self.train_labels,
                                                                            noise_type=noise_type,
                                                                            noise_rate=noise_rate,
                                                                            random_state=random_seed,
                                                                            nb_classes=self.cls_num)
                print('over all noise rate is ', self.actual_noise_rate)
                
                # self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]

                _train_labels = [i[0] for i in self.train_labels]
                for i in range(len(_train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.cls_num)]
                self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
                
                self.train_labels.squeeze()
            elif noise_type in ['clean_label','worse_label','aggre_label','random_label1','random_label2'\
                                ,'random_label3','noisy_label']:
                self.train_noisy_labels = np.asarray(self.load_label().tolist())
                print(f'noisy labels loaded from {self.noise_path}')
                for i in range(len(self.train_noisy_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The human noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
                # self.actual_noise_rate = np.sum(self.noise_or_not)/50000
                # print('over all noise rate is ', self.actual_noise_rate)
            else:
                raise NotImplementedError("Only support symmetric noise & Asymmetric noise & human noise!")
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_label(self):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  
                print(f'Loaded {self.noise_type} from {self.noise_path}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')


    def __getitem__(self, index):
        if self.train:
            if self.noise_type != 'clean':
                img, target, trues = self.train_data[index], self.train_noisy_labels[index], self.train_labels[index]
            else:
                img, target, trues = self.train_data[index], self.train_labels[index], self.train_labels[index]
        else:
            img, target, trues = self.test_data[index], self.test_labels[index], self.test_labels[index]

        img = Image.fromarray(img)
        if self.target_transform is not None:
            img2 = self.target_transform(img)
            img = self.transform(img)
            return [img, img2], target, trues, index
        else:
            img = self.transform(img)
            return [img], target, trues, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

class CIFAR100(Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


    def __init__(self, root='./data/', noise_path=None, random_seed=0, noise_type=None,
                 noise_rate=0.1, train=True, transform=None, target_transform=None, download=False):

        self.num_per_cls_dict = None
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'cifar100'
        self.noise_type = noise_type
        idx_each_class_noisy = [[] for i in range(self.cls_num)]
        self.noise_path = noise_path

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            if noise_type in ['clean', 'clean_label']:
                print("clean labels")
                self.train_noisy_labels = copy.deepcopy(self.train_labels)
                self.actual_noise_rate = 0.0
            elif noise_type == 'instance':
                self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data,
                                                                                    self.train_labels,
                                                                                    noise_rate=noise_rate,
                                                                                    random_state=random_seed)
                print('over all noise rate is ', self.actual_noise_rate)
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.cls_num)]
                self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                print(f'The instance noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(self.train_labels)
            # elif noise_type in ['pairflip', 'symmetric']: 
            elif noise_type in ['pairflip', 'symmetric', 'asymmetric']: 
                print(f"noise type is {noise_type}")
                # print("symmetric or pairflip ")
                self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                            train_labels=self.train_labels,
                                                                            noise_type=noise_type,
                                                                            noise_rate=noise_rate,
                                                                            random_state=random_seed,
                                                                            nb_classes=self.cls_num)
                print('over all noise rate is ', self.actual_noise_rate)
                
                # self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]

                _train_labels = [i[0] for i in self.train_labels]
                for i in range(len(_train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.cls_num)]
                self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels) != np.transpose(_train_labels)
                
                self.train_labels.squeeze()
            elif noise_type in ['clean_label','worse_label','aggre_label','random_label1','random_label2'\
                                ,'random_label3','noisy_label']:
                self.train_noisy_labels = np.asarray(self.load_label().tolist())
                print(f'noisy labels loaded from {self.noise_path}')
                for i in range(len(self.train_noisy_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The human noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
                # self.actual_noise_rate = np.sum(self.noise_or_not)/50000
                # print('over all noise rate is ', self.actual_noise_rate)
            else:
                raise NotImplementedError("Only support clean&human noisy& symmetric noise!")
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_label(self):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  
                print(f'Loaded {self.noise_type} from {self.noise_path}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')


    def __getitem__(self, index):
        if self.train:
            if self.noise_type != 'clean':
                img, target, trues = self.train_data[index], self.train_noisy_labels[index], self.train_labels[index]
            else:
                img, target, trues = self.train_data[index], self.train_labels[index], self.train_labels[index]
        else:
            img, target, trues = self.test_data[index], self.test_labels[index], self.test_labels[index]

        img = Image.fromarray(img)
        if self.target_transform is not None:
            img2 = self.target_transform(img)
            img = self.transform(img)
            return [img, img2], target, trues, index
        else:
            img = self.transform(img)
            return [img], target, trues, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

def load_data(args, dataset='cifar10', noisy_type='symmetric', noise_rate=0.1, random_seed=0):
    train_cifar10_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_cifar100_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_cifar100_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if args.aug_type == 'autoaug':
        train_cifar10_aug = transforms.Compose([
            Augmentation(autoaug_paper_cifar10()),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            CutoutDefault(16)
        ])
        train_cifar100_aug = transforms.Compose([
            Augmentation(autoaug_paper_cifar10()),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            CutoutDefault(16)
        ])
    elif args.aug_type == 'randaug':
        train_cifar10_aug = transforms.Compose([
            RandAugmentMC(n=2, m=10),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])   
        train_cifar100_aug = transforms.Compose([
            RandAugmentMC(n=2, m=10),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        raise NotImplementedError("args.aug_type not in ['autoaug', 'randaug'].")

    if dataset == 'cifar10':
        root = args.data_path
        noise_path = os.path.join(root, 'CIFAR-10_human.pt')
        # noise_path = './data/CIFAR-10_human.pt'
        train_dataset = CIFAR10(root=root, noise_path=noise_path,
                                transform=train_cifar10_transform, target_transform=train_cifar10_aug, noise_type=noisy_type,
                                noise_rate=noise_rate, random_seed=random_seed, train=True, download=False)
        test_dataset = CIFAR10(root=root, train=False, transform=test_cifar10_transform, download=False)
        num_classes = 10
        num_examples = len(train_dataset)
    elif dataset == 'cifar100':
        root = args.data_path
        noise_path = os.path.join(root, 'CIFAR-100_human.pt')
        # noise_path = os.path.join(root, 'CIFAR-10
        # noise_path = './data/CIFAR-100_human.pt'
        train_dataset = CIFAR100(root=root, noise_path=noise_path,
                                 transform=train_cifar100_transform, target_transform=train_cifar100_aug, noise_type=noisy_type,
                                 noise_rate=noise_rate, random_seed=random_seed, train=True, download=True)
        test_dataset = CIFAR100(root=root, train=False, transform=test_cifar100_transform, download=False)
        num_classes = 100
        num_examples = len(train_dataset)
    else:
        raise NotImplementedError("ONLY SUPPORT CIFAR DATASET")
    return train_dataset, test_dataset, num_classes, num_examples