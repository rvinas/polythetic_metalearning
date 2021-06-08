import torch
import torchvision.datasets
import random
import numpy as np
import pickle

"""
    Dataset handling.

    Add input sizes to the data_map when adding a new dataset.
"""

data_map = {
    'mnist': (1, 28, 28),
    'cifar-10': (3, 32, 32),
    'cifar-100': (3, 32, 32),
    'omniglot': (1, 28, 28)
}

class_map = {
    #  'dataset' : num train classes, num test classes
    'mnist': (10, 10),
    'cifar-10': (10, 10),
    'cifar-100': (100, 100),
    'omniglot': (3856, 659)
}


def get_input_size(config):
    if not 'input_size' in config.keys():
        try:
            input_size = data_map[config.dataset]
        except:
            raise ValueError(
                "No input_size provided and {config.dataset} is not supported.".format(config.dataset)
            )
    else:
        input_size = config.input_size

    return input_size


def get_dataset(name, train=True, download=True, root='./data'):
    if name == 'mnist':
        return torchvision.datasets.MNIST(root=root,
                                          download=download,
                                          train=train)
    elif name == 'cifar-10':
        return torchvision.datasets.CIFAR10(root=root,
                                            download=download,
                                            train=train)
    elif name == 'cifar-100':
        return torchvision.datasets.CIFAR100(root=root,
                                             download=download,
                                             train=train)
    elif name == 'omniglot':
        # regularOmniglot is a wrapper for the torchvision Omniglot dataset
        #  that behaves more typically than
        return regularOmniglot(root=root,
                               background=train,
                               download=download)

    else:
        raise ValueError(
            "Dataset {name} is not supported (get_datset step).".format(name)
        )


def data_transform(data, name, train):
    if name == 'mnist':
        return data.unsqueeze(1).float() / 255.
    elif name == 'cifar-10':
        return data.float() / 255.
    elif name == 'cifar-100':
        return data.float() / 255.
    elif name == 'omniglot':
        # use data augmentations during training
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(28),
                torchvision.transforms.RandomAffine(degrees=5,
                                                    translate=(0.1, 0.1),
                                                    scale=(0.9, 1.1),
                                                    fillcolor=None)
            ])
            # invert because RandomAffine only pads with 0s and the background is 1s
            data = data * -1 + 1
            data = transform(data)
            data = data * -1 + 1
            return data
            # return data.unsqueeze(1)
        # and do not augment during evaluation
        else:
            return data  # .unsqueeze(1)
    else:
        raise ValueError(
            "Dataset {name} is not supported (transform step).".format(name)
        )


class task_generator:
    """
    Generates task instances in the arbitrary-labelling, varying-number-of-
    -classes settings, drawn from a specified dataset.

    The complications here come pretty much exclusively from the different,
    fiddly ways that we might want to separate things.

    Most args are passed through the wandb.config for the experiment. Key are:
        classes_per_task      -    list, set of numbers of classes to draw from
        dataset               -    str, name of the dataset
    (note: merged classes will be relabelled in the order they are provided and
     treated as though they were base classes, and held_out and separate are then
     expected to refer to these merged groups. For example,
         train_merge = [[1,2,3],[4,5,6],[7,8,9]]
         held_out = [0]
     means to hold out the merged class [1,2,3])
    the others parameters are
        train                 -    bool, use train (or test) split of data
        download              -    bool, download the data if it isn't available
        root                  -    str, where to download to

    ----- For MNIST
    With [2,3,4] classes_per_task and 20 examples_per_class, %%timeit gives
        227 µs ± 9.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    and for [2,3,4,5,7] classes_per_task and 50 examples_per_class, it gave
        510 µs ± 30.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    on a mid-2015 MacBook pro, which seems fine? (1 µs = 1e-6 s)
    """

    def __init__(
            self,
            config,  # wandb config
            train=True,  # whether to use the train or test split
            download=True,  # whether to download the data if not available
            root=None,  # root directory to put data
    ):
        self.c_per_t = config.classes_per_task
        self.e_per_c = config.examples_per_class
        self.train = train

        self.name = config.dataset
        self.dataset = get_dataset(config.dataset, train, download, root)

        # check for what's in the config
        self.val_classes = None  # config.val_classes if 'val_classes' in config.keys() else None

        # first get things as they are presented in the original dataset
        # num_classes might depend on train/eval (e.g. Omniglot)
        num_classes = class_map[config.dataset][0] if train else class_map[config.dataset][1]
        classes = list(range(num_classes))
        # get the 'native' indices
        class_bools = [self.dataset.targets == c for c in classes]
        self.class_idx = [torch.arange(self.dataset.targets.size(0))[b] for b in class_bools]

        # get the counts
        self.class_counts = [c_idx.size(0) for c_idx in self.class_idx]

        self.train_classes = pickle.load(
            open('SPLITS_FOLDER/train_classes.pkl', 'rb'))
        self.val_classes = pickle.load(
            open('SPLITS_FOLDER/val_classes.pkl', 'rb'))
        self.test_classes = pickle.load(
            open('SPLITS_FOLDER//test_classes.pkl', 'rb'))
        self.held_out = self.val_classes + self.test_classes

        #
        self.input_shape = (28, 28, 1)

        # then remove/reduce class list as required:
        # if training, do held_out / separate stuff
        if train:
            # if holding out, remove those
            if self.held_out is not None:
                classes = list(set(classes) - set(self.held_out))
            # if a combo is not grouped, split by those
            class_splits = [classes]
        # if testing, use hold_out or the full set
        else:
            if self.held_out is not None:
                class_splits = [self.held_out]
            else:
                class_splits = [classes]
        self.class_splits = class_splits

        if len(self.class_splits[0]) < max(self.c_per_t):
            error_msg = (
                "Not enough classes ({classes}) to support the range of classes per task ({self.c_per_t})".format(
                    classes, self.c_per_t)
            )
            raise ValueError(error_msg)

    def get_task(self, validation=False, **task_kwargs):
        # select how many and which classes we're using
        if validation:
            split = self.val_classes
        else:
            split = random.sample(self.class_splits, 1)[0]
        num_classes = random.sample(self.c_per_t, 1)[0]
        classes_to_use = random.sample(split, num_classes)

        data = []
        labels = []

        for i, c in enumerate(classes_to_use):
            # draw idxs at random from the class without replacement
            sample_idx = torch.LongTensor(random.sample(range(self.class_counts[c]), self.e_per_c))
            idx = self.class_idx[c][sample_idx]
            # select that data
            data.append(self.dataset.data[idx])
            # generate an arbitrary relabelling
            labels.append(torch.LongTensor([i] * self.e_per_c))
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)

        # task and regime specific data transform
        data = data_transform(data, self.name, self.train)

        return data[..., None], labels, num_classes

    def split_context_target(self, x, y, nb_context_points=5, disjoint=False):
        """
        If disjoint, context is a subset of target set
        """
        example_shape = x.shape[1:]
        nb_examples = x.shape[0]
        nb_groups = nb_examples // self.e_per_c
        x_ = np.reshape(x, (nb_groups, self.e_per_c) + example_shape)
        y_ = np.reshape(y, (nb_groups, self.e_per_c))
        x_context = np.reshape(x_[:, :nb_context_points], (-1,) + example_shape)
        y_context = np.reshape(y_[:, :nb_context_points], (-1,))
        x_target = x
        y_target = y
        if disjoint:
            x_target = np.reshape(x_[:, nb_context_points:], (-1,) + example_shape)
            y_target = np.reshape(y_[:, nb_context_points:], (-1,))
        return x_context, y_context, x_target, y_target

    def get_shot_query(self, config, device, **task_kwargs):
        img_shape = (self.input_shape[2], self.input_shape[0], self.input_shape[1])
        data, label, num_classes = self.get_task(**task_kwargs)
        data = torch.Tensor(data).permute(0, 3, 1, 2).to(device)
        data_r = data.view((num_classes, config.shot + config.query) + img_shape)
        data_shot = data_r[:, :config.shot, ...].reshape((-1,) + (1, 28, 28))
        data_query = data_r[:, config.shot:, ...].reshape((-1,) + (1, 28, 28))
        label_shot = label.reshape(num_classes, -1)[:, :config.shot].reshape(-1).long().to(device)
        label_query = label.reshape(num_classes, -1)[:, config.shot:].reshape(-1).long().to(device)
        return data_shot, label_shot, data_query, label_query


class regularOmniglot(torch.utils.data.Dataset):
    """
    A (possibly dumb) way to wrap the base Omniglot dataset to get it to
    behave like the rest of the pytorch datasets, i.e. having the data
    and targets being attributes such that
        dataset.data[i] is the i-th data
        dataset.targets[i] is the i-th target

    For the training set, make new classes by rotating the base classes a
    quarter turn (as in CNPs, others also use 180 and 270 rotations.)
    """

    def __init__(
            self,
            root,
            download,
            background,
    ):
        base = torchvision.datasets.Omniglot(root=root,
                                             download=download,
                                             background=background)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor()
        ])
        data, targets = [], []
        for ex in base:
            data.append(transform(ex[0]))
            targets.append(ex[1])
            # if using the 'background' (ie training set) rotate 90 for new class
            #  (964 is the number of background classes)
            if background:
                data.append(torchvision.transforms.functional.rotate(transform(ex[0]), 90))
                data.append(torchvision.transforms.functional.rotate(transform(ex[0]), 180))
                data.append(torchvision.transforms.functional.rotate(transform(ex[0]), 270))
                targets.append(ex[1] + 964)
                targets.append(ex[1] + 964 * 2)
                targets.append(ex[1] + 964 * 3)
        self.data = torch.cat(data, 0)
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        return self.data[idx], self.targets[idx]