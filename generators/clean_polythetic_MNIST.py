import torch
import torchvision
import numpy as np


def transform_digit_color(img, color):
    if color == 'red':
        return img * np.array([1, 0, 0])
    elif color == 'green':
        return img * np.array([0, 1, 0])
    elif color == 'blue':
        return img * np.array([0, 0, 1])
    else:
        raise ValueError('Unknown color: ' + color)


def transform_quadrant(img, quadrant):
    if len(img.shape) == 4:
        n, w, h, c = img.shape
        new_imgs = np.zeros((n, 2 * w, 2 * h, c), dtype=np.uint8)
    elif len(img.shape) == 3:
        w, h, c = img.shape
        new_imgs = np.zeros((2 * w, 2 * h, c), dtype=np.uint8)
    else:
        raise ValueError('Invalid img shape: ' + str(img.shape))

    if quadrant == 1:
        new_imgs[..., :w, h:, :] = img
    elif quadrant == 2:
        new_imgs[..., :w, :h, :] = img
    elif quadrant == 3:
        new_imgs[..., w:, :h, :] = img
    elif quadrant == 4:
        new_imgs[..., w:, h:, :] = img
    else:
        raise ValueError('Invalid quadrant: ' + str(quadrant))

    return new_imgs


def transform_background(img, color):
    if color == 'white':
        pass  # img[img == 0] = 0
    elif color == 'black':
        img[img == np.array([0, 0, 0])] = 1
    else:
        raise ValueError('Unknown color: ' + color)
    return img


class task_generator:
    def __init__(self, config):
        self.examples_per_group = config.examples_per_group
        self.groups_per_class = config.groups_per_class
        self.examples_per_class = self.examples_per_group * self.groups_per_class

        # Load dataset
        path = config.path
        self.train_dataset = torchvision.datasets.MNIST(path,
                                                        train=True, download=True,
                                                        transform=None)
        self.x_val = self.train_dataset.data[50000:]
        self.x_train = self.train_dataset.data[:50000]
        self.y_val = self.train_dataset.targets[50000:]  # Digits
        self.y_train = self.train_dataset.targets[:50000]

        # Specify w and h of extended dataset
        w, h = self.x_train[0].shape
        self.input_shape = (2 * w, 2 * h, 3)  # Quadrant augmentations

        # Compute cartesian product of all digits, colors, and quadrants
        self.digits = np.arange(len(self.train_dataset.classes))
        self.colors = ['red', 'green', 'blue']
        self.quadrants = [1, 2, 3, 4]
        self.aux = [self.digits, self.digits, self.digits, self.digits, np.arange(len(self.colors)),
                    np.arange(len(self.colors)), np.arange(len(self.colors)), np.arange(len(self.colors))]

        train_class_bools = [self.y_train == c for c in self.digits]
        val_class_bools = [self.y_val == c for c in self.digits]
        self.train_digit_idx = [torch.arange(self.y_train.size(0))[b] for b in train_class_bools]
        self.val_digit_idx = [torch.arange(self.y_val.size(0))[b] for b in val_class_bools]

        self.nb_classes = 2

    def get_task(self, validation=False, xor_task=None, prob_xor=0.5):
        if validation:
            x = self.x_val
            y = self.y_val
            digit_idx = self.val_digit_idx
        else:
            x = self.x_train
            y = self.y_train
            digit_idx = self.train_digit_idx

        # Choose groups
        groups = np.zeros((2 * self.groups_per_class, len(self.aux)), dtype=int) - 1

        # Choose whether XOR task
        if xor_task is None:
            xor_task = np.random.rand() < prob_xor
        if xor_task:
            idxs = np.random.choice(len(self.aux), len(self.aux), replace=False)
            idx0 = idxs[0]
            c = np.random.choice(self.aux[idx0], self.groups_per_class, replace=False)
            groups[self.groups_per_class:, idx0] = c
            groups[:self.groups_per_class, idx0] = c

            idx1 = idxs[1]
            c = np.random.choice(self.aux[idx1], self.groups_per_class, replace=False)
            groups[self.groups_per_class:, idx1] = c
            groups[:self.groups_per_class, idx1] = c[::-1]
        else:
            idxs = np.random.choice(len(self.aux), len(self.aux), replace=False)
            idx0 = idxs[0]
            c1 = np.random.choice(self.aux[idx0], self.groups_per_class, replace=True)
            c2 = np.random.choice(list(set(self.aux[idx0]) - set(c1)), self.groups_per_class, replace=True)
            groups[self.groups_per_class:, idx0] = c1
            groups[:self.groups_per_class, idx0] = c2
            #  print(groups)

        # Eliminate noise
        for i in range(4):
            if groups[0, i] == -1 and groups[0, i + 4] == -1:
                groups[:, i] = -2
        # print(groups)

        # Append data for each group
        data = []
        for g in groups:
            selected_xs = [self.select_digits(x, digit_idx, g[i]) for i in range(4)]
            selected_xs = [self.select_colors(selected_xs[i], g[4 + i]) for i in range(4)]

            s = transform_quadrant(selected_xs[0], 1)
            for i in range(1, 4):
                s += transform_quadrant(selected_xs[i], i + 1)

            data.extend(s)

        data = np.stack(data) / 255.
        labels = [0] * self.examples_per_class + [1] * self.examples_per_class

        return data, labels, self.nb_classes

    def select_digits(self, x, digit_idx, g):
        if g == -2:
            return np.zeros((28, 28, 3))
        if g == -1:
            idxs = np.random.choice(x.shape[0], self.examples_per_group, replace=False)
        else:
            idxs = np.random.choice(digit_idx[g], self.examples_per_group, replace=False)
        selected_xs = x[idxs][..., None].numpy()
        return selected_xs

    def select_colors(self, selected_xs, g):
        if g == -1:
            colors = np.random.choice(3, self.examples_per_group, replace=True)
        else:
            colors = [g] * self.examples_per_group

        colors = np.eye(3)[colors]
        selected_xs = selected_xs * colors[:, None, None, :]  # transform_digit_color(selected_xs, self.colors[g])
        return selected_xs

    def get_shot_query(self, config, device, **task_kwargs):
        img_shape = (self.input_shape[2], self.input_shape[0], self.input_shape[1])
        data, label, _ = self.get_task(**task_kwargs)
        data = torch.Tensor(data).permute(0, 3, 1, 2).to(device)
        data_r = data.view((self.nb_classes, self.groups_per_class, self.examples_per_group) + img_shape)
        data_shot = data_r[:, :, :config.shot, ...].reshape((-1,) + img_shape)
        data_query = data_r[:, :, config.shot:, ...].reshape((-1,) + img_shape)
        label_shot = torch.arange(config.train_way).repeat(self.groups_per_class * config.shot).reshape(-1,
                                                                                                        config.train_way).permute(
            1, 0).reshape(-1).type(torch.float32).to(device)
        label_query = torch.arange(config.train_way).repeat(self.groups_per_class * config.query).reshape(-1,
                                                                                                          config.train_way).permute(
            1, 0).reshape(-1).type(torch.float32).to(device)
        return data_shot, label_shot, data_query, label_query

