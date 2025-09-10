from __future__ import print_function, division
import torch
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import sys
import pickle
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# %%
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']


def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes)
    return '%s%s' % (f, suffixes[i])


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return ((nb_samples + (-nb_samples % batch_size)) / batch_size)  # roundup division
    else:
        return nb_samples / batch_size


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))
    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size: (i + 1) * batch_size]


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


class Datafeed(data.Dataset):

    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)

class DatafeedImage(data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


### functions for BNN with gauss output: ###

def diagonal_gauss_loglike(x, mu, sigma):
    # note that we can just treat each dim as isotropic and then do sum
    cte_term = -(0.5)*np.log(2*np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu)/sigma
    dist_term = -(0.5)*(inner**2)
    log_px = (cte_term + det_sig_term + dist_term).sum(dim=1, keepdim=False)
    return log_px

def get_rms(mu, y, y_means, y_stds):
    x_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    return torch.sqrt(((x_un - y_un)**2).sum() / y.shape[0])


def get_loglike(mu, sigma, y, y_means, y_stds):
    mu_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    sigma_un = sigma * y_stds
    ll = diagonal_gauss_loglike(y_un, mu_un, sigma_un)
    return ll.mean(dim=0)


# %%
class DataGenerator:
    def __init__(self, x, y, batch_size, norm=False, rng_seed=1234):
        self.x = x
        self.y = y

        self.N = x.shape[0]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(rng_seed)
        self.Nbatch = math.ceil(self.N / self.batch_size)
        self.NORM = norm

        # self.x_mean = np.mean(x, axis=0)
        # self.x_std = np.std(x, axis=0)
        # self.y_mean = np.mean(y, axis=0)
        # self.y_std = np.std(y, axis=0)

        # self.x_norm = (x - self.x_mean) / self.x_std
        # self.y_norm = (y - self.y_mean) / self.y_std

        self.x_min = np.min(x, axis=0)
        self.x_max = np.max(x, axis=0)
        self.y_min = np.min(y, axis=0)
        self.y_max = np.max(y, axis=0)

        self.x_norm = (x - self.x_min) / (self.x_max - self.x_min) * 2 - 1
        self.y_norm = (y - self.y_min) / (self.y_max - self.y_min) * 2 - 1

        print("dataset size:", self.N)
        print("batch size:", self.batch_size)
        print("number of batches:", self.Nbatch)

    def __iter__(self):
        self.current_pos = 0
        self.indices = np.arange(self.N)
        self.rng.shuffle(self.indices)  # Shuffle at the beginning of each epoch
        return self

    def __next__(self):
        # print("current_pos:", self.current_pos)
        if self.current_pos >= self.N:
            raise StopIteration

        end_pos = min(self.current_pos + self.batch_size, self.N)
        batch_indices = self.indices[self.current_pos:end_pos]
        self.current_pos = end_pos

        if self.NORM:
            x_batch = self.x_norm[batch_indices]
            y_batch = self.y_norm[batch_indices]
        else:
            x_batch = self.x[batch_indices]
            y_batch = self.y[batch_indices]

        return x_batch, y_batch

    def __len__(self):
        return self.Nbatch
    
    def normalize(self, x=None, y=None):
        if self.NORM:
            if x is None:
                # return (y - self.y_mean) / self.y_std
                return (y-self.y_min)/(self.y_max-self.y_min)*2-1
            if y is None:
                # return (x - self.x_mean) / self.x_std
                return (x-self.x_min)/(self.x_max-self.x_min)*2-1
            else:
                # return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std
                return (x-self.x_min)/(self.x_max-self.x_min)*2-1, (y-self.y_min)/(self.y_max-self.y_min)*2-1
        else:
            if x is None:
                return y
            if y is None:
                return x
            else:
                return x, y
    
    def denormalize(self, x=None, y=None):
        if self.NORM:
            if x is None:
                # return y * self.y_std + self.y_mean
                return (y+1)/2*(self.y_max-self.y_min)+self.y_min
            if y is None:
                # return x * self.x_std + self.x_mean
                return (x+1)/2*(self.x_max-self.x_min)+self.x_min
            else:
                # return x * self.x_std + self.x_mean, y * self.y_std + self.y_mean
                return (x+1)/2*(self.x_max-self.x_min)+self.x_min, (y+1)/2*(self.y_max-self.y_min)+self.y_min
        else:
            if x is None:
                return y
            if y is None:
                return x
            else:
                return x, y


def plot_xy(x, y, title, xlabel, ylabel, save_path, y_log=True):
    plt.figure()
    if y_log:
        plt.semilogy(x, y, 'b-', linewidth=2)
    else:
        plt.plot(x, y, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_bayes_pred(x_train, y_train, x_real, y_real, x_test, y_pred, epistemic, total_unc, xlabel, ylabel, title, save_path, invert_yaxis=False, y_lim=[-1.5, 1]):
    x_test = x_test.squeeze()
    y_pred = y_pred.squeeze()
    epistemic = epistemic.squeeze()
    total_unc = total_unc.squeeze()
    y_train = y_train.squeeze()
    x_train = x_train.squeeze()
    x_real = x_real.squeeze()
    y_real = y_real.squeeze()
    label = True

    def split_monotonic_segments(x, y1, y2, y3):
        diff = np.diff(x)
        sign_diff = np.sign(diff)

        segments = []
        start_idx = 0

        for i in range(1, len(sign_diff)):
            if sign_diff[i] != sign_diff[i-1]:
                segments.append((x[start_idx:i+1], y1[start_idx:i+1], y2[start_idx:i+1], y3[start_idx:i+1]))
                start_idx = i

        segments.append((x[start_idx:], y1[start_idx:], y2[start_idx:], y3[start_idx:]))

        return segments
    
    segments = split_monotonic_segments(x_test, y_pred, epistemic, total_unc)


    plt.figure()
    plt.style.use('default')
    plt.ylim(y_lim[0], y_lim[1])
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.scatter(x_train, y_train, marker='+', c="k", s=20, label="Training data")
    plt.plot(x_real, y_real, 'r-', linewidth=2, label='Truth')
    plt.plot(x_test, y_pred, 'b-', linewidth=2, label='Prediction')
    for seg in segments:
        x_seg, y_pred_seg, epistemic_seg, total_unc_seg = seg
        if label:
            plt.fill_between(x_seg, y_pred_seg - total_unc_seg, y_pred_seg + total_unc_seg, color='b', alpha=0.2, label='Total uncertainty')
            plt.fill_between(x_seg, y_pred_seg - epistemic_seg, y_pred_seg + epistemic_seg, color='g', alpha=0.4, label='Epistemic uncertainty')
            label = False
        else:
            plt.fill_between(x_seg, y_pred_seg - total_unc_seg, y_pred_seg + total_unc_seg, color='b', alpha=0.2)
            plt.fill_between(x_seg, y_pred_seg - epistemic_seg, y_pred_seg + epistemic_seg, color='g', alpha=0.4)

    font = {'family' : 'Calibri',
    'weight' : 'normal',
    'size'   : 20,}
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)
    plt.title(title, font)
    plt.legend(loc=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)


def save_tecplot_result(flowdata_path, save_tecplotdata_path, data, variable, header, footer):
    tec_file = open(flowdata_path, 'r')
    tec_information = tec_file.readlines()
    tec_title = tec_information[:header]
    tec_final = tec_information[-footer:]   # hi-fidelity文件该值等于75640，即原文件中“ELEMENTS= "后面的数量
    tec_file.close()

    for i, line in enumerate(tec_title):
        if "ZONE T=" in line:
            # 在前面增加文本内容，表示新增的变量
            tec_title.insert(i, variable)
            break

    with open(save_tecplotdata_path, 'w') as write_file:
        write_file.writelines(tec_title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                write_file.write('%.9e\t' % (data[i, j]))
            write_file.write('\n')
        write_file.writelines(tec_final)
    print('tecplot_file write successfully---' + save_tecplotdata_path)