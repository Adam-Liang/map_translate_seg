__author__ = "charles"
__email__ = "charleschen2013@163.com"

import os
import os.path as osp
import json
import torch
from torch import nn

"""
Logger is built to log the information during the model train like loss, accuracy
ModelSaver is design to save the state_dict of model, optimizer, scheduler etc safetly
Author: Charles Chen
"""


class Logger:
    """
    Build for logging the data like losss, accuracy etc during the training
    """

    def __init__(self, save_path, json_name):
        self.create_dir(save_path)
        self.save_path = save_path
        self.json_name = json_name
        self.json_path = osp.join(save_path, json_name + '.json')
        # if .json file not exist create one
        if not osp.exists(self.json_path):
            with open(self.json_path, 'a') as f:
                json.dump({}, f)
        self.state = json.load(open(self.json_path, 'r'))
        # if 'max' not in self.state: self.state['max'] = {}

    def get_data(self, key):
        """
        :param key: key word of data
        :return: data[key]
        """
        if key not in self.state:
            print(f'*** find no {key} data!')
            return []
        else:
            return self.state[key]

    def get_max(self, key):
        """
        :param key:str
        :return:the max of state[key]
        """
        if key not in self.state:
            print(f'*** find no {key} data!')
            return float('-inf')
        try:
            # if key not in self.state['max']:
            return max(self.state[key])
        except Exception:
            print(f'sorry, cannot get the max of data {key}')
            return float('-inf')

    def log(self, key, data, show=False):
        if key not in self.state:
            self.state[key] = []
        self.state[key].append(data)
        if show:
            print(f'===> log key:{key} -> data:{data}')

    def save_log(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.state, f)
            print('*** Save log safely!')

    def size(self, key):
        try:
            return len(self.state[key])
        except Exception:
            return 0

    def visualize(self, key=None, range=None):
        """

        :param key: the key for dict to find data
        :param range: tuple, to get the range of data[key]
        :return:
        """
        if key is None:
            for key in self.state:
                data = self.state[key]
                if len(data) == 0:
                    continue
                self.save_training_pic(data=data,
                                       path=self.save_path, name=self.json_name,
                                       ylabel=key, xlabel='iteration')
        elif key not in self.state:
            print(f'*** find no data of {key}!')
        else:
            if range == None or not isinstance(range, tuple):
                if len(self.state[key]) != 0:
                    self.save_training_pic(data=self.state[key],
                                           path=self.save_path, name=self.json_name,
                                           ylabel=key, xlabel='iteration')
            else:
                if len(self.state[key]) != 0:
                    self.save_training_pic(data=self.state[key][range[0]:range[1]],
                                           path=self.save_path, name=self.json_name,
                                           ylabel=key, xlabel='iteration')

    @staticmethod
    def save_training_pic(data, path, name, ylabel, xlabel, smooth=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def moving_average_filter(data, n):
            assert smooth % 2 == 1
            res = [0 for i in data]
            length = len(data)
            for i in range(length):
                le = max(0, i - n // 2)
                ri = min(i + n // 2 + 1, length)
                s = sum(data[le:ri])
                l = ri - le
                res[i] = s / float(l)
            return res

        if isinstance(smooth, int):
            data = moving_average_filter(data, n=smooth)

        x = range(1, len(data) + 1)
        y = data
        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.set(xlabel=xlabel, ylabel=ylabel,
               title='{} {}'.format(name, ylabel))
        ax.grid()
        fig.savefig(path + '/{}_{}.png'.format(name, ylabel), dpi=330)
        plt.cla()
        plt.clf()
        plt.close('all')

    @staticmethod
    def create_dir(dir_path):
        if not osp.exists(dir_path):
            os.mkdir(dir_path)


class ModelSaver:
    """
    Build for loading and saving pytorch model automatically
    """

    def __init__(self, save_path, name_list, try_mode=True, strict=False, ext='pkl'):
        self.create_dir(save_path)
        self.save_path = save_path
        self.ext = ext
        self.name_dict = {name: osp.join(save_path, name + f'.{self.ext}') for name in name_list}
        self.try_mode = try_mode
        self.strict = strict

    def load(self, name, model, key=None):
        self.load_tool(self.name_dict[name], model, key, self.try_mode, self.strict)

    def save(self, name, model):
        if not self.try_mode:
            self.save_safely(model.state_dict(), self.save_path, file_name=name + f'.{self.ext}')
            print(f'*** Saving {name} successfully')
        else:
            try:
                self.save_safely(model.state_dict(), self.save_path, file_name=name + f'.{self.ext}')
            except Exception:
                print(f'*** Saving {name} fail!')
            else:
                print(f'*** Saving {name} successfully')

    @staticmethod
    def save_safely(file, dir_path, file_name):
        """
        save the file safely, if detect the file name conflict,
        save the new file first and remove the old file
        """
        if not osp.exists(dir_path):
            os.mkdir(dir_path)
            print('*** dir not exist, created one')
        save_path = osp.join(dir_path, file_name)
        if osp.exists(save_path):
            temp_name = save_path + '.temp'
            torch.save(file, temp_name)
            os.remove(save_path)
            os.rename(temp_name, save_path)
            print('*** find the file conflict while saving, saved safely')
        else:
            torch.save(file, save_path)

    @staticmethod
    def create_dir(dir_path):
        if not osp.exists(dir_path):
            os.mkdir(dir_path)

    @staticmethod
    def load_tool(load_path, model, key=None, try_mode=True, strict=True):
        """
        Loading the saved state_dict, under following situation, loading is ok, but it will destory the training process
        | 1. model is nn.DataParallel, state dict is normal :
        | 2. model is normal, state dict is for nn.DataParallel :
        :param load_path: str
        :param model: nn.Module
        :param key: str
        :param try_mode: bool
        :param strict: bool
        :return:
        """
        if not try_mode:
            if key is None:
                state_dict = torch.load(load_path, map_location='cpu')
            else:
                state_dict = torch.load(load_path, map_location='cpu')[key]

            # optimizer and scheduler do not have 'strict'
            try:
                model.load_state_dict(state_dict, strict=strict)
            except Exception:
                model.load_state_dict(state_dict)
            print(f'*** 1:Loading {load_path} successfully')
        else:
            try:
                if key is None:
                    state_dict = torch.load(load_path, map_location='cpu')
                else:
                    state_dict = torch.load(load_path, map_location='cpu')[key]

                try:
                    model.load_state_dict(state_dict, strict=strict)
                except Exception:
                    model.load_state_dict(state_dict)
                print(f'*** 1:Loading {load_path} successfully')
            except Exception:
                print(f'*** Trying Load {load_path} from multi-GPUs type...')
                try:
                    if key is None:
                        state_dict = torch.load(load_path, map_location='cpu')
                    else:
                        state_dict = torch.load(load_path, map_location='cpu')[key]

                    # create new OrderedDict that does not contain `module.`
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if 'module.' in k:
                            # print(k)
                            temp_name = k[7:]  # remove `module.`
                            # print(temp_name)
                            new_state_dict[temp_name] = v

                    # load params
                    try:
                        model.load_state_dict(new_state_dict, strict=strict)
                    except Exception:
                        model.load_state_dict(new_state_dict)

                    print(f'*** 2:Loading {load_path} successfully')
                except Exception:
                    print(f'*** Loading {load_path} fail!')


def _test_logger():
    test_dict = {
        'lr': [0.1, 0.1, ],
        'time': [1, 1]
    }
    with open('./test_dict.json', 'w') as f:
        json.dump(test_dict, f)
    with open('./test_dict.json', 'r') as f:
        temp_dict = json.load(f)
        print(temp_dict)

    logger = Logger(save_path='./', json_name='test')
    logger.log(key='lr', data=0.1)
    print(logger.get_data('lr'))
    logger.save_log()
    logger.visualize()


def _test_logger_load():
    from train_utils import get_model
    path = '../deeplabv3plusxception.pkl'
    model = get_model('deeplabv3plusxception', 21)
    # model = nn.DataParallel(model)
    # if isinstance(model, nn.DataParallel):
    #     print(True)
    # print(model)
    # model = model.module
    ModelSaver.load_tool(model=model, load_path=path)
    print(model)
    # d = torch.load(path, map_location='cpu')
    # print(d)


def from_multi_gpu_to_cpu(model_name, n_calss, load_path):
    """
    transfer nn.DataParallel model to original model
    :param model_name: str
    :param n_calss: int
    :param load_path: str
    :return:
    """
    from train_utils import get_model
    model = get_model(model_name, n_calss)
    model = nn.DataParallel(model)
    d = torch.load(load_path, map_location='cpu')
    model.load_state_dict(d)
    torch.save(model.module.state_dict(), load_path)
    pass


def from_cpu_to_multi_gpu(model_name, n_calss, load_path):
    """
    transfer original model to nn.DataParallel model
    :param model_name: str
    :param n_calss: int
    :param load_path: str
    :return:
    """
    from train_utils import get_model
    model = get_model(model_name, n_calss)
    d = torch.load(load_path, map_location='cpu')
    model.load_state_dict(d)
    model = nn.DataParallel(model)
    torch.save(model.state_dict(), load_path)
    pass


if __name__ == '__main__':
    _test_logger_load()
    pass