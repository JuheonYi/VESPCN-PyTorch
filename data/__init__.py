import os
import glob
from importlib import import_module

from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        if args.task == 'zssr':
            apath = os.path.join(args.dir_data, args.data_test)
            #print(apath)
            self.path_hr = sorted(glob.glob(os.path.join(
                apath, 'HR', '*.png'
            )))
            self.path_lr = []
            for f in self.path_hr:
                name = os.path.splitext(os.path.split(f)[-1])[0]
                self.path_lr.append(os.path.join(
                    apath,
                    'LR_bicubic',
                    'X{}'.format(args.scale[0]),
                    '{}x{}.png'.format(name, args.scale[0])
                ))


    def get_loader(self, idx=0):
        list_benchmarks = ['Set5', 'Set14', 'B100', 'Urban100']
        benchmark = self.data_test in list_benchmarks
        if self.args.task == 'zssr':
            m = import_module('data.zeroshot')
            trainset = getattr(m, 'ZeroShot')(self.args, self.path_lr[idx])
            testset = getattr(m, 'ZeroShot')(
                self.args,
                self.path_hr[idx],
                train=False,
                benchmark=benchmark
            )
        else:
            if not self.args.test_only:
                m_train = import_module('data.' + self.data_train.lower())
                trainset = getattr(m_train, self.data_train)(self.args)

            if benchmark:
                class_name = 'Benchmark'
                if self.args.task == 'fusion' or self.args.task == 'fusion2' or self.args.task.find('ft') >= 0:
                    class_name += 'Alpha'
                m_test = import_module('data.' + class_name.lower())
                testset = getattr(m_test, class_name)(self.args, train=False)
            else:
                class_name = self.data_test
                if self.args.task == 'fusion' or self.args.task == 'fusion2' or self.args.task.find('ft') >= 0:
                    class_name += 'Alpha'
                m_test = import_module('data.' + class_name.lower())
                testset = getattr(m_test, class_name)(self.args, train=False)

        if not self.args.test_only:
            loader_train = DataLoader(
                self.args,
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu
            )
        else:
            loader_train = None

        loader_test = DataLoader(
            self.args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.args.cpu
        )

        return loader_train, loader_test
