import os
import glob
from importlib import import_module

from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        list_benchmarks = ['Set5', 'Set14', 'B100', 'Urban100']
        benchmark = self.data_test in list_benchmarks
        
        list_benchmarks_video = ['Vid4']
        benchmark_video = self.data_test in list_benchmarks_video
        if not self.args.test_only:
            m_train = import_module('data.' + self.data_train.lower())
            trainset = getattr(m_train, self.data_train)(self.args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu
            )
        else:
            self.loader_train = None

        if benchmark:
            m_test = import_module('data.benchmark')
            testset = getattr(m_test, 'Benchmark')(self.args, name=args.data_test, train=False)
        elif benchmark_video:
            m_test = import_module('data.benchmark_video')
            testset = getattr(m_test, 'Benchmark_video')(self.args, name=args.data_test, train=False)
        else:
            class_name = self.data_test
            m_test = import_module('data.' + class_name.lower())
            testset = getattr(m_test, class_name)(self.args, train=False)

        self.loader_test = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=not self.args.cpu)

