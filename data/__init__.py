import os
import glob
from importlib import import_module

from torch.utils.data import DataLoader


class Data:

    """
    Data class: Wrapper class containing both training and testing data. DataLoaders are used to load data from disk.
    """
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        list_benchmarks = ['Set5', 'Set14', 'B100', 'Urban100']  # Benchmark datasets used for testing performance
        benchmark = self.data_test in list_benchmarks
        if not self.args.test_only:
            m_train = import_module('data.' + self.data_train.lower())  # Import module for some specified dataset
            trainset = getattr(m_train, self.data_train)(self.args)  # Get training data set for a pre-specified task
            self.loader_train = DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu  # Copy tensors to CUDA pinned memory if using gpu
            )

        if benchmark:  # If we are using a benchmark dataset, use the pre-implemented Benchmark class
            m_test = import_module('data.benchmark')
            testset = getattr(m_test, 'Benchmark')(self.args, name=args.data_test, train=False)
        else:  # Else, implement a new module for the specified super-resolution task
            class_name = self.data_test
            m_test = import_module('data.' + class_name.lower())
            testset = getattr(m_test, class_name)(self.args, train=False)

        self.loader_test = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=not self.args.cpu)

