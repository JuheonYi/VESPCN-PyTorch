import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True):
        super(DIV2K, self).__init__(
            args, name=name, train=train
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        if self.args.template == 'SY':
            self.apath = os.path.join(dir_data, self.name)
            self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
            self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic', 'X{}'.format(self.args.scale))
        ################################################
        #                                              #
        # Fill in your directory with your own template#
        #                                              #
        ################################################
        elif self.args.template == 'JH':
            print("Loading DIV2K")
            self.dir_hr = os.path.join(dir_data, 'DIV2K')
            self.dir_lr = os.path.join(dir_data, 'DIV2K_LR')
            #print(self.dir_hr)
            #print(self.dir_lr)

