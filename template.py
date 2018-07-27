def set_template(args):
    if args.template == 'SY':
        args.data_train = 'DIV2K'
        args.dir_data = '../../Dataset'
        args.data_test = 'Set5'
        args.dir_data_test = '../../Dataset'
        args.process = True
    elif args.template == 'JH':
        args.data_train = 'CDVL100'
        args.dir_data = '/home/johnyi/deeplearning/research/SISR_Datasets/train'
        args.data_test = 'Set5'
        args.dir_data_test = '/home/johnyi/deeplearning/research/SISR_Datasets/test'
        args.process = True
