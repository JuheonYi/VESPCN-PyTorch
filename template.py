def set_template(args):
    if args.template == 'SY':
        args.data_train = 'CDVL_Video'
        args.dir_data = '../../Dataset'
        args.data_test = 'CDVL_Video'
        args.dir_data_test = '../../Dataset'
        args.process = True
    elif args.template == 'JH':
        args.model = "ESPCN_modified"
        args.epochs = 1000
        args.data_train = 'CDVL100'
        args.dir_data = '/home/johnyi/deeplearning/research/SISR_Datasets/train'
        args.data_test = 'Set5'
        args.dir_data_test = '/home/johnyi/deeplearning/research/SISR_Datasets/test'
        args.process = True
    elif args.template == 'JH_video':
        args.model = "MotionCompensator"
        args.epochs = 1000
        args.data_train = 'CDVL_VIDEO'
        args.dir_data = '/home/johnyi/deeplearning/research/VSR_Datasets/train'
        args.data_test = 'Vid4'
        args.dir_data_test = '/home/johnyi/deeplearning/research/VSR_Datasets/test'
        args.process = True
    else:
        # TODO: Download train/test data & modify args for real testing
        args.batch_size = 2
        args.epochs = 1000
        args.dir_data = '/Users/junhokim/videoSR/data'
        args.dir_data_test = '/Users/junhokim/videoSR/data'
        args.process = True
        args.n_sequence = 4
