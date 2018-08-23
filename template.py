def set_template(args):
    if args.template == 'SY':
        args.data_train = 'CDVL_VIDEO'
        args.data_range = '1-16/90-100'
        args.dir_data = '../../Dataset'
        args.data_test = 'Vid4'
        args.dir_data_test = '../../Dataset'
        args.process = True
        args.lr = 5e-4
        if args.task == 'MC':
            args.model = 'MotionCompensator'
            args.n_sequence = 2

        elif args.task == 'Image':
            args.model = 'ESPCN'

        elif args.task == 'Video':
            args.model = 'ESPCN_multiframe2'
            args.patch_size = 17
            args.n_sequence = 3

    elif args.template == 'JH':
        args.task = "Image"
        args.save = args.model
        args.data_train = 'CDVL100'
        args.dir_data = '/home/johnyi/deeplearning/research/SISR_Datasets/train'
        args.data_test = 'Set5'
        args.dir_data_test = '/home/johnyi/deeplearning/research/SISR_Datasets/test'
        args.process = True
    elif args.template == 'JH_Video':
        args.task = "Video"
        args.save = args.model
        args.test_every = 1000
        args.n_sequence = 3
        args.n_frames_per_video = 15
        args.data_range = '1-135/91-100'
        args.data_train = 'CDVL_VIDEO'
        args.dir_data = '/home/johnyi/deeplearning/research/VSR_Datasets/train'
        args.data_test = 'Vid4'
        args.dir_data_test = '/home/johnyi/deeplearning/research/VSR_Datasets/test'
        args.process = True
    elif args.template == 'JH_MC':
        args.task = "MC"
        args.model = "MotionCompensator"
        args.save = args.model
        args.n_sequence = 2
        args.n_frames_per_video = 15
        args.data_range = '1-5/91-100'
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
