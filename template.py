def set_template(args):
    if args.template == 'SY':
        args.data_train = 'DIV2K'
        args.dir_data = '../../Dataset'
        args.data_test = 'Set5'
        args.dir_data_test = '../../Dataset'
        args.process = True
