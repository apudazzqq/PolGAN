class FLAGES(object):

    pan_size = 512
    ms_size = 8

    num_spectrum = 1

    ratio = 64
    stride = 2
    norm = True

    batch_size = 16
    lr = 0.0001
    decay_rate = 0.99
    decay_step = 10000
    
    img_path='./data/source_data'
    data_path='./data/train/train_qk.h5'
    log_dir='./log_generator'
    model_save_dir='./model_generator'
    model_path = './model_generator/Generator-5000'
    #model_path = './Models/model_30000/Generator-30000'
    result_path = './data/result'
    
    is_pretrained = False
    
    iters = 5000
    model_save_iters = 1000
    valid_iters = 20


