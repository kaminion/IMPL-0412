import os


def get_lr(opt):
    """
        description: 현재 learning_rate 계산
    """
    for param_group in opt.param_groups:
        return param_group['lr']


def create_directory(directory):
    """
        description: 모델 저장소 만듬
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('error')
