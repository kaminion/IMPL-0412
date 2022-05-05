import os
import json 
import matplotlib.pyplot as plt

# global_util 파일임


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

def history_to_JSON(name="Model", history=[]):

    with open(f'./{name}_history_info.json', 'w') as file:
        json.dump(history, file)

def JSON_to_history(name):
    print(name)

    with open(f'./{name}_history_info.json', 'r') as file:
        history = json.load(file)
    print(history)
    return history
 
def draw(legend, *history_list):
    for history in history_list:
        plt.xlabel('epochs')
        plt.ylabel('top-1 error (%)')
        plt.plot(history['train'], '--r')
        plt.plot(history['val'], 'b')
        plt.legend([f'{legend} Train', f'{legend} Val'])
        plt.savefig('result_graph.png')

history_to_JSON('resNet', {'train': [1, 2], 'val': [3, 4]})
history = JSON_to_history('resNet')
draw('resNet', history)
