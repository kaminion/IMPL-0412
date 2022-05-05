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
 
def get_color(cursor):
    color = ['red', 'blue', 'orange', 'green']

    return color[cursor]

def draw(history_dict):
    plt.xlabel('epochs')
    plt.ylabel('top-1 Accuracy (%)')
    cursor = 0
    for legend in history_dict:
        history = history_dict[legend]
        color = get_color(cursor)
        plt.plot(history['train'], color=color, linestyle='--', label=f"{legend} Train")
        cursor += 1
        color = get_color(cursor)
        plt.plot(history['val'], color=color, label=f"{legend} Val")
        cursor += 1
        # legend_list.append(f"{legend} Train, {legend} Val")
    plt.legend()
    plt.savefig('result_graph.png')

# history = JSON_to_history('resNet')
# draw({"resnet":{'train':[1, 2], 'val': [3, 4]}, "resNext": {'train': [3, 4], 'val': [5, 6]}})
