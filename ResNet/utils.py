import torch
from global_utils import create_directory


def get_param_train(opt, loss_func, train_dl, val_dl, lr_scheduler, sanity_check=False):
    """
        description: 하이퍼파리미터 정의
    """

    params_train = {
        'num_epochs': 10000,
        'optimizer': opt,
        'loss_func': loss_func,
        'train_dl': train_dl,
        'val_dl': val_dl,
        'sanity_check': sanity_check,
        'lr_scheduler': lr_scheduler,
        'path2weights': './models/weights.pt'
    }

    create_directory('./models')

    return params_train


def metric_batch(output: torch.Tensor, target: torch.Tensor):
    """ 
    description:
        배치당 metric을 계산, cross entropy에서 많이 사용
    """
    # 차원 수 유지, 어느 차원 기준으로 argmax 적용할 건지.
    pred = output.argmax(1, keepdim=True)
    # prediction 과 똑같은 차원으로 만든 뒤, 동일한 것들만 더해서 반환함 (맞은 갯수 반환)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


def loss_batch(loss_func, output: torch.Tensor, target, opt=None):
    """ 
    description:
        batch 당 loss 구하기 
    """
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    description:
        epoch당 loss과 metric을 정의하는 함수
        Args:
            santity_check: 적은 데이터만 돌려서 모델이 잘 학습되는지 체크
    """
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    # 데이터 배치에서 데이터 적재
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)  # y^hat

        # loss와 metric 계산
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        # metric 계산 시 틀린게 있을 수도 있으므로 is not None
        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric


# 위 함수들을 활용해서 학습하는 함수
def train_val(model, params, lr_scheduler, ):
    """
        description:
            학습을 진행하는 함수
    """
