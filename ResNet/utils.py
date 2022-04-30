import torch
from global_utils import create_directory, get_lr
import time


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
def train_val(model, params):
    """
        description:
            학습을 진행하는 함수
    """

    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # 초기 best_loss 설정
    best_loss = float('int')
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f"Epoch {epoch}/{num_epochs} current_lr = {current_lr}")

        # train part
        model.train()
        train_loss, train_metric = loss_epoch(
            model, loss_func, train_dl, sanity_check, opt)

        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        # validation part
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(
                model, loss_func, val_dl, sanity_check, opt)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path2weights)
            print('Copied Best Model Weights')

        lr_scheduler.step(val_loss)

        # metric은 계산할 때 소숫점으로 나오므로 백분위 환산
        print(
            f"train_loss: {train_loss}, val_loss: {val_loss}, accuracy: {100 * val_metric} time: {(start_time - time.time()) / 60}")
        print("=" * 10)

    return model, loss_history, metric_history
