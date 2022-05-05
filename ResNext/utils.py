import torch 
from torch.utils.data import DataLoader
from global_utils import create_directory, get_lr
import time

# dl: data-loader
def get_param_train(opt, loss_func, train_dl, val_dl, lr_scheduler, device, sanity_check=False):
    """
        description: 하이퍼 파라미터 정의
    """

    params_train = {
        'num_epochs': 10,
        'optimizer': opt,
        'loss_func': loss_func,
        'train_dl': train_dl,
        'val_dl': val_dl,
        'sanity_check': sanity_check,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'path2weights': './models/resnext_weight.pt'
    }

    create_directory('./models')

    return params_train

def metric_batch(output: torch.Tensor, target: torch.Tensor, k=1):
    """
    description:
        배치 당 metric을 계산, 여기선 cross entropy 기준으로 지표 측정
    Args:
        output: Prediction을 의미 함
        target: Ground Truth
    """

    # Top-K accuracy
    if(k == 1):
        # 10개의 Label을 가진 256개의 배열이 들어 올 것 (배치 당 이므로)
        # 그러므로 10개의 정답 중 가장 큰 것을 찾아야 함
        # 배치를 감싼 배열의 차원: 0, 정답을 가진 배치의 차원 : 1
        # 고로 1을 기준으로 계산
        # 라벨은 원 핫 인코딩 벡터로 들어온다는 것을 유의
        pred = output.argmax(1, keepdim=1)
        # prediction 과 똑같은 차원으로 만든 뒤, 동일한 것들만 더해서 반환함 (맞은 갯수 반환)
        # 예측값과 정답 비교, 정답은 동일 차원으로 만든 뒤(view_as) 비교
        # 비교 후 일치하는 것들만 모두 더함
        corrects = pred.eq(target.view_as(pred)).sum().item()
    else:
        # 1을 기준으로 계산함(차원)
        _, pred = output.topk(k, 1, True, True)
        # 행렬 기준 바꿈 reshape
        pred = pred.t()
        # [[1], [2]] 라벨 들어있는 것을 [1, 2] 이런식으로 reshape 함
        corrects = pred.eq(target.view(1, -1)).sum().item() # 5, 128을 128쪽에 flatten, 앞에 1차원만 남김

    return corrects


def loss_batch(loss_func, output: torch.Tensor, target, opt=None):
    """
    description:
        배치당 loss를 구하는 함수 
    Args:
        output: Prediction 값을 의미 
        target: 잘 알듯이 y값을 의미 함
    """
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    # 평가 함수와 동일 함수를 사용하기 위해 optimizer를 받지 않았으면 업데이트를 하지 않음
    if opt is not None:
        # 훈련 과정: 기울기 초기화 -> loss 계산(기울기 계산) -> 가중치 업데이트
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl: DataLoader, sanity_check=False, device='cpu', opt=None):
    """
    description:
        epoch당 loss와 metric을 정의하는 함수
        Args: 
            santity_check: 적은 데이터만 돌려서 모델이 잘 학습되는지 체크
    """
    running_loss = 0.0
    running_metric = 0.0    
    # 이번 에폭의 데이터셋 크기를 기준으로 배치에서 구한 loss로 전체 loss를 계산 함
    len_data = len(dataset_dl.dataset)

    # 데이터 배치에서 데이터 적재
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        
        pred = model(xb)

        # 배치 당 loss와 metric 계산
        loss_b, metric_b = loss_batch(loss_func, pred, yb, opt)

        running_loss += loss_b

        # metric 계산 시 
        if metric_b is not None:
            running_metric += metric_b
        
        # 학습이 잘 작동되는지 보는 목적이므로 배치 하나만 계산함
        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric

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
    device = params['device']

    # 그래프 그리기 위해서 필요
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # 지표 측정을 위해 필요
    best_loss = 1
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f"Epoch {epoch}/{num_epochs} current_lr = {current_lr}")

        # train part
        model.train()
        train_loss, train_metric = loss_epoch(
            model, loss_func, train_dl, sanity_check, device, opt
        )

        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        # validation part
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(
                model, loss_func, val_dl, sanity_check, device
            )
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path2weights)
            print("Copied Best Model Weights")

        lr_scheduler.step(val_loss)

        # metric은 계산할 때 소숫점으로 나오므로 백분위 환산
        print(
            f"train_loss: {train_loss}, val_loss: {val_loss}, accuracy: {100 * val_metric} time: {(start_time - time.time()) / 60}")
        print("=" * 10)

    return model, loss_history, metric_history