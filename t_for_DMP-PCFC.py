import argparse
import torch
import numpy as np
from util import *
from metrics import *
from torch_cfc import Cfc
import random
import warnings
warnings.filterwarnings("ignore")

# 确保与训练时相同的参数设置
BEST_LTC = {
    "optimizer": "adam",
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "forget_bias": 2.4,
    "epochs": 80,
    "class_weight": 8,
    "clipnorm": 0,
    "hidden_size": 16,
    "backbone_units": 16,
    "backbone_dr": 0.2,
    "backbone_layers": 2,
    "weight_decay": 0,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 64,
    "out_lens": 24 * 1,  # 根据实际预测长度修改
    "in_lens": 168,
    "period_len": [6, 12, 24],
    "sd_kernel_size": [6, 24],
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    # 保持与训练代码相同的评估函数
    model.eval()
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    predict = data._de_z_score_normalized(predict, 'cpu')
    Ytest = data._de_z_score_normalized(Ytest, 'cpu')

    mape = MAPE(predict, Ytest)
    mae = MAE(predict, Ytest)
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return mae, mape, correlation


def t_DMP_PCFC(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载数据（确保与训练相同的路径和参数）
    Data = DataLoaderS(args.data, 0.8, 0.1, device, args.horizon, args.seq_in_len, args.normalize)

    # 初始化模型结构（必须与训练时完全一致）
    model = Cfc(
        in_features=12,
        hidden_size=BEST_LTC["hidden_size"],
        out_feature=3,
        return_sequences=True,
        hparams=BEST_LTC,
        use_mixed=BEST_LTC["use_mixed"],
        use_ltc=BEST_LTC["use_ltc"],
    ).to(device)

    # 加载保存的模型权重
    model = torch.load(args.save, map_location=device)
    model.eval()

    # 定义损失函数
    evaluateL2 = torch.nn.MSELoss(size_average=False).to(device)
    evaluateL1 = torch.nn.L1Loss(size_average=False).to(device)

    # 在测试集上评估
    test_mae, test_mape, test_corr = evaluate(Data, Data.test[0], Data.test[1][:, :, :3],
                                              model, evaluateL2, evaluateL1, args.batch_size)

    # 打印结果
    print(f"Test Results:")
    print(f"MAE: {test_mae:.4f}")
    print(f"MAPE: {test_mape:.4f}%")
    print(f"Correlation: {test_corr:.4f}")

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cuda_seed = torch.cuda.initial_seed()
    # print("当前GPU随机种子:", cuda_seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


if __name__ == "__main__":
    fix_seed(2020)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/dataset_input.csv')
    parser.add_argument('--save', type=str, default='model/24-steps/model.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seq_in_len', type=int, default=24 * 7)
    parser.add_argument('--seq_out_len', type=int, default=24 * 1)
    parser.add_argument('--horizon', type=int, default=24 * 1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--normalize', type=int, default=2)
    args = parser.parse_args()

    t_DMP_PCFC(args)