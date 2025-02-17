import argparse
import time
import random
import torch.nn as nn
from Save_result import show_pred
from util import *
from trainer import Optim
from metrics import *
from torch_cfc import Cfc
import warnings
warnings.filterwarnings("ignore")

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        # [64,12,3]
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

    # scale = data.scale.expand(predict.size(0), predict.size(1), 3).cpu().numpy()
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    predict = data._de_z_score_normalized(predict, 'cpu')
    Ytest = data._de_z_score_normalized(Ytest, 'cpu')
    mape = MAPE(predict, Ytest)
    mae = MAE(predict, Ytest)
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return mae, mape, correlation


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    total_mae_loss = 0

    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        # print(X.shape, Y.shape)
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        # print(X.shape)
        tx = X
        ty = Y
        output = model(tx)
        # print(output.shape)
        output = torch.squeeze(output)
        # print(output.shape)
        # scale = data.scale.expand(output.size(0), output.size(1), 3)
        # print("ty", ty, "output", output, "scale", scale, "output*scale", output * scale, "ty*scale", ty * scale)
        ty = data._de_z_score_normalized(ty, 'gpu')
        output = data._de_z_score_normalized(output, 'gpu')

        loss = mape_loss(ty, output)
        loss_mae = MAE((ty).cpu().detach().numpy(), (output).cpu().detach().numpy())
        loss_mse = RMSE((ty).cpu().detach().numpy(), (output).cpu().detach().numpy())

        loss_sum = loss + loss_mae + loss_mse

        coef_loss = ((loss_mae + loss_mse) * loss + (loss + loss_mae) * loss_mse + (loss * loss_mse) * loss_mae)/(loss_sum * loss_sum)

        loss.backward()
        total_loss += loss.item()
        total_mae_loss += loss_mae.item()
        grad_norm = optim.step()

        # if iter % 100 == 0:
        #     print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / 3))
        iter += 1
    return total_loss / iter, total_mae_loss / iter


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:

        _dict = {}
        for _, param in enumerate(model.named_parameters()):
            # print(param[0])
            # print(param[1])
            total_params = param[1].numel()
            # print(f'{total_params:,} total parameters.')
            k = param[0].split('.')[0]
            if k in _dict.keys():
                _dict[k] += total_params
            else:
                _dict[k] = 0
                _dict[k] += total_params
            # print('----------------')
        total_param = sum(p.numel() for p in model.parameters())
        bytes_per_param = 1
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes, _dict

def count_flops(model, input_size):
    def flops_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            # FLOPs for Conv2d: (H_out * W_out * K * K * C_in * C_out)
            H_out, W_out = output.shape[2], output.shape[3]
            K = module.kernel_size[0]
            C_in = module.in_channels
            C_out = module.out_channels
            flops = H_out * W_out * K * K * C_in * C_out
            module.__flops__ += flops

        elif isinstance(module, nn.Linear):
            # FLOPs for Linear: (in_features * out_features)
            flops = module.in_features * module.out_features
            module.__flops__ += flops

    # Register hooks to calculate FLOPs
    for layer in model.modules():
        layer.__flops__ = 0
        layer.register_forward_hook(flops_hook)

    # Create a dummy input with the specified input size
    dummy_input = torch.randn(input_size)
    model(dummy_input)  # Forward pass to trigger hooks

    total_flops = sum(layer.__flops__ for layer in model.modules() if hasattr(layer, '__flops__'))
    return total_flops

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/dataset_input.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='./model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
# parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes/variables')
# parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=15, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24 * 7, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=24*1, help='output sequence length')
parser.add_argument('--horizon', type=int, default=24*1)
parser.add_argument('--layers', type=int, default=5, help='number of layers')

parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

# 24 steps
parser.add_argument('--epochs', type=int, default=39, help='')
parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
parser.add_argument('--patience', type=int, default=2, help='patience')
parser.add_argument('--lr_d', type=float, default=0.5, help='inverse data')

# 48 steps
# parser.add_argument('--seq_out_len', type=int, default=24*2, help='output sequence length')
# parser.add_argument('--horizon', type=int, default=24*2)
# parser.add_argument('--epochs', type=int, default=39, help='')
# parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
# parser.add_argument('--patience', type=int, default=1, help='patience')
# parser.add_argument('--lr_d', type=float, default=0.45, help='inverse data')

# 72 steps
# parser.add_argument('--seq_out_len', type=int, default=24*3, help='output sequence length')
# parser.add_argument('--horizon', type=int, default=24*3)
# parser.add_argument('--epochs', type=int, default=39, help='')
# parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
# parser.add_argument('--patience', type=int, default=1, help='patience')
# parser.add_argument('--lr_d', type=float, default=0.45, help='inverse data')

# 96 steps
# parser.add_argument('--seq_out_len', type=int, default=24*4, help='output sequence length')
# parser.add_argument('--horizon', type=int, default=24*4)
# parser.add_argument('--epochs', type=int, default=39, help='')
# parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
# parser.add_argument('--patience', type=int, default=1, help='patience')
# parser.add_argument('--lr_d', type=float, default=0.45, help='inverse data')


parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes/variables')
parser.add_argument('--out_nodes', type=int, default=3, help='out_nodes')

args = parser.parse_args()

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
    "out_lens": args.seq_out_len,
    "in_lens": 168,
    "period_len":  [6, 12, 24],
    "sd_kernel_size": [6, 24],
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}


device = torch.device(args.device)
torch.set_num_threads(3)


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


def main():
    print("============Preparation==================")
    # seed = 1515
    seed = 2020
    fix_seed(seed)
    train_mae_losses = []
    val_mae_losses = []
    test_mae_losses = []
    Data = DataLoaderS(args.data, 0.8, 0.1, device, args.horizon, args.seq_in_len, args.normalize)
    model = Cfc(
        in_features=12,
        hidden_size=BEST_LTC["hidden_size"],
        out_feature=3,
        return_sequences=True,
        hparams=BEST_LTC,
        use_mixed=BEST_LTC["use_mixed"],
        use_ltc=BEST_LTC["use_ltc"],
    )
    model = model.to(device)
    #
    # flops, params = get_model_complexity_info(model, (1, 12, 168), as_strings=True, print_per_layer_stat=False)
    # print('flops: ', flops, 'params: ', params)

    total_param, total_megabytes, _dict = count_parameters(model)
    for k, v in _dict.items():
        print("Module:", k, "param:", v, "%3.3fM" % (v / (1024 * 1024)))
    print("Total megabytes:", total_megabytes, "M")
    print("Total parameters:", total_param)

    print(args)
    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    with open('./result/data.txt', 'a') as f:  # 设置文件对象
        print('Number of model parameters is', nParams, flush=True, file=f)

    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, 'min', args.lr_d, args.patience, 1, args.epochs, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss, train_mae_loss = train(Data, Data.train[0], Data.train[1][:, :, :3], model, criterion, optim,
                                               args.batch_size)
            val_mae, val_mape, val_corr = evaluate(Data, Data.valid[0], Data.valid[1][:, :, :3], model, evaluateL2,
                                                   evaluateL1,
                                                   args.batch_size)

            optim.lronplateau(val_mape)

            # optim.EXlr()
            # 将当前轮次的训练MAE损失添加到列表
            train_mae_losses.append(train_mae_loss)
            # 将当前轮次的验证MAE损失添加到列表
            val_mae_losses.append(val_mae)

            with open('./result/data.txt', 'a') as f:  # 设置文件对象
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f} | train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f}  learning rate  {:f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss,train_mae_loss, val_mae, val_mape, val_corr, optim.optimizer.param_groups[0]['lr']), flush=True,
                    file=f)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f}| train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f} | learning rate  {:f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss,train_mae_loss, val_mae, val_mape, val_corr, optim.optimizer.param_groups[0]['lr']), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            if val_mape < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_mape
            if epoch % 1 == 0:
                test_mae, test_mape, test_corr = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model, evaluateL2,
                                                          evaluateL1,
                                                          args.batch_size)
                with open('./result/data.txt', 'a') as f:  # 设置文件对象
                    print(
                        "test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f}".format(test_mae, test_mape,
                                                                                          test_corr),
                        flush=True, file=f)

                print("test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f}".format(test_mae, test_mape, test_corr),
                      flush=True)
                # 将当前轮次的测试MAE损失添加到列表
                test_mae_losses.append(test_mae)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    #保存最后一轮
    with open(args.save, 'wb') as f:
        torch.save(model, f)

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    test_mae, test_mape, test_corr = evaluate(Data, Data.test[0], Data.test[1][:, :, :3], model, evaluateL2, evaluateL1,
                                              args.batch_size)
    with open('./result/data.txt', 'a') as f:  # 设置文件对象
        print("final test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f}".format(test_mae, test_mape, test_corr),
              file=f)
    print("final test mae {:5.4f} | test mape {:5.4f} | test corr {:5.4f}".format(test_mae, test_mape, test_corr))

    all_y_true, all_predict_value = plow(Data, Data.test[0], Data.test[1][:, :, :3], model, args.batch_size)
    # 保存为 .pt 文件
    torch.save(all_y_true, './result/all_y_true.pt')
    torch.save(all_predict_value, './result/all_predict_value.pt')

    # 转换为 numpy 数组
    all_y_true_np = all_y_true.cpu().numpy()
    all_predict_value_np = all_predict_value.cpu().numpy()

    # 保存为 .npy 文件
    np.save('./result/all_y_true.npy', all_y_true_np)
    np.save('./result/all_predict_value.npy', all_predict_value_np)

    show_pred(all_y_true.cpu().numpy(), all_predict_value.cpu().numpy(), args.horizon)
    # 绘制训练、验证、测试的MAE损失随轮次变化的图像
    # plt.figure(figsize=(20, 10))  # 宽度、高度
    # plt.plot(train_mae_losses, label='Train MAE Loss', color='blue')
    # plt.plot(val_mae_losses, label='Val MAE Loss',color='skyblue')
    # plt.plot(test_mae_losses, label='Test MAE Loss', color='red')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE Loss')
    # plt.title('MAE Losses over Epochs')
    # plt.legend()
    # plt.show()
    return test_mae, test_mape, test_corr


def plow(data, X, Y, model, batch_size):
    model.eval()
    model.eval()
    all_predict_value = 0
    all_y_true = 0
    num = 0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        # scale = data.scale.expand(output.size(0), output.size(1), 3)  # zuijin xiugai
        # y_true = Y * scale
        # predict_value = output * scale
        y_true = data._de_z_score_normalized(Y, 'gpu')
        predict_value = data._de_z_score_normalized(output, 'gpu')

        if num == 0:
            all_predict_value = predict_value
            all_y_true = y_true
        else:
            all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
            all_y_true = torch.cat([all_y_true, y_true], dim=0)
        num = num + 1

    return all_y_true, all_predict_value


if __name__ == "__main__":
    main()
