from src.utils import *
from src.rMAP.model import *

import numpy as np
import os
import time

from configs import *

args = rmap_default_args


# %% mian
if __name__ == '__main__':
    dir = args['case_dir']
    data_dir = os.path.join(dir, 'data/' + args['case_name'])
    result_dir = os.path.join(dir, 'results/' + args['case_name'])
    mkdir(result_dir)
    model_name = args['model_name']

    x_train = np.loadtxt(os.path.join(data_dir, 'x_train.dat'))
    y_train = np.loadtxt(os.path.join(data_dir, 'y_train.dat'))
    x_test = np.loadtxt(os.path.join(data_dir, 'x_test.dat'))
    y_test = np.loadtxt(os.path.join(data_dir, 'y_test.dat'))

    if np.ndim(x_train) == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
    if np.ndim(y_train) == 1:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    train_dataset = DataGenerator(x_train, y_train, args['batch'])
    test_dataset = DataGenerator(x_test, y_test, args['batch'])

    model_map = {
        'rmap': lambda: rMAP_BayesModel(layers=args['layers'],
                                        lamda=args['lamda'],
                                        lr=args['lr'],
                                        data_noise_scale=args['data_noise_scale'],
                                        param_noise_scale=args['param_noise_scale'],
                                        device=args['device']),

        }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model type: {model_name}\n Available models: {list(model_map.keys())}")
    model = model_map[model_name]()
    
    t0 = time.time()
    model.train(train_dataset=train_dataset, epochs=args['epochs'], n_samples=args['n_samples'], burn_in_samples=args['burn_in_samples'])
    t1 = time.time()

    train_time = t1 - t0
    print('训练时间：%.2f秒' % train_time)

    loss = np.stack((model.loss_log, model.err_log, model.comp_log, model.llhd_log), axis=1)
    np.savetxt(os.path.join(result_dir, 'loss.dat'), loss)
    plot_xy(x=np.arange(len(model.loss_log)), y=model.loss_log, title='log_loss', xlabel='epochs', ylabel='loss',
            save_path=os.path.join(result_dir, 'log_loss.png'), y_log=False)
    plot_xy(x=np.arange(len(model.err_log)), y=model.err_log, title='log_err', xlabel='epochs', ylabel='l2_error',
            save_path=os.path.join(result_dir, 'log_err.png'))
    plot_xy(x=np.arange(len(model.comp_log)), y=model.comp_log, title='log_lgprior', xlabel='epochs', ylabel='prior',
            save_path=os.path.join(result_dir, 'log_lgprior.png'), y_log=False)
    plot_xy(x=np.arange(len(model.llhd_log)), y=model.llhd_log, title='log_lgllhd', xlabel='epochs', ylabel='llhd',
            save_path=os.path.join(result_dir, 'log_lgllhd.png'), y_log=False)

    
    # 测试集预测
    pred = []
    aleatoric = []
    test_model = model_map[model_name]()
    for i in range(args['n_samples']):
        model_path = model.model_dir + '/' + model_name + '_bayesmodel_%d' % i + '.pth'
        test_model.load(model_path)
        pred_i, aleatoric_i = test_model.predict(x_test)
        pred_i = pred_i.detach().cpu().numpy()
        aleatoric_i = aleatoric_i.detach().cpu().numpy()
        pred.append(pred_i)
        aleatoric.append(aleatoric_i)
    pred = np.stack(pred, axis=0)
    aleatoric = np.stack(aleatoric, axis=0)
    print('pred shape: ', pred.shape)
    print('aleatoric shape: ', aleatoric.shape)
    mean = pred.mean(axis=0)
    epistemic = 2*pred.std(axis=0)
    aleatoric = 2*aleatoric.mean(axis=0)
    print("aleoratic :", aleatoric)
    total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

    result_data = np.concatenate((x_test, mean, epistemic, total_unc), axis=1)
    np.savetxt(os.path.join(result_dir, 'prediction.dat'), result_data)
    print("l2 loss in testset: ", np.linalg.norm(y_test-mean)/np.linalg.norm(y_test))

    plot_bayes_pred(x_train=train_dataset.x, y_train=train_dataset.y, x_real=test_dataset.x, y_real=test_dataset.y,
                    x_test=x_test, y_pred=mean, epistemic=epistemic, total_unc=total_unc, xlabel='x', ylabel='y',
                    title=model_name, save_path=os.path.join(result_dir, 'prediction_' + model_name + '.png'),
                    invert_yaxis=True)