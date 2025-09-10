from src.utils import *
from src.Bayes_By_Backprop.model import *
from src.Bayes_By_Backprop.model_pbnn import *
from src.MC_Dropout.model import *

import numpy as np
import os
import time

from configs import *

# args = mcdropout_default_args
# args = bbp_default_args
args = pbnnbbp_default_args


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
        'bbp': lambda: VI_BayesModel(layers=args['layers'],
                                        prior_class=isotropic_gauss_prior(mu=0, sigma=args['prior_sigma']),
                                        learning_rate=args['lr'],
                                        device=args['device']),
        'mcdropout': lambda: MCDropout_BayesModel(layers=args['layers'],
                                        pdrop=args['pdrop'],
                                        prior_class=isotropic_gauss_prior(mu=0, sigma=args['prior_sigma']),
                                        lamda = args['lamda'],
                                        learning_rate=args['lr'],
                                        device=args['device']),
        'pbnnbbp': lambda: VI_PBayesModel(dnnlayers=args['dnnlayers'],
                                        bnnlayers=args['bnnlayers'],
                                        prior_class=isotropic_gauss_prior(mu=0, sigma=args['prior_sigma']),
                                        learning_rate=args['lr'],
                                        device=args['device']),
        }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model type: {model_name}\n Available models: {list(model_map.keys())}")
    model = model_map[model_name]()
    print("model used: ", model_name)
    
    t0 = time.time()
    model.train(train_dataset=train_dataset, epochs=args['epochs'], mc_samples=args['mc_samples'])
    t1 = time.time()

    train_time = t1 - t0
    print('训练时间：%.2f秒' % train_time)

    loss = np.stack((model.comp_log, model.llhd_log, model.loss_log, model.err_log), axis=1)
    np.savetxt(os.path.join(result_dir, 'loss.dat'), loss)
    plot_xy(x=np.arange(len(model.comp_log)), y=model.comp_log, title='log_comp', xlabel='epochs', ylabel='comp', 
            save_path=os.path.join(result_dir, 'log_comp.png'), y_log=False)
    plot_xy(x=np.arange(len(model.llhd_log)), y=-1*np.array(model.llhd_log), title='-log_llhd', xlabel='epochs', ylabel='llhd',
            save_path=os.path.join(result_dir, '-log_llhd.png'))
    plot_xy(x=np.arange(len(model.loss_log)), y=model.loss_log, title='log_loss', xlabel='epochs', ylabel='loss',
            save_path=os.path.join(result_dir, 'log_loss.png'))
    plot_xy(x=np.arange(len(model.err_log)), y=model.err_log, title='log_err', xlabel='epochs', ylabel='l2_error',
            save_path=os.path.join(result_dir, 'log_err.png'))

    
    # 测试集预测
    test_model = model_map[model_name]()
    test_model.load(model.model_path)
    # mean, epistemic, aleatoric, mse_error = model.distribution_estimate(test_dataset=test_dataset, mc_samples=args['mc_samples'])
    # x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
    mean, epistemic, aleatoric = test_model.predict(train_dataset.normalize(x=x_test))

    mean = mean.detach().cpu().numpy()
    epistemic = epistemic.detach().cpu().numpy()
    aleatoric = aleatoric.detach().cpu().numpy()

    mean, epistemic, aleatoric = train_dataset.denormalize(y=mean), train_dataset.denormalize(y=epistemic), train_dataset.denormalize(y=aleatoric)
    print("aleoratic :", aleatoric)
    total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

    

    # result_data = np.concatenate((test_dataset.x, test_dataset.y, mean, epistemic, total_unc), axis=1)
    result_data = np.concatenate((x_test, mean, epistemic, total_unc), axis=1)
    np.savetxt(os.path.join(result_dir, 'prediction.dat'), result_data)

    plot_bayes_pred(x_train=train_dataset.x, y_train=train_dataset.y, x_real=test_dataset.x, y_real=test_dataset.y,
                    x_test=x_test, y_pred=mean, epistemic=epistemic, total_unc=total_unc, xlabel='x', ylabel='y',
                    title=model_name, save_path=os.path.join(result_dir, 'prediction_' + model_name + '.png'), 
                    invert_yaxis=True)