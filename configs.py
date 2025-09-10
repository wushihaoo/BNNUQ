bbp_default_args = {
  "model_name": 'bbp',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "layers": [1, 64, 64, 1],              # 网络结构
  "prior_sigma": 0.5,                # 先验分布的标准差

  "device": 'cpu', 

  "lr": 1e-1,
  "epochs":3000,
  "batch": 20,
  "mc_samples": 3,
}
# test 参数
  # "layers": [1, 64, 1],              # 网络结构
  # "prior_sigma": 0.5,                # 先验分布的标准差

  # "device": 'cpu', 

  # "lr": 1e-1,
  # "epochs": 10000,
  # "batch": 20,
  # "mc_samples": 1,

# rae2822参数
  # "layers": [1, 100, 100, 100, 100, 1],              # 网络结构
  # "prior_sigma": 2.5,                # 先验分布的标准差

  # "device": 'cpu', 

  # "lr": 1e-1,
  # "epochs": 10000,
  # "batch": 20,
  # "mc_samples": 1,

pbnnbbp_default_args = {
  "model_name": 'pbnnbbp',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "dnnlayers": [1, 64, 64, 40],           # DNN网络结构
  "bnnlayers": [1, 64, 40],                # 网络结构
  "prior_sigma": 0.5,                   # 先验分布的标准差

  "device": 'cpu', 

  "lr": 1e-2,
  "epochs": 3000,
  "batch": 20,
  "mc_samples": 3,
}

# test 参数
  # "dnnlayers": [1, 64, 40],           # DNN网络结构
  # "bnnlayers": [1, 40],                # 网络结构
  # "prior_sigma": 10.0,                   # 先验分布的标准差

  # "device": 'cpu', 

  # "lr": 1e-2,
  # "epochs": 10000,
  # "batch": 20,
  # "mc_samples": 1,

mcdropout_default_args = {
  "model_name": 'mcdropout',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "layers": [1, 64, 64, 1],              # 网络结构
  "pdrop": 0.5,                      # MC Dropout的dropout概率
  "prior_sigma": 0.5,                # 先验分布的标准差
  "lamda": 1e-6,                     # 正则项系数

  "device": 'cpu', 

  "lr": 1e-1,
  "epochs": 3000,
  "batch": 20,
  "mc_samples": 3,

}
# test 参数
  # "layers": [1, 64, 1],              # 网络结构
  # "pdrop": 0.1,                      # MC Dropout的dropout概率
  # "prior_sigma": 0.5,                # 先验分布的标准差
  # "lamda": 1e-4,                     # 正则项系数

  # "device": 'cpu', 

  # "lr": 1e-1,
  # "epochs": 10000,
  # "batch": 20,
  # "mc_samples": 1,

# rae2822参数
  # "layers": [1, 100, 100, 100, 100, 1],              # 网络结构
  # "pdrop": 0.5,                      # MC Dropout的dropout概率
  # "prior_sigma": 0.5,                # 先验分布的标准差
  # "lamda": 1e-4,                     # 正则项系数

  # "device": 'cpu', 

  # "lr": 1e-2,
  # "epochs": 10000,
  # "batch": 20,
  # "mc_samples": 1,


hmc_default_args = {
  "model_name": 'hmc',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "layers": [1, 64, 64, 1],              # 网络结构
  "prior_sigma": 2.0,                # 先验分布的标准差

  "device": 'cpu', 

  "step_size": 1.0294e-2,                 # 这个参数很难调，要跟正则项系数一起调，建议先调正则项系数
  "n_steps": 10,
  "burn_in_epochs": 4000,
  "sample_nets": 50,
  "mix_epochs": 10,
  "batch": 20,

}
# test 参数
  # "layers": [1, 64, 1],              # 网络结构
#   "prior_sigma": 2.5,                # 先验分布的标准差

#   "device": 'cpu', 

#   "step_size": 4.0294e-2,                 # 这个参数很难调，要跟正则项系数一起调，建议先调正则项系数
#   "n_steps": 10,
#   "burn_in_epochs": 200,
#   "sample_nets": 50,
#   "batch": 20,

pbnnhmc_default_args = {
  "model_name": 'pbnnhmc',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "dnnlayers": [1, 64, 64, 40],           # DNN网络结构
  "bnnlayers": [1, 64, 40],                # 网络结构
  "prior_sigma": 2.0,                   # 先验分布的标准差

  "device": 'cpu', 

  "dnnlr": 1e-3,
  "step_size": 7e-3,                 
  "n_steps": 10,
  "map_epochs": 2000,
  "burn_in_epochs": 4000,
  "sample_nets": 50,
  "mix_epochs": 10,
  "batch": 20,
}

# test 参数
  # "dnnlayers": [1, 64, 64, 40],           # DNN网络结构
  # "bnnlayers": [1, 64, 40],                # 网络结构
  # "prior_sigma": 2.5,                   # 先验分布的标准差

  # "device": 'cpu', 

  # "dnnlr": 1e-3,
  # "step_size": 7.1e-3,                 
  # "n_steps": 10,
  # "burn_in_epochs": 200,
  # "sample_nets": 100,
  # "batch": 20,


sghmcm_default_args = {
  "model_name": 'sghmcm',
  "case_name": 'test', 
  "case_dir": '.',

  "layers": [1, 64, 1],              # 网络结构
  "prior_sigma": 2.5,                # 先验分布的标准差

  "device": 'cpu', 

  "step_size": 4e-1,                 # 这个参数很难调，要跟正则项系数一起调，建议先调正则项系数
  "friction": 0.01,
  "burn_in_epochs": 4000,
  "sample_nets": 50,
  "mix_epochs": 200,
  "batch": 40,

}

sghmc_default_args = {
  "model_name": 'sghmc',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "layers": [1, 64, 64, 1],              # 网络结构
  "prior_sigma": 2.0,                # 先验分布的标准差

  "device": 'cpu', 

  "lr": 1e-2,                 #
  "base_C": 0.05,
  "burn_in_epochs": 6000,
  "sample_nets": 50,
  "mix_epochs": 100,
  "batch": 20,

}

# test 参数
  # "layers": [1, 64, 1],              # 网络结构
  # "prior_sigma": 10,                # 先验分布的标准差

  # "device": 'cpu', 

  # "lr": 5e-4,                 #
  # "alpha": 0.01,
  # "burn_in_epochs": 1000,
  # "sample_nets": 50,
  # "batch": 20,

pbnnsghmc_default_args = {
  "model_name": 'pbnnsghmc',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "dnnlayers": [1, 64, 64, 40],           # DNN网络结构
  "bnnlayers": [1, 64, 40],                # 网络结构
  "prior_sigma": 2.0,                # 先验分布的标准差

  "device": 'cpu', 

  "dnnlr": 1e-3,
  "lr": 1e-2,                 #
  "base_C": 0.05,
  "map_epochs": 500,
  "burn_in_epochs": 6000,
  "sample_nets": 50,
  "mix_epochs": 100,
  "batch": 20,

}


rmap_default_args = {
  "model_name": 'rmap',
  "case_name": 'rae2822', 
  "case_dir": '.',

  "layers": [1, 64, 64, 1],              # 网络结构
  "lamda": 1e-2,                     # 正则项系数

  "device": 'cpu', 

  "lr": 1e-3,
  "data_noise_scale": 0.01,
  "param_noise_scale": 0.02,
  "n_samples": 100,
  "burn_in_samples": 20,
  "epochs": 201,
  "batch": 20,

}



sghmczxs_default_args = {
  "model_name": 'sghmczxs',
  "case_name": 'rae2822_zxs', 
  "case_dir": '.',

  "layers": [1, 64, 64, 1],              # 网络结构
  "lamda": 1e-2,                     # 正则项系数

  "device": 'cpu', 

  "lr": 1e-3,
  "data_noise_scale": 0.01,
  "param_noise_scale": 0.02,
  "n_samples": 100,
  "burn_in_samples": 20,
  "epochs": 201,
  "batch": 20,

}