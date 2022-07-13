import argparse
from remove_space import remove_space
from test_on_running import Tester
import tensorflow as tf
from tensorflow.python.tools import freeze_graph 
import numpy as np
import matplotlib.pyplot as plt
# Allows a model to be tested from the terminal.

# You need to input your test data directory
test_directory="~"
dir = 'E:/dataset/'
parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")
appName = "microwave" # kettle, fridge, washing machine, dishwasher, microwave.
parser.add_argument('--datadir',type=str,default='REDD') #UK-DALE REFIT REDD
parser.add_argument("--appliance_name", type=remove_space, default=appName)
parser.add_argument("--batch_size", type=int, default="400")
parser.add_argument("--crop", type=int, default="128000")
parser.add_argument("--algorithm", type=remove_space, default="seq2point") # 'mobilenet' 'densenet'
parser.add_argument("--network_type", type=remove_space, default="") 
parser.add_argument("--input_window_length", type=int, default="599")
parser.add_argument("--test_directory", type=str, default=test_directory)
parser.add_argument("--plot_result", type=bool, default=False)

arguments = parser.parse_args()
test_directory   = dir + arguments.datadir + '/' + arguments.appliance_name + '/' + arguments.appliance_name + '_test_' + '.csv'  #training, validation, test
saved_model_dir = "saved_models/" + arguments.datadir + '_s/' + arguments.appliance_name + "_" + arguments.algorithm + "_model.h5" #_best

log_file_dir = "saved_models/" + arguments.datadir + '_s/' + arguments.appliance_name + "_" + arguments.algorithm + "_" + arguments.network_type + ".log"
npy_output_dir = "saved_models/" + arguments.datadir + '_s/' #_best


gt = np.load(npy_output_dir+arguments.appliance_name +'_gt.npy')
pred = np.load(npy_output_dir+arguments.appliance_name +'_pred.npy')
mains = np.load(npy_output_dir+arguments.appliance_name +'_mains.npy')
k = 1
'''
gt = np.array([np.sum(gt1[index:index+k])/k for index in range(0,gt1.size,k)])
pred = np.array([np.sum(pred1[index:index+k])/k for index in range(0,pred1.size,k)])
mains = np.array([np.sum(mains1[index:index+k])/k for index in range(0,mains1.size,k)])
'''
window_offset = 299
skip = 3000
size = 5000
markersize = 18

'''
my_x_ticks = np.arange(-5, 5, 0.5)      #显示范围为-5至5，每0.5显示一刻度
my_y_ticks = np.arange(-2, 2, 0.2)      #显示范围为-2至2，每0.2显示一刻度
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
'''
for i in range (10000):
    skip = size + skip
    offs = (int)(window_offset/k)#-43
    plt.figure(figsize=(8, 6))
    plt.plot(mains[skip : size + skip], label="Aggregate",marker="*", markersize= markersize,markevery=23, markerfacecolor='none', linewidth=3.0,color='darkslategray')
    plt.plot(gt[skip - offs : size + skip - offs], label="Ground Truth",marker="x", markersize= markersize,markevery=29, markerfacecolor='none', linewidth=3.0,color='forestgreen')
    plt.plot(pred[skip - offs : size + skip - offs], label="Predicted",marker="o", markersize= markersize,markevery=37, markerfacecolor='none', linewidth=3,color='firebrick')
    font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 25,
    }
    plt.xticks([])  #去掉x轴
    plt.ylabel("Watts",font)
    plt.legend(prop={'size':27})
    #plt.yscale('log')
    plt.yticks([0,500,4000])#,[1e1,1e2,1e3]
    plt.tick_params(labelsize=13) #刻度字体大小13
    plt.show()
