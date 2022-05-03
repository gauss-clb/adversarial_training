import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
import os
import torch

def plot_acc(attack_mode):
    save_dict = torch.load(os.path.join('models', attack_mode + '.pkl'))
    xs, ys = [], []
    for item in save_dict['acc_list']:
        xs.append(item['step'])
        ys.append(item['acc'])
    print(xs)
    plt.plot(xs, ys)

if __name__ == '__main__':
    attack_modes = ['none', 'fgsm']
    plot_acc('none')
    plot_acc('fgsm')
    plt.title('不同对抗训练方式在验证集上准确率')
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.legend(attack_modes)
    # plt.show()
    plt.savefig('1.png')