import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
import os
import torch

def plot_acc(attack_mode):
    save_dict = torch.load(os.path.join('models', attack_mode + '.pkl'))
    print('[{}] time: {:.2f} min'.format(attack_mode, save_dict['training_time'] / 60))
    xs, ys = [], []
    for item in save_dict['acc_list']:
        xs.append(item['step'])
        ys.append(item['acc'])
    plt.plot(xs, ys)

if __name__ == '__main__':
    attack_modes = ['none', 'fgsm', 'free', 'pgd']
    for attack_mode in attack_modes:
        plot_acc(attack_mode)
    plt.title('不同对抗训练方式在验证集上准确率')
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.legend(attack_modes)
    # plt.show()
    plt.savefig(os.path.join('images', 'val_acc.png'))