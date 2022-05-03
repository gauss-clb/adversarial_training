import os
import argparse

from cv2 import log
from model import TextCNN, TextCNNConfig
from dataset import *
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np
from sklearn import metrics
import time
from tqdm import tqdm

def test(model, dataloader, device):
    states = torch.load(os.path.join('models', args.attack_mode + '.pkl'))
    model.load_state_dict(states['net'])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            logits = model(input_ids).cpu().numpy()
            pred = np.argmax(logits, axis=-1)
            y_pred.extend(np.squeeze(pred))
            y_true.extend(labels.numpy())

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_true, y_pred, target_names=THUCNews.category2id, digits=4))

    print("Confusion Matrix...")
    print(metrics.confusion_matrix(y_true, y_pred))

def eval(model, dataloader, device):
    n_correct, n_total = 0, 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            logits = model(input_ids).cpu().numpy()
            pred = np.argmax(logits, axis=-1)
            n_correct += np.sum(pred == labels.numpy())
            n_total += pred.size
    acc = n_correct / n_total
    return acc

# 对抗训练
# free: 更新对抗样本和参数同时进行，并且对同一批训练样本进行多次对抗攻击，对不同训练样本delta不会重置
# pgd: 更新多轮对抗样本，更新一轮参数，对不同训练样本delta会重置
# fgsm: 更新一轮对抗样本，更新一轮参数，每次随机初始化delta
def train(config, model, train_dataloader, val_dataloader, device):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)

    # 用于存储对抗样本和原样本的embedding增量
    delta = torch.zeros(args.bs, args.max_len, config.embedding_dim).to(device)
    delta.requires_grad = True

    if args.attack_mode == 'free': # free方式迭代轮数除以args.attack_iters
        args.epochs = (args.epochs + args.attack_iters - 1) // args.attack_iters # 向上取整
    best_acc, step, acc = 0, 0, 0
    acc_list = []
    save_dict = {}
    training_time = 0
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        with tqdm(train_dataloader, desc='Epoch [{}/{}, {}] '.format(epoch+1, args.epochs, args.attack_mode), ncols=120) as pbar:
            for input_ids, labels in pbar:
                pbar.set_postfix(step=step, val_acc='{:.3f}'.format(acc)) 
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                start_time = time.time()
                if args.attack_mode == 'none':
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                elif args.attack_mode == 'free':
                    for _ in range(args.attack_iters): # 更新delta同时更新参数
                        logits = model(input_ids, delta[:input_ids.size(0)])
                        optimizer.zero_grad()
                        loss = F.cross_entropy(logits, labels)
                        loss.backward()
                        delta.data = delta.data + args.alpha * torch.sign(delta.grad)
                        delta.data = torch.clamp(delta.data, -args.epsilon, args.epsilon)
                        delta.grad.zero_()
                        optimizer.step()
                        step += 1
                elif args.attack_mode == 'pgd':
                    delta.data.zero_() # 初始化0
                    for _ in range(args.attack_iters): # 累计更新delta，不更新参数
                        logits = model(input_ids, delta[:input_ids.size(0)])
                        optimizer.zero_grad()
                        loss = F.cross_entropy(logits, labels)
                        loss.backward()
                        delta.data = delta.data + args.alpha * torch.sign(delta.grad)
                        delta.data = torch.clamp(delta.data, -args.epsilon, args.epsilon)
                        delta.grad.zero_()
                elif args.attack_mode == 'fgsm':
                    delta.data.uniform_(-args.epsilon, args.epsilon)  # 均匀分布初始化
                    logits = model(input_ids, delta[:input_ids.size(0)])  # 更新delta，不更新参数
                    optimizer.zero_grad()
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    delta.data = delta.data + args.alpha * torch.sign(delta.grad)
                    delta.data = torch.clamp(delta.data, -args.epsilon, args.epsilon)
                    delta.grad.zero_()
                if args.attack_mode in ('pgd', 'fgsm'): # 使用对抗样本更新一轮参数
                    logits = model(input_ids, delta[:input_ids.size(0)])
                    optimizer.zero_grad()
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    delta.grad.zero_()
                    step += 1
                training_time += time.time() - start_time 
                if step <= 100 and step % 10 == 0 or step % 200 == 0:
                    # 验证阶段
                    acc = eval(model, val_dataloader, device)
                    acc_list.append({'step': step, 'acc': acc})
                    if acc > best_acc:
                        best_acc = acc
                        save_dict = {'net': model.state_dict(), 'best_step': step, 'best_acc': best_acc}
    save_dict['acc_list'] = acc_list
    save_dict['training_time'] = training_time
    torch.save(save_dict, os.path.join('models', args.attack_mode + '.pkl'))
    print('Total train time: {:.2f} minutes'.format(training_time / 60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--max_len', type=int, default=512, help='max sequence length')
    parser.add_argument('--attack_iters', type=int, default=5, help='hyper-parameter for pgd/free')
    parser.add_argument('--lr', type=float, default=1e-3, help ='learning rate')
    parser.add_argument('--alpha', type=float, default=0.04, help ='hyper-parameter for adversarial training')
    parser.add_argument('--epsilon', type=float, default=0.1, help ='hyper-parameter for adversarial training')
    parser.add_argument('--attack_mode', required=True, choices=['none', 'free', 'pgd', 'fgsm'], help ='adversarial training algorithm')
    parser.add_argument('--dataset_dir', type=str, default='data')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_paths = get_dataset_paths(args.dataset_dir)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(file_paths, batch_size=args.bs, max_len=args.max_len)
    print('vocab size: {}'.format(THUCNews.vocab_size()))
    config = TextCNNConfig(
        vocab_size = THUCNews.vocab_size(),
        num_class=THUCNews.num_class(),
        max_len = args.max_len
    )
    textcnn = TextCNN(config)
    textcnn.to(device)
    if args.test:
        test(textcnn, test_dataloader, device)
    else:
        train(config, textcnn, train_dataloader, val_dataloader, device)
