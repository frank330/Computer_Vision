import os
import sys
import torch.optim as optim
from torchvision import transforms, datasets
from urt.utils import read_split_data, train_one_epoch, evaluate
from urt.model import resnet34
from urt.my_dataset import MyDataSet
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

def result_test(real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='weighted')
    recall = recall_score(real, pred, average='weighted')
    f1 = f1_score(real, pred, average='weighted')
    patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
    print(patten % (acc, precision, recall, f1,))


def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/acc.png', dpi=400)

def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('results/loss.png', dpi=400)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
data_path = './dataset/data1\\'
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),]),
    "val": transforms.Compose([
                               transforms.ToTensor(),
                               ])}

# 实例化训练数据集
train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=data_transform["train"])

# 实例化验证数据集
val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])


batch_size = 8
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=nw,
                                           collate_fn=train_dataset.collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=nw,
                                         collate_fn=val_dataset.collate_fn)


"""
该函数主要功能是使用预训练的ResNet34模型进行迁移学习，对给定的数据集进行训练和验证，并保存训练好的模型。具体步骤如下：
1、加载预训练的ResNet34模型，并将其迁移到指定的设备上；
2、修改模型的全连接层，将其输出维度改为40；
3、定义损失函数为交叉熵损失，使用Adam优化器进行优化；
4、设置训练的轮数、学习率等参数；
5、在每个训练轮次中，对训练数据进行迭代训练，计算损失并进行反向传播；
6、在每个训练轮次结束后，对验证数据进行评估，计算准确率；
7、将每个训练轮次的损失和准确率记录下来，并绘制损失和准确率曲线；
8、如果当前轮次的验证准确率高于历史最佳准确率，则保存当前模型；
9、输出训练完成的信息"""
def train():
    """
    训练ResNet34模型。

    该函数初始化一个ResNet34模型，加载预训练的权重，然后微调模型以适应特定任务。
    训练过程包括多个epoch，每个epoch中模型在训练集上进行训练，并在验证集上评估性能。
    最后，函数将保存训练后的模型权重，并绘制训练过程中的准确率和损失曲线。
    """
    # 初始化ResNet34模型
    net = resnet34()
    # 指定预训练模型权重的路径
    model_weight_path = "ModelFile/resnet34-pre.pth"
    # 确保预训练模型权重文件存在
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # 加载预训练模型权重
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu',weights_only=False))
    # 修改模型的最后全连接层，以适应目标任务的类别数
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 40)
    # 将模型移动到指定的设备上（如GPU或CPU）
    net.to(device)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 筛选出需要优化的参数
    params = [p for p in net.parameters() if p.requires_grad]
    # 初始化优化器
    optimizer = optim.Adam(params, lr=0.0001)
    # 定义训练的epoch数
    epochs = 50
    # 初始化最佳准确率
    best_acc = 0.0
    # 定义保存训练后模型的路径
    save_path = 'ModelFile/ResNet34.pth'
    # 计算训练集的总步数
    train_steps = len(train_loader)
    # 初始化记录训练过程中的准确率和损失的列表
    acc11 = []
    loss11 = []
    # 对每个epoch进行训练
    for epoch in range(epochs):
        # 设置模型为训练模式
        net.train()
        # 初始化训练过程中的损失
        running_loss = 0.0
        # 使用tqdm库为训练数据加载器创建一个进度条
        train_bar = tqdm(train_loader, file=sys.stdout)
        # 对训练集中的每个batch进行处理
        for step, data in enumerate(train_bar):
            # 提取图像和标签
            images, labels = data
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            logits = net(images.to(device))
            # 计算损失
            loss = loss_function(logits, labels.to(device))
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 累加训练损失
            running_loss += loss.item()
            # 更新进度条描述，显示当前epoch和训练损失
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # 设置模型为评估模式
        net.eval()
        # 初始化验证过程中的准确率
        acc = 0.0  # accumulate accurate number / epoch
        # 关闭梯度计算，提高内存效率
        with torch.no_grad():
            # 使用tqdm库为验证数据加载器创建一个进度条
            val_bar = tqdm(val_loader, file=sys.stdout)
            # 对验证集中的每个batch进行处理
            for val_data in val_bar:
                # 提取图像和标签
                val_images, val_labels = val_data
                # 前向传播
                outputs = net(val_images.to(device))
                # 获取预测结果
                predict_y = torch.max(outputs, dim=1)[1]
                # 累加正确预测的数量
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                # 更新进度条描述
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        # 计算验证集的准确率
        val_num = len(val_dataset)
        val_accurate = acc / val_num
        # 记录验证准确率和训练损失
        acc11.append(val_accurate)
        loss11.append(running_loss / train_steps)
        # 打印当前epoch的训练损失和验证准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        # 如果当前的验证准确率高于之前的最佳准确率，则更新最佳准确率，并保存当前模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    # 绘制训练过程中的准确率和损失曲线
    plot_acc(acc11)
    plot_loss(loss11)
    # 打印训练完成的消息
    print('Finished Training')


# 模型评估
def evals():
    net = resnet34(num_classes=40).to(device)
    weights_path = "ModelFile/ResNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    net.eval()
    labels=[]
    predicts=[]
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            val_labels = val_labels.to(device).data.cpu().numpy().tolist()
            for i in val_labels:
                labels.append(i)
            predict_y = predict_y.data.cpu().numpy()
            for i in predict_y:
                predicts.append(i)
    result_test(labels,predicts)

if __name__ == '__main__':
    train()
    evals()
