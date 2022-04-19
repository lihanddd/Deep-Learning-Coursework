import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch.nn as nn
from time import time

class HandWritingNumberRecognize_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        # 这里添加数据集的初始化内容
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 将图片(Image)转成Tensor
            transforms.Normalize(mean=[.5], std=[.5]) # 常用标准化
        ])

        images_path = Path(root+"/images")
        images_list = list(images_path.glob('*.bmp'))
        images_list_str = [ str(x) for x in images_list ]
        images_list_str.sort(key=lambda x : int(x.split('.')[0].split('_')[1] )  )
        self.images = images_list_str

        if train: #训练集或验证集
            self.is_train = True
            labels_path = Path(root)
            labels_list = list(labels_path.glob('*.txt'))
            with open(str(labels_list[0]),'r') as f:
                labels = f.read().split("\n")[:-1]
                self.labels = torch.tensor(list(map(int, labels)))
        else: #测试集
            self.is_train = False

    def __getitem__(self, index):
        # 这里添加getitem函数的相关内容
        image_path = self.images[index]
        image = Image.open(image_path)  # 读取到的是RGB， W, H, C
        image = self.transform(image)   # transform转化image为：C, H, W

        label = self.labels[index] if self.is_train else None

        return image, label if self.is_train else image

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.images)

class HandWritingNumberRecognize_Network(torch.nn.Module):
    def __init__(self, drop_out_prob=0.5):
        super(HandWritingNumberRecognize_Network, self).__init__()
        # 此处添加网络的相关结构，下面的pass不必保留
        self.drop_out_prob = drop_out_prob
        self.layer1 = nn.Sequential(
            nn.Linear(1*28*28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.hid_layers = self.make_layer(2)
        self.layer2 = nn.Sequential(
            nn.Linear(256, 10)
        )

    def make_layer(self, num_of_hid):
        layers = []
        for i in range(num_of_hid):
            layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
            )
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, input_data):
        # 此处添加模型前馈函数的内容，return函数需自行修改
    
        out = input_data.view(input_data.shape[0], -1)
        out = self.layer1(out)
        for layer in self.hid_layers:
            out = layer(out)
        out = self.layer2(out)
        return out


def validation():
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in data_loader_val:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)   #有GPU则将数据置入GPU加速
    
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)  #输出得分最高的类
            total += labels.size(0) #统计50个batch 图片的总个数
            correct += (predicted == labels).sum()  #统计50个batch 正确分类的个数
    accuracy = 100*correct.item()/total
    global best_val_acc, best_epoch
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        best_epoch = epoch
    print("验证集数据总量：", total, "预测正确的数量：", correct.item())
    print("当前模型在验证集上的准确率为：", accuracy)


def alltest():
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留
    res = []
    best_path = str(best_epoch)+".pth"
    output_path = "./labels_test.txt"
    batch_count, total_len = 0, len(data_loader_test)
    model.load_state_dict(torch.load(best_path))
    print("testing")
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for i,data in enumerate(data_loader_test):
            inputs, _ = data
            inputs = inputs.to(device)   #有GPU则将数据置入GPU加速
    
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)  #输出得分最高的类
            res.append(predicted.tolist())
            batch_count += 1
            if i % 99 == 0:
                print('process: %.2f%%'
                    % (100*batch_count/total_len))
    print("done")
    with open(output_path, 'w', encoding='utf-8') as f:
        for labels in res:
            for label in labels:
                f.write(str(label)+'\n')


def train(epoch_num):
    # 循环外可以自行添加必要内容
    train_l_sum, train_acc_sum, n, batch_count, total_len, start = 0.0, 0.0, 0, 0, len(data_loader_train), time()
    for i,data in enumerate(data_loader_train):
        inputs,labels = data
        inputs, labels = inputs.to(device), labels.to(device)   #有GPU则将数据置入GPU加速
 
        # 梯度清零
        optimizer.zero_grad()
 
        # 传递损失 + 更新参数
        output = model(inputs)
        loss = loss_function(output,labels)
        loss.backward()
        optimizer.step()
 
        _, predicted = torch.max(output.data, 1)  #输出得分最高的类
        train_l_sum += loss.cpu().item()
        train_acc_sum += (predicted == labels).sum()
        n += inputs.shape[0]
        batch_count += 1

        if i % 99 == 0:
            print('process: %.2f%%, epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
                % (100*batch_count/total_len, epoch_num, train_l_sum / batch_count, train_acc_sum / n, time() - start))

    return train_l_sum / batch_count


if __name__ == "__main__":

    BATCH_SIZE = 64
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 构建数据集，参数和值需自行查阅相关资料补充。
    dataset_train = HandWritingNumberRecognize_Dataset("datas/train", train=True)
    # dataset_train = datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())

    dataset_val = HandWritingNumberRecognize_Dataset("datas/val", train=True)

    dataset_test = HandWritingNumberRecognize_Dataset("datas/test", train=False)

    # 构建数据加载器，参数和值需自行完善。
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE)

    data_loader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE)

    data_loader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE)

    print("device:",device)

    # 初始化模型对象，可以对其传入相关参数
    model = HandWritingNumberRecognize_Network().to(device)

    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)  # torch.optim中的优化器进行挑选，并进行参数设置

    max_epoch = 30  # 自行设置训练轮数
    num_val = 2  # 经过多少轮进行验证

    best_val_acc = 0
    best_epoch = 0
    

    # 然后开始进行训练
    for epoch in range(0, max_epoch):
        loss = train(epoch)
        torch.save(model.state_dict(),'%d.pth' % (epoch))
        # 在训练数轮之后开始进行验证评估
        if (epoch+1) % num_val == 0:
            validation()
        print("最好的模型为epoch:",best_epoch, " 验证集上准确率为:",best_val_acc)


    # 自行完善测试函数，并通过该函数生成测试结果
    alltest()