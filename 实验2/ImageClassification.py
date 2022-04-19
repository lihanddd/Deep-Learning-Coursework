from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from pathlib import Path
from PIL import Image
import torch.nn.functional as neural_funcs
from torchvision import transforms, datasets
from time import time
from tqdm import tqdm


# 数据加载
class CifarDataset(Dataset):
    def __init__(self, mode='train'):
        # 这里添加数据集的初始化内容
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        self.transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
                transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2))
            ])

        self.transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std)
            ])

        if mode == "train":  # 训练集
            print("loading trainset...")
            filepath = 'Dataset/trainset.txt'
            self.transform = self.transform_train
            self.test = False
        elif mode == "val":
            print("loading validset...")
            filepath = 'Dataset/validset.txt'
            self.transform = self.transform_test
            self.test = False
        elif mode == "test":
            print("loading testset...")
            filepath = 'Dataset/testset.txt'
            self.transform = self.transform_test
            self.test = True

        self.images_path = []
        self.labels = [] if not self.test else None
        with open(filepath, 'r') as file:
            for line in file.readlines():
                image, label = line.split() if not self.test else [
                    line.strip(), None]
                self.images_path.append('Dataset/image/'+image)
                self.labels.append(int(label)) if not self.test else None

        print("loaded!")

    def __getitem__(self, index):
        # 这里添加getitem函数的相关内容
        # image = self.images[index]
        path = self.images_path[index]
        image = Image.open(path)  # 读取到的是RGB， W, H, C
        image = self.transform(image)   # transform转化image为：C, H, W

        label = self.labels[index] if not self.test else None

        return image, label if not self.test else image

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.images_path)


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places,
                  kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class BaseBlock(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(BaseBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.baseblock = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places *
                      self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places *
                          self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.baseblock(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# 构建模型


class Net(nn.Module):
    def __init__(self, blocks, num_classes=10, expansion=4):
        # 定义模型的网络结构
        super(Net, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self._make_layer(
            in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self._make_layer(
            in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self._make_layer(
            in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self._make_layer(
            in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(BaseBlock(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(BaseBlock(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 定义模型前向传播的内容
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18():
    return Net([2, 2, 2, 2])


def ResNet50():
    return Net([3, 4, 6, 3])


def ResNet101():
    return Net([3, 4, 23, 3])


def ResNet152():
    return Net([3, 8, 36, 3])


# 定义 train 函数
def train():
    # 参数设置
    epoch_num = 40
    val_num = 1
    global epoch

    for tmp_epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch = tmp_epoch
        train_l_sum, train_acc_sum, n, batch_count, total_len, start = 0.0, 0.0, 0, 0, len(
            train_loader), time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(
                device)  # 有GPU则将数据置入GPU加速

            # 梯度清零
            optimizer.zero_grad()

            # 传递损失 + 更新参数
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)  # 输出得分最高的类
            train_l_sum += loss.cpu().item()
            train_acc_sum += (predicted == labels).sum()
            n += inputs.shape[0]
            batch_count += 1

            if (i+1) % 10 == 0:
                print('\rprocess: %.2f%%, epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
                      % (100*batch_count/total_len, epoch, train_l_sum / batch_count, train_acc_sum / n, time() - start), end=' ')
        scheduler.step()
        print('\rprocess: %.2f%%, epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (100*batch_count/total_len, epoch, train_l_sum / batch_count, train_acc_sum / n, time() - start))

        # 模型训练n轮之后进行验证
        if (epoch+1) % val_num == 0:
            validation()

    print('Finished Training!')


# 定义 validation 函数
def validation():
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():
        for data in dev_loader:
            inputs, labels = data
            # 在这一部分撰写验证的内容
            inputs, labels = inputs.to(device), labels.to(
                device)  # 有GPU则将数据置入GPU加速

            output = net(inputs)
            _, predicted = torch.max(output.data, 1)  # 输出得分最高的类
            total += labels.size(0)  # 统计50个batch 图片的总个数
            correct += (predicted == labels).sum()  # 统计50个batch 正确分类的个数
    accuracy = 100*correct.item()/total
    global best_val_acc, best_epoch
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        best_epoch = epoch
        torch.save(net.state_dict(), "best."+str(best_val_acc)+".ckpt")

    print("验证集数据总量：", total, "预测正确的数量：",
          correct.item(), "最好的epoch：", best_epoch)
    print("当前模型在验证集上的准确率为：", accuracy, "最好的准确率为：", best_val_acc)


# 定义 test 函数
def test():
    # 将预测结果写入result.txt文件中，格式参照实验1
    res = []
    output_path = "./labels_test.txt"
    batch_count, total_len = 0, len(test_loader)
    print("testing")
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for i, data in enumerate(test_loader):
            inputs, _ = data
            inputs = inputs.to(device)  # 有GPU则将数据置入GPU加速

            output = net(inputs)
            _, predicted = torch.max(output.data, 1)  # 输出得分最高的类
            res.append(predicted.tolist())
            batch_count += 1
            if (i+1) % 10 == 0:
                print('\rprocess: %.2f%%'
                      % (100*batch_count/total_len), end='')
    print("\ndone")
    with open(output_path, 'w', encoding='utf-8') as f:
        for labels in res:
            for label in labels:
                f.write(str(label)+'\n')


if __name__ == "__main__":

    BATCH_SIZE = 32
    best_epoch = 0
    best_val_acc = 0
    epoch = 0

    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 构建数据集
    train_set = CifarDataset("train")
    dev_set = CifarDataset("val")
    test_set = CifarDataset("test")

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE)
    dev_loader = DataLoader(dataset=dev_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE)

    # 初始化模型对象
    net = ResNet50().to(device)

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 定义优化器
    # torch.optim中的优化器进行挑选，并进行参数设置
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    # 模型训练
    train()

    # 对模型进行测试，并生成预测结果
    test()
