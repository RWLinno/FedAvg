###################
# 导入相关库
###################
import os
import sys
import gzip
import numpy
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset,DataLoader

###################
# 加载数据集
###################
# 读取字节流
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4),dtype=dt)[0]

# 创建路径
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# 解压图像并返回[index,y,x,depth]格式
def extract_images(filename):
    print('Extracting',filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic,filename)
            )
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf,dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

# 将标签中的标量转化为one-hot向量
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.revel()] = 1
    return labels_one_hot

# 将标签转化为一维uint8数组
def extract_labels(filename):
    print('Extracting',filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic !=2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic,filename)
            )
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf,dtype=np.uint8)
        return dense_to_one_hot(labels)

def get_data():
    print("prepare for dataset")
    mnistDataSet = GetDataSet('mnist',True) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray \
            and type(mnistDataSet.test_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

# 获取数据类
class GetDataSet(object):
    def __init__(self,dataSetName,isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self.index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        else:
            pass

    def mnistDataSetConstruct(self,isIID):
        data_dir = r'.data\MNIST'
        train_images_path = os.path.join(data_dir,'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir,'train_labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_images(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_images(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1

        train_images = train_images.reshape(train_images.shape[0],train_images.shape[1]*train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0],test_images.shape[1]*test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images,1.0/255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images,1.0/255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels,axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

###################
# 构造模型
###################
class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,10)

    def forward(self,inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512,10)

    def forward(self,inputs):
        tensor = inputs.view(-1,1,28,28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1,7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

###################
# 定义客户端类
###################
class client(object):
    def __init__(self,trainDataSet,dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self,localEpoch,localBatchSize,Net,lossFun,opti,global_parameters):
        Net.load_state_dict(global_parameters,strict=True)
        self.train_dl = DataLoader(self.train_ds,batch_size=localBatchSize,shuffle=True)
        for epoch in range(localEpoch):
            for data,label in self.train_dl:
                data,label = data.to(self.dev),label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds,label)
                loss.backward()
                opti.step()
                opti.zero_grad()
        return Net.state_dict()

    def local_val(self):
        pass

###################
# 客户群
###################

class ClientsGroup(object):
    def __init__(self,dataSetName,isIID,numOfClients,dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

def test_Client():
    print("prepare for the clients group")
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])

###################
# 定义服务器端类
###################

class Server(object):
    def __init__(self):
        self.gpu = '0'
        self.num_of_clients = 100
        self.cfraction = 0.1
        self.epoch = 5
        self.batchsize = 10
        self.model_name ='mnist_2nn'
        self.learning_rate = 0.01
        self.val_freq = 5
        self.save_freq = 20
        self.num_comm = 1000
        self.save_path ='./checkpoints'
        self.IDD = 0

    def FedAvg(self,i):
        order = np.random.permutation(self.num_of_clients)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None  # 一开始w0为0
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(self.epoch, self.batchsize, net,
                                                                         loss_func, opti, global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % self.val_freq == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))
        # 传回并保存参数
        if (i + 1) % self.save_freq == 0:
            torch.save(net, os.path.join(self.save_path,
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(self.model_name,
                                                                                                i, self.epoch,
                                                                                                self.batchsize,
                                                                                                self.learning_rate,
                                                                                                self.num_of_clients,
                                                                                                self.cfraction)))

###################
# 主程序
###################

if __name__=="__main__":
    get_data()
    test_Client()
    server = Server()
    test_mkdir(server.save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = server.gpu
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if server.model_name == 'mnist_2nn':
        net = Mnist_2NN()
    elif server.model_name == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=server.learning_rate)

    myClients = ClientsGroup('mnist', server.IID, server.num_of_clients, dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(server.num_of_clients * server.cfraction, 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(server.num_comm):
        print("communicate round {}".format(i+1))
        server.FedAvg(i)  #Server再每一轮通信执行FedAvg算法