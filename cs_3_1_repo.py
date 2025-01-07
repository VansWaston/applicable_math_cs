from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils import parallel_backend
# from thundersvm import SVC  # 支持GPU加速

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
import numpy as np
# import cv2
import math
import pandas as pd
import os
import logging
import argparse
import joblib

train_data_PATH = './data/kddcup99_train.csv'
test_data_PATH = './data/kddcup99_test.csv'

parser = argparse.ArgumentParser(description='job configuration')
parser.add_argument('--job_index', type=int, default=2, help='job index')
parser.add_argument('--job_stride', type=int, default=4, help='job stride')
parser.add_argument('--cls_model', type=str, default='SVM', help='classifier model')
parser.add_argument('--reduced_model', type=str, default='PCA', help='reduced model')
parser.add_argument('--percentage', type=float, default=0.1, help='percentage of training data')
parser.add_argument('--random_state', type=int, default=42, help='random state')
args = parser.parse_args()

CLS_MODEL = args.cls_model
REDUCED_MODEL = args.reduced_model

PRECENTAGE = args.percentage
RANDOM_STATE = args.random_state

JOB_INDEX = args.job_index
JOB_STRIDE = args.job_stride

FIG_PATH = './fig'
if os.path.exists(FIG_PATH) == False:
    os.mkdir(FIG_PATH)
FIG_PATH = FIG_PATH + '/'+CLS_MODEL+'_'+REDUCED_MODEL + '_'+ str(JOB_INDEX) + '_' + str(JOB_STRIDE) + '_' + str(PRECENTAGE) + '.png'

LOGGING_PATH = './log'
if os.path.exists(LOGGING_PATH) == False:
    os.mkdir(LOGGING_PATH)
LOGGING_PATH = LOGGING_PATH + '/'+CLS_MODEL+'_'+REDUCED_MODEL+'_'+ str(JOB_INDEX) + '_' + str(JOB_STRIDE) + '_' + str(PRECENTAGE) + '.log'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=LOGGING_PATH)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_SAVE_PATH = './model'
if os.path.exists(MODEL_SAVE_PATH) == False:
    os.mkdir(MODEL_SAVE_PATH)
AE_PATH = MODEL_SAVE_PATH + "/" + REDUCED_MODEL + '_' + str(PRECENTAGE) + '.pth'
MODEL_SAVE_PATH = MODEL_SAVE_PATH + '/'+CLS_MODEL+'_'+REDUCED_MODEL+'_'+ str(JOB_INDEX) + '_' + str(JOB_STRIDE) + '_' + str(PRECENTAGE) + '.pkl'

class EnDecoder(nn.Module):
    def __init__(self):
        super(EnDecoder,self).__init__()
        ## 定义Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(41,32),
            nn.Tanh(),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,4),
            nn.Tanh(),
            nn.Linear(4,2), 
            nn.Tanh(),
        )
        ## 定义Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(2,4),
            nn.Tanh(),
            nn.Linear(4,16),
            nn.Tanh(),
            nn.Linear(16,32),
            nn.Tanh(),
            nn.Linear(32,41),
            nn.Sigmoid(),
        )

    ## 定义网络的向前传播路径   
    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder,decoder
    
    def fit_transform(self, x):
        return self.Encoder(x)

def train_autoencoder(
    model, 
    train_data,
    test_data,
    batch_size,
    num_epochs,
    MODEL_PATH,
    device
):

    train_loader = Data.DataLoader(
        dataset = train_data, ## 使用的数据集
        batch_size = batch_size, # 批处理样本大小
        shuffle = True, # 每次迭代前打乱数据
        num_workers = 8, # 使用两个进程
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    criterion = nn.MSELoss()   # 损失函数
    
    model.to(device)
    model.train()
    train_num = 0
    ## 对模型进行迭代训练,对所有的数据训练num_epochs轮
    for epoch in range(num_epochs):
        train_loss_epoch = 0
        ## 对训练数据的迭代器进行迭代计算
        for _, b_x in enumerate(train_loader): 
            b_x = b_x.to(device)
            ## 使用每个batch进行训练模型
            _, output = model(b_x)         # 在训练batch上的输出
            loss = criterion(output, b_x)   # 平方根误差
            optimizer.zero_grad()           # 每个迭代步的梯度初始化为0
            loss.backward()                 # 损失的后向传播，计算梯度
            optimizer.step()                # 使用梯度进行优化
            train_loss_epoch += loss.item() * b_x.size(0)
            train_num = train_num + b_x.size(0)
        ## 计算一个epoch的损失
        train_loss = train_loss_epoch / train_num
        ## 保存每个epoch上的输出loss
        print('epoch [{}/{}], loss:{:.4f}, average loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item(), train_loss))
        logger.info('epoch [{}/{}], loss:{:.4f}, average loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item(), train_loss))
    
    print('Train Done!, Start Test...')
    logger.info('Train Done!, Start Test...')
    model.eval()
    for _, b_x in enumerate(test_loader):
        b_x = b_x.to(device)
        _, output = model(b_x)
        loss = criterion(output, b_x)
        print('Test Loss: {:.4f}'.format(loss.item()))
        logger.info('Test Loss: {:.4f}'.format(loss.item()))
    
    torch.save(MODEL_PATH)
    return model

if __name__ == '__main__':
    train_ds = pd.read_csv(train_data_PATH, header=None, on_bad_lines='skip')
    test_ds = pd.read_csv(test_data_PATH, header=None, on_bad_lines='skip')
    
    list_types = {}

    for i in range(train_ds.shape[1]):
        if isinstance(train_ds[i][0], str):
            list_type = train_ds[i].unique()
            dic = dict.fromkeys(list_type)
            for j in range(len(list_type)):
                dic[list_type[j]] = j
            list_types[i] = dic
            train_ds[i] = train_ds[i].map(dic)
            test_ds[i] = test_ds[i].map(dic)

    train_ds = train_ds.dropna()
    test_ds = test_ds.dropna()
    train_ds = train_ds.fillna(0)
    test_ds = test_ds.fillna(0)

    if abs(PRECENTAGE) > 1e-6:
        train_ds = train_ds.sample(frac=PRECENTAGE, random_state=RANDOM_STATE, axis=0)

    logger.info(f"list_types: {list_types}")
    logger.info(f"train data shape: {train_ds.shape}")
    
    train_target = train_ds.iloc[:, -1]
    train_ds = train_ds.iloc[:, :-1]
    test_target = test_ds.iloc[:, -1]
    test_ds = test_ds.iloc[:, :-1]
    
    x_shape=[]
    scores = []
    print('preprocess done,start process...')
    logger.info('preprocess done')
    logger.info(f'Process start:\njob index: %d, job stride: %d', JOB_INDEX, JOB_STRIDE)
    
    if JOB_STRIDE == 0:
        logger.info('JOB_STRIDE is 0, using all features')
        classifier = SVC(kernel='linear')
        if os.path.exists(MODEL_SAVE_PATH):
            print("classifier using loaded params.")
            logger.info('classifier using loaded params.')
            classifier = joblib.load(MODEL_SAVE_PATH)
        else:
            history = classifier.fit(train_ds,train_target)
        print('fit done,start score...')
        logger.info('fit done, start score...')
        score = classifier.score(test_ds,test_target)
        x_shape.append(train_ds.shape[1])
        scores.append(score)
        classifier.save_to_file(MODEL_SAVE_PATH)
        
        print('number of features:',train_ds.shape[1],'accuracy:',score)
        logger.info(f'number of features: {train_ds.shape[1]}, \naccuracy: {score}')
    else:
        # need to reduce the dimension
        for i in range(JOB_INDEX, train_ds.shape[1], JOB_STRIDE):
            if REDUCED_MODEL == 'PCA':
                reduced_model = PCA(i)
            else:
                # using AE to reduce
                reduced_model = EnDecoder()
                
                if os.path.exists(AE_PATH):
                    print("AE using loaded params.")
                    logger.info('AE using loaded params.')
                    reduced_model.load_state_dict(torch.load(AE_PATH))
                    reduced_model.eval()
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = EnDecoder().to(device)
                    train_data = torch.tensor(train_ds.values, dtype=torch.float32)
                    test_data = torch.tensor(test_ds.values, dtype=torch.float32)
                    batch_size = 512
                    num_epochs = 50
                    print('start training AE...')
                    logger.info('start training AE...')
                    reduced_model = train_autoencoder(model, train_data, test_data, batch_size, num_epochs, AE_PATH, device)
            
            train_x = reduced_model.fit_transform(train_ds)
            test_x = reduced_model.fit_transform(test_ds)
            
            classifier = SVC(kernel='linear')
            
            MODEL_SAVE_PATH = './model/'+CLS_MODEL+'_'+REDUCED_MODEL+'_'+ str(i) + '_' + str(JOB_STRIDE) + '_' + str(PRECENTAGE) + '.pkl'
            
            if os.path.exists(MODEL_SAVE_PATH):
                print("classifier using loaded params.")
                logger.info('classifier using loaded params.')
                classifier = joblib.load(MODEL_SAVE_PATH)
            else:
                history = classifier.fit(train_x,train_target)
                
            print('fit done,start score...')
            logger.info('fit done, start score...')
            score = classifier.score(test_x,test_target)
            x_shape.append(train_x.shape[1])
            scores.append(score)
            
            classifier.save_to_file(MODEL_SAVE_PATH)

            print('number of features:',train_x.shape[1],'accuracy:',score)
            logger.info(f'number of features: {train_x.shape[1]}, accuracy: {score}')


    plt.plot(x_shape,scores)
    plt.xlabel('number of features')
    plt.ylabel('accuracy')
    plt.savefig(FIG_PATH, dpi=300)
    plt.show()