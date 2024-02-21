import torch
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

start = time.time()
SEED = 0
TEXT = torchtext.legacy.data.Field(lower=True,fix_length = 200, batch_first=False)
LABEL = torchtext.legacy.data.Field(sequential = False)
#sequential = 데이터에 순서가 있는지 나타낸다. ( 기본 True ) ( imbd 데이터 셋은 긍정과 부정이므로 False )

#데이터 다운로드 
from torch.legacy import datasets
train_data , test_data = datasets.IMDB.splits(TEXT,LABEL)


print(train_data.example[0])#데이터 확인

#데이터 전처리
import string
for example in train_data.example:
    text = [x.lower() for x in vars(example)['text']]
    text = [x.replac("<br","") for x in text]
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]
    text = [s for s in text if s]
    
#데이터셋을 Train Validation 으로 구분 + 데이터 개수 확인
import random
train_data , val_data = train_data.split(random_stat=random.seed(SEED),split_ratio = 0.8)
print(f'Number of training examples : {len(train_data)}')
print(f'Number of validation examples : {len(val_data)}')
print(f'Number of test examples : {len(test_data)}')


#단어집합(중복을 제거한 딕셔너리와 같은 집합) 생성
TEXT.build_vocab(train_data,max_size = 10000, min_freq = 10, vectors = None)
LABEL.build_vocab(train_data)

#데이터 집합 확인
print(TEXT.vocab.stoi)
print(LABEL.vocab.stoi)

#데이터셋 메모리로 가져오기
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embeding_dim = 100 #각 단어를 100차원으로 조정 (embeding 계층을 통과한 뒤의 벡터의 크기)
hidden_size = 300 #은닉층 유닛(노드)의 개수 
#일반적으로 유닛 개수 늘리기보단 계층(layer)개수 늘리는 것이 성능에 더 좋다. 
#은닉층의 층 수 => 좀 더 비선형 문제를 잘 학습하게 함
#노드 => 가중치와 바이어스 계산하는 용도

train_iterator,valid_iterator,test_iterator = torchtext.legacy.data.BucketIterator.splits((train_data,val_data,train_data),batch_size=BATCH_SIZE,device=device)
#BucketIterator == DataLoader
#배치 크기 단위로 값을 차례대로 메모리로 가져옴
#비슷한 길이의 데이터를 한 배치에 할당하여 Padding을 최소화 시켜준다.
#각 파라미터 설명 ( (배치 크기 단위로 가져올 데이터 셋) , (배치 사이즈) , (장치 설정) )


# 워드 임베딩 및 RNN 셀 정의
class RNNCell_Encoder(nn.Module):
    def __init__(self,input_dim,hidden_size):
        super(RNNCell_Encoder,self).__init__()
        self.rnn = nn.RNNCell(input_dim,hidden_size) 
        #RNN Cell (trainset의 feature의 수, 은닉층 뉴런의 수)
    def forward(self,inputs):
        bz = inputs.shape[1]
        #배치를 가져온다.
        ht = torch.zeros((bz,hidden_size)).to(device)
        
        for word in inputs :
            #inputs에서 단어를 하나씩 가져옴
            ht = self.rnn(word,ht)
            #재귀적으로 발생하는 상태값을 처리하기 위한 현재 시점 ht
            #현재 시점 ht = (입력벡터(x_t),(이전 상태 h_t-1))
        return ht
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.em = nn.Embedding(len(TEXT.vocab.stoi),embeding_dim)#임베딩 처리를 위한 구문 (임베딩할 단어의 수, 임베딩할 벡터의 차원)
        self.rnn = RNNCell_Encoder(embeding_dim,hidden_size) # Cell 생성
        self.fc1 = nn.Linear(hidden_size,256) # fully Connected
        self.fc2 = nn.Linear(256,3) #fully Connected
    
    def forward(self,x):
        x = self.em(x)
        x = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#옵티마이저/손실함수 정의
model = Net()
model.to(device)


loss_fn = nn.CrossEntopyLoss() #CrossEntropyLoss는 다중 분류에 사용된다. torch.nn.CrossEntropyLoss는 LogSoftmax + NLLLoss 연산 조합
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

#모델 학습을 위한 함수
def training(epoch,model,train_loader,valid_loader):
    correct = 0
    total = 0
    running_loss = 0
    
    model.train()
    for b in train_loader:
        x,y = b.text,b.label #b데이터에서 text,label 분리
        x,y = x.to(device), y.to(device) #디바이스 연결
        y_pred = model(x) # 예측 결과 
        loss = loss_fn(y_pred,y) #loss 계산
        optimizer.zero_grad() #gradient 초기화를 해주어야 다음 step에서 이번 step과 무관하게 학습 가능
        loss.backward() #backward를 통해 loss에 끼친 파라미터들의 영향력(gradient)를 구한다.
        optimizer.step() #파라미터 업데이트
        with torch.no_grad():
            y_pred = torch.argmax(y_pred,dim=1)
            correct += (y_pred==y).sum().item() #맞춘거 개수 correct에 넣기
            total += y.size(0) #y item의 수를 total에 넣기
            running_loss += loss.item() #loss 값을 running_loss에 더해줌
            
    # 모든 단어들에 대해 수행된 후에 이 곳으로 나옴
    epoch_loss = running_loss / len(train_loader.dataset) #loss를 데이터 셋의 길이로 나눈다.
    epoch_acc = correct / total # 정답 비율 
    
    valid_correct = 0 
    valid_total = 0
    valid_running_loss = 0
    
    model.eval()
    with torch.no_grad():
        #검증과정에서는 grad nope
        for b in valid_loader:
            x,y = b.text,b.label
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            y_pred = torch.argmax(y_pred,dim=1)
            valid_correct += (y_pred==y).sum().item()
            valid_total += y.size(0)
            valid_running_loss += loss.item()
            
    epoch_valid_loss = valid_running_loss / len(valid_loader.dataset)
    epoch_valid_acc = valid_correct / valid_total
    
    print('epoch: ',epoch, 'loss: ',round(epoch_loss,3),'accuracy: ',round(epoch_acc,3),'valid_loss: ',round(epoch_valid_loss,3),'valid_accuracy: ',round(epoch_valid_acc,3))
    
    return epoch_loss,epoch_acc,epoch_valid_loss,epoch_valid_acc

#실제 훈련
epochs = 5
train_loss=[]
valid_loss=[]
train_acc=[]
valid_acc=[]

for epoch in range(epochs):
    epoch_loss,epoch_acc,epoch_valid_loss,epoch_valid_acc = training(epoch,model,train_iterator,valid_iterator)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    valid_loss.append(epoch_valid_loss)
    valid_acc.append(epoch_valid_acc)
    
end = time.time()

print(end-start)