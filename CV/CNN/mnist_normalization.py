import torch # framework
import matplotlib.pyplot as plt # for data visualizing
import numpy as np # for data manifulation

import torchvision 
import torchvision.transforms as transforms # for data preprocessing

import torch.nn as nn #for building model
import torch.optim as optim #for optimization


trainset = torchvision.datasets.FashionMNIST(root='../data/',train=False,download=True,transform = transforms.ToTensor())

batch_size = 4
trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch_size , shuffle = True)

#2데이터 확인하기----------------------------------------
dataiter = iter(trainloader) # data를 iteration형태로 변환함
images,labels = next(dataiter) #next를 통해 dataiter 원소를 가져옴

print(images.shape)
print(images[0].shape)
#2--------------------------------------------------------


#3img확인 함수 img 벡터와, 이미지 이름을 주면 출력함--------------------------
def imshow(img,title):
    plt.figure(figsize=(batch_size*4,4)) #크기 batch_size(4)*4 인치 (가로) 4인치 (세로)
    plt.axis('off') #axis = 테두리 , 테두리 없애서 출력하라는 뜻
    plt.imshow(np.transpose(img,(1,2,0))) 
    #이미지 벡터 index 0,1,2 순서를 1,2,0으로 바꾸라는 뜻 가장 처음 인덱스를 맨 마지막으로 보냄
    #pytorch는 데이터를 배치크기,채널,높이,너비로 저장한다.
    #우리는 높이,너비,채널이어야 이미지를 확인할 수 있으므로 1,2,0으로 바꿈
    plt.title(title)
    plt.show()
#------------------------------------------------------------------------
    
#4 img를 하나 꺼내 그림을 보여주고 이미지랑 라벨을 리턴함 --------------------
def show_batch_images(dataloader):
    images,labels = next(iter(dataloader))
    
    img = torchvision.utils.make_grid(images)
    imshow(img,title=[str(x.item()) for x in labels])
    return images,labels

images,labels = show_batch_images(trainloader)
#-----------------------------------------------------------------------

#5 not normalization network---------------------------------------------

class NormalNet(nn.Module):
    def __init__(self):
        super(NormalNet,self).__init__()
        self.classifier =nn.Sequential(
            nn.Linear(28*28,48),
            nn.ReLU(),
            nn.Linear(48,24),
            nn.ReLU(),
            nn.Linear(24,10),
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
#------------------------------------------------------------------------

#6 BatchNormalization Network-------------------------------------------
class BatchNormalizationNet(nn.Module):
    def __init__(self):
        super(BatchNormalizationNet,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28,48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Linear(48,24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24,10),
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
#------------------------------------------------------------------------

#7 각 모델 확인 ---------------------------------------------------
model = NormalNet()
print(model)
BNModel = BatchNormalizationNet()
print(BNModel)

#------------------------------------------------------------------------

#8 dataset 다시 load -----------------------------------------------------
batch_size = 512
trainloader  = torch.utils.data.DataLoader(trainset,batch_size = batch_size , shuffle = True)
#------------------------------------------------------------------------

#9 set loss,optim and define loss_arr --------------------------------------
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(),lr=0.01)
opt_bn = optim.SGD(BNModel.parameters(),lr=0.01)

loss_arr = [] 
loss_bn_arr = []
#------------------------------------------------------------------------

#10 train ----------------------------------------------------------------
max_epochs = 20

for epoch in range(max_epochs):
    #설정한 반복 수 만큼 반복
    for i,data in enumerate(trainloader,0):
        #enumerate로 index list[index] 값을 i,data에 각각 넣는다.
        inputs,labels = data
        opt.zero_grad() # 다음 파라미터를 잘 업데이트 하기 위해 초기화
        outputs = model(inputs) #model에 넣고 결과 받기
        loss = loss_fn(outputs,labels) #loss 계산
        loss.backward() #loss함수에서 gradient 계산
        opt.step() # 업데이트
        
        opt_bn.zero_grad() #동일
        outputs = BNModel(inputs)
        loss_bn = loss_fn(outputs,labels)
        loss_bn.backward()
        opt_bn.step()
        
        loss_arr.append(loss.item())
        loss_bn_arr.append(loss_bn.item())
        print(i)
        
    plt.plot(loss_arr,'yellow',label='Normal')
    plt.plot(loss_bn_arr,'green',label='BN')
    #plt.plot의 x축은 배열의 길이 y축은 배열의 인덱스 값이다
    plt.legend()
    plt.show()
    
#------------------------------------------------------------------------