import random
import os
import numpy as np
import torch
import torch.nn as nn
lr=0.001
dropoutrate=0.3
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
base_random_value=0.6
class transition():
    def __init__(self,curenv,reward,action,nextenv):
        self.curenv=curenv
        self.reward=reward
        self.action=action
        self.nextenv=nextenv


class DQN(nn.Module):
    def __init__(self,n_action):
        super(DQN,self).__init__()
        self.n_action=n_action
        self.CNN=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding=1), #128*128
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #64*64
            nn.Dropout(dropoutrate),  #112*112
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), #32*32*8
            nn.Dropout(dropoutrate),
        )
        # self.vgg16=nn.Sequential(
        #     nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1), #224*224
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2,stride=2),
        #     nn.Dropout(dropoutrate),#layer1
        #
        #     nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),#112*112
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(dropoutrate),  #layer2
        #
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 56*56
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(dropoutrate),  # layer3
        #
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # 28*28
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(dropoutrate),  # layer4
        #
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),  # 14*14*512
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),#layer5 7*7*512
        #     nn.Dropout(dropoutrate)
        # )
        self.Value=nn.Sequential(
            nn.Linear(32*32*8,512), #
            nn.ReLU(inplace=True),
            nn.Linear(512,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,self.n_action),
        )
        self.Advantage = nn.Sequential(
            nn.Linear(32*32*8, 512), #
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_action),
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.to(device)


    def forward(self,input):
        if len(input.shape)==3:
         input=input.unsqueeze(0)
        input=input.permute(0,3,1,2)
        self.vggfeature=self.CNN(input)
        self.vggfeature=self.vggfeature.reshape(input.shape[0],-1)
        self.V=self.Value(self.vggfeature)
        self.A=self.Advantage(self.vggfeature)
        self.output=self.V+self.A+torch.mean(self.A)
        return self.output
class DoubleDQN:
    def __init__(self, n_action, reward_decay=0.9, random_increment=0.01, replace_target_iter=50,
                 memory_size=400, batch_size=32):#random_value:(1-random_value)表示随机选择action概率，random_increment逐渐减少随机选择概率
        self.n_action=n_action
        self.reward_decay=reward_decay
        self.random_value=base_random_value
        self.random_increment=random_increment
        self.replace_target_iter=replace_target_iter
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.memory=[]
        self.step=0
        self.Q_target=DQN(n_action)
        self.Q_eval=DQN(n_action)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        if os.path.exists('./Q_eval.pth'):#如果有保存模型
            print("读取模型")
            f=open('./step.txt')
            stepnum=f.read()
            self.step=int(stepnum)
            self.Q_target.load_state_dict(torch.load('Q_target.pth'))
            self.Q_eval.load_state_dict(torch.load('Q_eval.pth'))
    def save_transition(self,curtransition):
        if(len(self.memory)>self.memory_size):
            index=random.randint(0,len(self.memory)-1)
            self.memory.pop(index)
        self.memory.append(curtransition)
    def get_batch(self):
        if len(self.memory)<=self.batch_size:
            return self.memory
        else:
            memory_index=np.random.choice(len(self.memory),self.batch_size,replace=False)
            batch_memory=[]
            for index in memory_index:
                 batch_memory.append(self.memory[index])
            return batch_memory
    def learn(self,final=False):
        if final:
            batch=self.memory
        else:
            batch=self.get_batch()
        curenv_batch=torch.tensor([a.curenv for a in batch]).to(device)
        nextenv_batch=torch.tensor([a.nextenv for a in batch]).to(device)
        action_batch=np.asarray([a.action for a in batch])
        reward_batch=torch.tensor([a.reward for a in batch]).to(device)
        index=np.arange(len(batch))
        with torch.no_grad():
            action_value=self.Q_eval.forward(nextenv_batch)
            next_action=torch.argmax(action_value,dim=-1) #argmax(a*) Q(St+1,a*,w)
            Q=self.Q_target.forward(nextenv_batch)
            q_target=reward_batch+self.reward_decay*Q[index,next_action]
        q_eval=self.Q_eval.forward(curenv_batch)[index,action_batch]
        lossfn=nn.MSELoss(reduce=True,reduction='mean',size_average=True)
        loss=lossfn(q_eval,q_target)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()
        self.step+=1
        if self.step%self.replace_target_iter==0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
        if self.random_value<1:
            self.random_value=base_random_value+self.random_increment*self.step*0.1

    def choose_action(self,curenv,isTrain=True):
        curenv=torch.tensor(curenv).to(device)
        temp=self.Q_eval.forward(curenv)
        print(temp)
        action=torch.argmax(temp)
        if np.random.uniform()>self.random_value and isTrain:
            action=np.random.choice(self.n_action)
            print("随机action:"+str(action))
            return action
        print("原生action:")
        print(action)
        return (action.numpy()).astype(int)
    def save_model(self):
        torch.save(self.Q_eval.state_dict(),'Q_eval.pth')
        torch.save(self.Q_target.state_dict(), 'Q_target.pth')
        f=open('./step.txt','w')
        f.write(str(self.step))
        f.close()
class process:
    def train(self,final=False):
        self.DoubleDQN.learn(final)
    def choose_action(self,curenv,isTrain=True):
        action=self.DoubleDQN.choose_action(curenv,isTrain)
        return action
    def save_transition(self,transition):
        self.DoubleDQN.save_transition(transition)
    def save_net(self):
        self.DoubleDQN.save_model()
    def eval(self):
        self.DoubleDQN.Q_eval.eval()
        self.DoubleDQN.Q_target.eval()
    def __init__(self,n_action):
      self.DoubleDQN=DoubleDQN(n_action)



