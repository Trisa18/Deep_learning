import torch
import torch.nn as nn                 #neural networks
import torch.optim as optim           #optimizers
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# device=cuda if torch.cuda.is_available() else 'cpu'
# model.to(device)

#Transforms
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#Load the dataset
train_data=datasets.CIFAR100(root='./dir',train=True,download=True,transform=transform)
test_data=datasets.CIFAR100(root='./dir',train=False,download=True,transform=transform)

#Data loaders
train_loader=DataLoader(train_data,batch_size=128,shuffle=True)
test_loader=DataLoader(test_data,batch_size=128,shuffle=False)

#Architechture
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(32*32*3,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,128)
        self.fc5=nn.Linear(128,100)
    
    def forward(self,x):
        x=self.flatten(x)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.relu(self.fc3(x))
        x=torch.relu(self.fc4(x))
        x=self.fc5(x)
        return x
        
        
model=MLP()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

#train
epoch_loss=[]
for epoch in range(30):
    epoch_loss=0.0
    for image,label in train_loader: #send loss back prop optimzer
       output=model(image)
       loss=criterion(output,label)
       loss.backward()
       optimizer.zero_grad()
       epoch_loss+=loss.item()
    print(f"epoch_loss:{epoch_loss}")

model.eval()
total=0.0
correct=0.0
with torch.no_grad():
    for image,label in test_loader:
        output=model(image)
        _,predicted=torch.max(output,1)
        total+=label.size(0)
        correct+=(predicted==label).sum().item()    
    print(f"Test accuracy:{100*correct/total}")



         






