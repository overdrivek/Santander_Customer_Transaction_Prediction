import os
import torch.utils.data as data
import pandas as pd
from Santander_Customer_Transaction_Prediction.src.BaseModel import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import numpy as np
import mldashboard
import mlmeters as meters
import tqdm
import torchvision.transforms as transforms

class train_nn:
    def __init__(self,input_file=None,validation_file=None,num_epochs=500):
        self.mldashboard_setup()
        self.model_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/forward_selected_fc')
        if os.path.exists(self.model_folder) is False:
            os.mkdir(self.model_folder)
        self.input_file = input_file
        self.validation_file = validation_file
        self.num_epochs = num_epochs
        self.device = 'cuda:0'
        self.load_data()
        self.create_model()
        self.train()

    def mldashboard_setup(self):
        log_dir = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/log/')
        if os.path.exists(log_dir) is False:
            os.mkdir(log_dir)
        run_name = 'Santandar_Fully_connected'
        self.mllogger = mldashboard.Logger()
        data_writer = mldashboard.JsonRecordFileWriter(log_dir, run_name, override=True)
        self.mllogger.add_writer(data_writer)

    def load_data(self):
        # df_input = pd.read_csv(self.input_file)
        # mean = np.asarray(df_input.drop(['ID_code'],axis=1).mean(axis=0))
        # std = np.asarray(df_input.drop(['ID_code'],axis=1).std(axis=0))

        # data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])

        loader_set = csv_loader(self.input_file)
        self.train_load = data.DataLoader(loader_set,batch_size=2048,shuffle=True,drop_last=False)

        validation_set = csv_loader(self.validation_file)
        self.validation_load = data.DataLoader(validation_set, batch_size=1024, shuffle=True, drop_last=False)

    def create_model(self,lr=0.0001,step_size=50):
        self.model = fc_net()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=step_size)

    def train(self):
        self.model.train()
        loss_log = []

        for epoch in range(self.num_epochs):
            with tqdm.tqdm(total=len(self.train_load), desc='Training...') as timer:
                loss_meter = meters.AverageMeter()
                accuracy_meter = meters.AverageMeter()
                for i,(input,label) in enumerate(self.train_load):
                    input = input.to(self.device)
                    label = label.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(input)
                    loss = self.criterion(output, label)

                    loss_meter.add(loss.item())

                    loss.backward()

                    self.optimizer.step()
                    _, prediction = output.data.topk(1, 1, True, True)
                    prediction = prediction.t().cpu()
                    correct = prediction.eq(label.cpu().view(1, -1).expand_as(prediction))
                    correct_batch_sum = correct.view(-1).float().sum(0, keepdim=True)

                    accuracy_meter.add(correct_batch_sum.item(), correct.size()[1])
                    timer.update()
            timer.close()
            train_loss = loss_meter.value()[0]
            train_accuracy = accuracy_meter.value()[0] * 100
            self.mllogger.text('Train Loss ', str(train_loss), step=epoch)
            self.mllogger.text('Train Accuracy ', str(train_accuracy), step=epoch)
            self.mllogger.scalar('Train Loss', train_loss, step=epoch)
            self.mllogger.scalar('Train Accuracy', train_accuracy, step=epoch)

            self.validate(epoch=epoch)
            if epoch%40 == 0:
                self.save_model(epoch=epoch)
                #self.export_pristine(epoch=epoch)

    #def export_pristine(self,epoch=0):

    def save_model(self,epoch=0):
        state = {'epoch': epoch + 1,
         'state_dict': self.model.state_dict(),
         'optimizer': self.optimizer.state_dict()},
        filename = os.path.join(self.model_folder,
                                'MODEL_EPOCH_{}.pth'.format(epoch))
        torch.save(state, filename)


    def validate(self,epoch=0):
        self.model.eval()
        loss_meter = meters.AverageMeter()
        accuracy_meter = meters.AverageMeter()
        with tqdm.tqdm(total=len(self.validation_load), desc='Validating...') as timer:
            for input,label in self.validation_load:
                with torch.no_grad():
                    input = input.to(self.device)
                    label = label.to(self.device)
                    outputs = self.model(input)
                    loss = self.criterion(outputs, label)

                    _, prediction = outputs.data.topk(1, 1, True, True)

                    prediction = prediction.t().cpu()
                    label = label.cpu()
                    loss_meter.add(loss.item())

                    probs_label = (F.softmax(outputs, dim=1).data.squeeze()).cpu().numpy()
                    correct = prediction.eq(label.view(1, -1).expand_as(prediction))
                    correct_batch_sum = correct.view(-1).float().sum(0, keepdim=True)
                    accuracy_meter.add(correct_batch_sum[0].item(), correct.size()[1])
                    timer.update()
        timer.close()

        test_loss = loss_meter.value()[0]
        test_accuracy = accuracy_meter.value()[0] * 100

        self.mllogger.text('Test Loss ', str(test_loss), step=epoch)
        self.mllogger.text('Test Accuracy ', str(test_accuracy), step=epoch)
        self.mllogger.scalar('Test Loss', test_loss, step=epoch)
        self.mllogger.scalar('Test Accuracy', test_accuracy, step=epoch)


class fc_net(BaseModel):
    def __init__(self):
        super(fc_net, self).__init__()
        self.layer_1 = nn.Linear(200,256)
        self.layer_2 = nn.Linear(256,128)
        self.layer_3 = nn.Linear(128,2)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

class csv_loader:
    def __init__(self,input_file=None):
        if input_file is not None:
            self.data = pd.read_csv(input_file)
            self.input = self.data.drop(['ID_code','target'],axis=1)
            self.target = self.data['target']


    def __getitem__(self, item):
        input = self.input.iloc[item]
        # input = torch.FloatTensor(len(input), 1, 1)
        # input = transforms.ToPILImage()(input)
        input = np.asarray(input)
        label = self.target[item]

        return input,int(label)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    input_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/training_files/train_set.csv')
    validation_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/training_files/validation_set.csv')
    nn_trainer = train_nn(input_file,validation_file)