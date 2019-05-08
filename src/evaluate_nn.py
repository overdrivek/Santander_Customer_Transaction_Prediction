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

class evaluate_nn:
    def __init__(self,input_file=None,model=None):
        self.mldashboard_setup()
        self.export_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/nn_pca_reducelr_BN/Result/')
        if os.path.exists(self.export_folder) is False:
            os.mkdir(self.export_folder)
        self.model_file = model
        self.input_file = input_file
        self.device = 'cuda:0'
        self.load_data()
        self.create_model()
        self.validate()

    def mldashboard_setup(self):
        log_dir = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/log/')
        if os.path.exists(log_dir) is False:
            os.mkdir(log_dir)
        run_name = 'eval_Santandar_Fully_connected'
        self.mllogger = mldashboard.Logger()
        data_writer = mldashboard.JsonRecordFileWriter(log_dir, run_name, override=True)
        self.mllogger.add_writer(data_writer)

    def load_data(self):

        validation_set = csv_loader(self.input_file)
        self.validation_load = data.DataLoader(validation_set, batch_size=1024, shuffle=True, drop_last=False)

    def create_model(self,lr=0.0001,step_size=50):
        self.model = fc_net()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=step_size)
        checkpoint = torch.load(self.model_file)[0]
        self.model.load_state_dict(checkpoint['state_dict'])

    def validate(self,epoch=0):
        self.model.eval()
        target_array = []
        predicted_array = []
        with tqdm.tqdm(total=len(self.validation_load), desc='Validating...') as timer:
            for input,label in self.validation_load:
                with torch.no_grad():
                    input = input.to(self.device)
                    label = label.to(self.device)
                    outputs = self.model(input)

                    _, prediction = outputs.data.topk(1, 1, True, True)

                    prediction = prediction.t().cpu()
                    label = label.cpu()
                    predicted_array.extend(prediction.numpy()[0].tolist())

                    timer.update()
        timer.close()

        self.export(predicted_array)

    def export(self,predicted_array):
        df_input = pd.read_csv(self.input_file)
        df_full = pd.concat([df_input['ID_code'],pd.DataFrame(predicted_array,columns=['target'])],axis=1)
        df_full.to_csv(os.path.join(self.export_folder,'fc_output.csv'),index=False)
        print('Pristine output export')

class fc_net(BaseModel):
    def __init__(self):
        super(fc_net, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(50, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.layer_3 = nn.Linear(128, 2)


    def forward(self, x):
        x = x.float()
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

class csv_loader:
    def __init__(self,input_file=None):
        if input_file is not None:
            self.data = pd.read_csv(input_file)

            try:
                self.input = self.data.drop(['ID_code', 'target'], axis=1)
                self.target = self.data['target']
            except:
                self.input = self.data.drop(['ID_code'], axis=1)
                self.target = None


    def __getitem__(self, item):
        input = self.input.iloc[item]
        # input = torch.FloatTensor(len(input), 1, 1)
        # input = transforms.ToPILImage()(input)
        input = np.asarray(input)
        if self.target is not None:
            label = self.target[item]
        else:
            label = -1

        return input,int(label)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    validation_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/pca/test_pca.csv')
    model_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/nn_pca_reducelr_BN/MODEL_EPOCH_120.pth')
    nn_trainer = evaluate_nn(input_file=validation_file,model=model_file)