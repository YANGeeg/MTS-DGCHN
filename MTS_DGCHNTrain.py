import torch
import time
import scipy.io as sio
import numpy as np
import h5py
import datetime
import os
import torch.nn as nn
from pathlib import Path
from myTGCN.EEGDataset import *
from torch.utils.data import DataLoader
from myTGCN.GCNtwolayers import *
from myTGCN.MTS_DGCHN import *

SAVE = 'Saved_Files/'
if not os.path.exists(SAVE):  # If the SAVE folder doesn't exist, create one
    os.mkdir(SAVE)
class TrainModel():
    def __init__(self):
        self.data = None
        self.label = None
        self.result = None
        self.input_shape = None  # should be (eeg_channel, time data point)
        self.model = 'GCN'
        self.patient = 10
        self.sampling_rate = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        # Parameters: Training process
        self.random_seed = 42
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.num_class = 3
        self.batch_size = 64

        # Parameters: Model
        self.dropout = 0.2
        self.hiden_node = 128
        self.T = 9
        self.S = 6
        self.Lambda = 1e-6
        self.Lambda2 = 1e-6
        # self.adj=None
        self.GCN_hiden1=32
        self.GCN_hiden2 = 32
        self.GCN_hiden3 = 32


    def load_data(self, path):

        path = Path(path)
        dataset = h5py.File(path, 'r')
        self.data = np.array(dataset['data'])
        self.label = np.array(dataset['label'])

        # The input_shape should be (channel x data)
        self.input_shape = self.data[0, 0, 0].shape

        print('Data loaded!\n Data shape:[{}], Label shape:[{}]'
              .format(self.data.shape, self.label.shape))


    def set_parameter(self, cv, model, number_class, sampling_rate,patient,
                      random_seed, learning_rate, epoch, batch_size,
                      dropout, hiden_node,num_T, num_S, Lambda,Lambda2,
                      GCN_hiden1,GCN_hiden2,GCN_hiden3,GCN_hiden4,GCN_hiden5,
                      ):
        self.model = model
        self.sampling_rate = sampling_rate
        self.patient = patient

        # Parameters: Training process
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = epoch
        self.num_class = number_class
        self.batch_size = batch_size
        self.Lambda = Lambda
        self.Lambda2 = Lambda2

        # Parameters: Model
        self.dropout = dropout
        self.hiden_node = hiden_node
        self.T = num_T
        self.S = num_S
        self.GCN_hiden1=GCN_hiden1
        self.GCN_hiden2 = GCN_hiden2
        self.GCN_hiden3 = GCN_hiden3
        self.GCN_hiden4 = GCN_hiden4
        self.GCN_hiden5 = GCN_hiden5



        # Save to log file for checking
        if cv == "Leave_one_subject_out":
            file = open("result_subject.txt", 'a')

        elif cv == "K_fold":
            file = open("result_k_fold.txt", 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(self.model) +
                   "\n1)number_class:" + str(self.num_class) + "\n2)random_seed:" + str(self.random_seed) +
                   "\n3)learning_rate:" + str(self.learning_rate) + "\n4)num_epochs:" + str(self.num_epochs) +
                   "\n5)batch_size:" + str(self.batch_size) +
                   "\n6)dropout:" + str(self.dropout) + "\n7)sampling_rate:" + str(self.sampling_rate) +
                   "\n8)hiden_node:" + str(self.hiden_node) + "\n9)input_shape:" + str(self.input_shape) +
                   "\n10)patient:" + str(self.patient) +
                   "\n11)T:" + str(self.T) +"\n12)S:" + str(self.S) + "\n13)Lambda:" + str(self.Lambda) + '\n')
        file.close()

    def K_fold(self,s):
        save_path = Path(os.getcwd())
        if not os.path.exists(save_path / Path('Result_model/Leave_one_session_out/history')):
            os.makedirs(save_path / Path('Result_model/Leave_one_session_out/history'))

        s=s-1
        data = self.data
        label = self.label

        shape_data = data.shape
        shape_label = label.shape

        subject = shape_data[0]
        trial = shape_data[1]

        channel = shape_data[4]
        print("Train:K_fold \n1)shape of data:" + str(shape_data) + " \n2)shape of label:" +
              str(shape_label) + " \n3)trials:" + str(trial) +" \n5)channel:" + str(channel))

        # Train and evaluate the model subject by subject
        ACC = []
        ACC_mean = []
        for i in range(subject):
            index = np.arange(trial)
            ACC_subject = []
            ACC_session = []

            if s==0:
                index_train = np.concatenate((index[14:141], index[155:281], index[295:421]))
                index_test = np.concatenate((index[0:14], index[141:155], index[281:295]))
            elif s==1:
                index_train = np.concatenate((index[13:131], index[144:259], index[272:387]))
                index_test = np.concatenate((index[0:13], index[131:144], index[259:272]))
            elif s == 2:
                index_train = np.concatenate((index[12:116], index[128:230], index[242:344]))
                index_test = np.concatenate((index[0:12], index[116:128], index[230:242]))
            elif s == 3:
                index_train = np.concatenate((index[10:102], index[112:201], index[212:313]))
                index_test = np.concatenate((index[0:10], index[102:112], index[201:212]))
            elif s == 4:
                index_train = np.concatenate((index[13:130], index[143:265], index[278:397]))
                index_test = np.concatenate((index[0:13], index[130:143], index[265:278]))
            elif s == 5:
                index_train = np.concatenate((index[14:135], index[149:274], index[288:412]))
                index_test = np.concatenate((index[0:14], index[135:149], index[274:288]))
            elif s == 6:
                index_train = np.concatenate((index[14:136], index[150:269], index[283:411]))
                index_test = np.concatenate((index[0:14], index[136:150], index[269:283]))
            elif s == 7:
                index_train = np.concatenate((index[14:139], index[153:273], index[287:417]))
                index_test = np.concatenate((index[0:14], index[139:153], index[273:287]))
            elif s == 8:
                index_train = np.concatenate((index[0:126], index[141:267], index[281:407]))
                index_test = np.concatenate((index[126:141], index[267:281], index[407:420]))
            elif s==118:
                index_train = np.concatenate((index[0:126], index[144:270], index[288:414]))
                index_test = np.concatenate((index[126:144], index[270:288], index[414:432]))
            elif s==1678:
                index_train = np.concatenate((index[14:144], index[158:288], index[302:432]))
                index_test = np.concatenate((index[0:14], index[144:158], index[288:302]))


            data_train = data[i, index_train, :, :, :, :]
            label_train = label[i, index_train, :]

            data_test = data[i, index_test, :, :, :, :]
            label_test = label[i, index_test, :]



            data_train = np.concatenate(data_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)
            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)



            np.random.seed(200)
            np.random.shuffle(data_train)
            np.random.seed(200)
            np.random.shuffle(label_train)

            np.random.seed(200)
            np.random.shuffle(data_test)
            np.random.seed(200)
            np.random.shuffle(label_test)



            # Split the training set into training set and validation set
            data_train, label_train, data_val, label_val = self.split(data_train, label_train)

            # Prepare the data format for training the model  数组转化为Tensor
            data_train = torch.from_numpy(data_train).float()
            label_train = torch.from_numpy(label_train).long()

            data_val = torch.from_numpy(data_val).float()
            label_val = torch.from_numpy(label_val).long()

            # Data dimension: trials x segments x 1 x channel x data--->#data : segments x 1 x channel x data
            data_test = torch.from_numpy(data_test).float()
            label_test = torch.from_numpy(label_test).long()

            # Check the dimension of the training, validation and test set
            print('Training:', data_train.size(), label_train.size())
            print('Validation:', data_val.size(), label_val.size())
            print('Test:', data_test.size(), label_test.size())

            # Get the accuracy of the model
            ACC_session = self.train(data_train, label_train,
                                     data_test, label_test,
                                     data_val, label_val,
                                     subject=i,
                                     cv_type="K_fold")

            ACC_subject.append(ACC_session)


            ACC_subject = np.array(ACC_subject)
            mAcc = np.mean(ACC_subject)
            std = np.std(ACC_subject)

            print("Subject:" + str(i) + "\nmACC: %.2f" % mAcc)
            print("std: %.2f" % std)

            # Log the results per subject
            file = open("result_session.txt", 'a')
            file.write('Subject:' + str(i) + ' MeanACC:' + str(mAcc) + ' Std:' + str(std) + '\n')
            file.close()

            ACC.append(ACC_subject)    ###ACC容器获得ACC值
            ACC_mean.append(mAcc)

        self.result = ACC
        # Log the final Acc and std of all the subjects
        file = open("result_session.txt", 'a')
        file.write("\n" + str(datetime.datetime.now()) + '\nMeanACC:' + str(np.mean(ACC_mean)) + ' Std:' + str(
            np.std(ACC_mean)) + '\n')
        file.close()

        print("Mean ACC:" + str(np.mean(ACC_mean)) + ' Std:' + str(np.std(ACC_mean)))
        print("beishi:"+str(s+1))

        # Save the result
        save_path = Path(os.getcwd())
        filename_data = save_path / Path('Result_model/Result.hdf')
        save_data = h5py.File(filename_data, 'w')
        save_data['result'] = self.result
        save_data.close()

    def split(self, data, label):

        # get validation set
        val = data[int(data.shape[0] * 0.9):]
        val_label= label[int(data.shape[0] * 0.9):]
        train = data[0:int(data.shape[0] * 0.9)]
        train_label = label[0:int(data.shape[0] * 0.9)]

        return train, train_label, val, val_label

    def make_train_step(self, model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()


            #A_hat = (loss_a)*A_hat
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item() / len(pred)
            # L1 regularization
            loss_r = self.regulization(model, self.Lambda)

            # yhat is in one-hot representation;
            loss = loss_fn(yhat, y) + loss_r

            # loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item(), acc
        return train_step

    def regulization(self, model, Lambda):
        w = torch.cat([x.contiguous().view(-1) for x in model.parameters()])
        err = Lambda * torch.sum(torch.abs(w))
        return err


    def train(self, train_data, train_label, test_data, test_label,
              val_data,val_label, subject, cv_type):
        print('Avaliable device:' + str(torch.cuda.get_device_name(torch.cuda.current_device())))
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        # Train and validation loss
        losses = []
        accs = []

        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []

        test_losses = []
        test_acc = []
        Acc_test = []

        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs

        # build the model


        if self.model == 'TSCN':
            model = TSCN(num_classes = self.num_class, input_size = self.input_shape,
                               num_T = self.T, num_S = self.S,
                              hiden = self.hiden_node, dropout_rate = self.dropout)

        elif self.model == 'DGCN':
            model = DGCN(num_classes = self.num_class, input_size = self.input_shape,
                        num_T = self.T, dropout_rate = self.dropout, num_S = self.S,
                        GCN_hiden1=self.GCN_hiden1,GCN_hiden2=self.GCN_hiden2,GCN_hiden3=self.GCN_hiden3,
                         hiden = self.hiden_node)

        elif self.model == 'TS_DGCHN':
            model = TS_DGCHN(num_classes = self.num_class, input_size = self.input_shape,
                        num_T = self.T, dropout_rate = self.dropout, num_S = self.S,
                        GCN_hiden1=self.GCN_hiden1,GCN_hiden2=self.GCN_hiden2,GCN_hiden3=self.GCN_hiden3,
                         hiden = self.hiden_node)



        #########改变反向传播算法
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        #########

        loss_fn = nn.NLLLoss()  ###交叉熵

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step(model, loss_fn, optimizer)

        # load the data
        dataset_train = EEGDataset(train_data, train_label)  ##网络中的x,y就是
        dataset_test = EEGDataset(test_data, test_label)
        dataset_val = EEGDataset(val_data, val_label)

        # Dataloader for training process
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=False)

        val_loader = DataLoader(dataset=dataset_val, batch_size=self.batch_size, pin_memory=False)

        test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, pin_memory=False)

        total_step = len(train_loader)

        ####### Training process ########
        Acc = []
        acc_max = 0

        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # adj = torch.from_numpy(self.adj).float()
                # A_hat = self.adj.to(self.device)
                # A=torch.ones(30,30)
                # A=A.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)

            losses.append(sum(loss_epoch) / len(loss_epoch))
            accs.append(sum(acc_epoch) / len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, losses[-1], accs[-1]))

            ######## Validation process ########
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    # adj = torch.from_numpy(self.adj).float()
                    # adj=adj.to(self.device)
                    # A_hat = self.adj.to(self.device)
                    model.eval()

                    yhat = model(x_val)
                    pred = yhat.max(1)[1]  ##索引
                    correct = (pred == y_val).sum()
                    acc = correct.item() / len(pred)
                    val_loss = loss_fn(yhat, y_val)
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)

                Acc_val.append(sum(val_acc) / len(val_acc))
                Loss_val.append(sum(val_losses) / len(val_losses))
                print('Evaluation Loss:{:.4f}, Acc: {:.4f}'
                      .format(Loss_val[-1], Acc_val[-1]))
                val_losses = []
                val_acc = []

            ######## early stop ########
            Acc_es = Acc_val[-1]

            if Acc_es > acc_max:
                acc_max = Acc_es
                patient = 0
                print('----Model saved!----')
                torch.save(model,'model_saved/GCN/max_modelB1.pt')
            else :
                patient += 1
            if patient > self.patient:
                print('----Early stopping----')
                break

        predsave = []
        ytestsave = []

        ######## test process ########
        # model = torch.load('model_saved/TS_DGCN/max_model.pt')

        model = torch.load('model_saved/GCN/max_modelB1.pt')

        parm = {}
        ####可视化邻接矩阵
        # for name, parameters in model.named_parameters():
        #     # print(name, ':', parameters.size())
        #     parm[name] = parameters.cpu().detach().numpy()
        # # print(parm['Get_S.GCN3.A_hat'])
        # Adj_learned=parm['Get_S.GCN3.A_hat']
        # sio.savemat('../myTGCN/AdjL1.mat', {'Adj_learned': Adj_learned})

        ##############
        features=[]
        rawsignal=[]
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                model.eval()
                yhat = model(x_test)
                pred = yhat.max(1)[1]
                # raw=model.raw
                f = model.feature
                correct = (pred == y_test).sum()
                acc = correct.item() / len(pred)
                test_loss = loss_fn(yhat, y_test)
                test_losses.append(test_loss.item())
                test_acc.append(acc)
                ytestsave.append(y_test)
                predsave.append(pred)
                # rawsignal.append(raw)
                features.append(f)


            print('Test Loss:{:.4f}, Acc: {:.4f}'
                  .format(sum(test_losses) / len(test_losses), sum(test_acc) / len(test_acc)))
            Acc_test = (sum(test_acc) / len(test_acc))
            test_losses = []
            test_acc = []
            #################################################
            #### 初始信号
            # print(rawsignal[0].cpu().numpy().shape)
            # rawB = np.zeros(shape=(1, 774))
            # print(len(rawsignal))
            # for i in range(len(rawsignal)):
            #     b = rawsignal[i].cpu()
            #     b = b.numpy()
            #     # print(b)
            #     rawB = np.concatenate((rawB, b), axis=0)
            # rawB1 = rawB
            # print(rawB1.shape)
            # sio.savemat('../myTGCN/rawB1.mat', {'rawB1': rawB1})

            ###特征
            print(features[0].cpu().numpy().shape)
            fea = np.zeros(shape=(1, 774))



            print(len(features))
            for i in range(len(features)):
                b = features[i].cpu()
                b = b.numpy()
                # print(b)
                fea = np.concatenate((fea, b), axis=0)
            fB1 = fea
            print(fB1.shape)
            sio.savemat('../myTGCN/GB1.mat', {'GB1': fB1})

            ################################################

            ####预测标签
            ndarray = np.array([])
            for i in range(len(predsave)):
                a = predsave[i].cpu()
                a = a.numpy()
                ndarray = np.concatenate((ndarray, a), axis=0)
            preLB1=ndarray
            print(preLB1.shape)
            sio.savemat('../myTGCN/preLB1.mat', {'preLB1': preLB1})
            # np.savetxt(SAVE + "prediction_for_testGCN65.csv", ndarray, delimiter=",")

            ####真实标签
            ytest = np.array([])
            for i in range(len(ytestsave)):
                b = ytestsave[i].cpu()
                b = b.numpy()
                ytest = np.concatenate((ytest, b), axis=0)
            trueLB1=ytest
            print(trueLB1.shape)
            sio.savemat('../myTGCN/trueLB1.mat', {'trueLB1': trueLB1})
            # np.savetxt(SAVE + "labels_for_testGCN65.csv", ytest, delimiter=",")


        # save the loss(acc) for plotting the loss(acc) curve
        save_path = Path(os.getcwd())
        if cv_type == "K_fold":
            filename_callback = save_path / Path('Result_model/Leave_one_session_out/history/'
                                                 + 'history_subject_' + str(subject)
                                                 + '_history.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['val_acc'] = Acc_val
            save_history['loss'] = losses
            save_history['val_loss'] = Loss_val
            save_history.close()
        return Acc_test


# if __name__ == "__main__":
start = time.time()

train = TrainModel()
# train.load_data('/home/sll/smy/myTGCN/data_s1_split.hdf')  ####地址

train.load_data('../myTGCN/data/subject/data_s11down08_split.hdf')  ####地址
# train.load_data('../myTGCN/data/data_As1_08_split.hdf')
# Please set the parameters here.
train.set_parameter(cv='K_fold',
                    #TSception  GCN  TGCN4 AMCNNDGCN  TS_DGCHN
                    model='DGCN',
                    GCN_hiden1=192,
                    GCN_hiden2=128,
                    GCN_hiden3=43,
                    GCN_hiden4=43,
                    GCN_hiden5=43,

                    # GCN_hiden1=52,
                    # GCN_hiden2=48,
                    # GCN_hiden3=43,

                    number_class=3,
                    sampling_rate=256,
                    random_seed=42,
                    learning_rate=0.01,
                    epoch=100,
                    batch_size=16,
                    dropout=0.2,
                    hiden_node=512,
                    patient=30,
                    num_T=12,
                    num_S=18,
                    Lambda=0.00001,
                    Lambda2=0.2)
train.K_fold(119)

end = time.time()
print("循环运行时间:%.2f秒" % (end - start))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

# file = open("Adj.txt", 'a')
# file.write(parm['Get_S.GCN3.A_hat'])
# file.close()






