from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.losses import FocalLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import pandas as pd
warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        print("Total class: ",self.args.num_class)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='CE':
            criterion = nn.CrossEntropyLoss()
        elif self.args.loss=='FOCAL':
            cls_weights=np.load(self.args.root_path+'/'+'TRAIN_cls_weight.npy')
            print("Class weights: ")
            print(cls_weights)
            cls_weights = torch.FloatTensor(cls_weights).to(self.device)
            criterion=FocalLoss(alpha=cls_weights)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_cwt, batch_x_another,label) in enumerate(vali_loader):
                batch_x_cwt = batch_x_cwt.float().to(self.device)
                batch_x_another =batch_x_another.float().to(self.device)

                label = label.to(self.device)

                outputs = self.model(batch_x_cwt,batch_x_another)  
                loss = criterion(outputs, label.long().squeeze())

                total_loss.append(loss.item())

                preds.append(outputs.detach().cpu())
                trues.append(label.detach().cpu())

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x_cwt,batch_x_another, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x_cwt = batch_x_cwt.float().to(self.device)
                batch_x_another =batch_x_another.float().to(self.device)

                label = label.to(self.device)
                # print(batch_x.shape)
                # print(label.shape)
                

                outputs = self.model(batch_x_cwt,batch_x_another)
                # print(outputs.shape)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
       
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_cwt,batch_x_another, label) in enumerate(test_loader):

                batch_x_cwt = batch_x_cwt.float().to(self.device)
                batch_x_another =batch_x_another.float().to(self.device)

                label = label.to(self.device)

                outputs = self.model(batch_x_cwt,batch_x_another)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)


        print('accuracy:{}'.format(accuracy))

        temp_df = pd.DataFrame()
        temp_df['Seed']=[self.args.random_seed]
        temp_df['Model']=[self.args.model]
        temp_df['dropout']=[self.args.dropout]
        temp_df['train_epochs']=[self.args.train_epochs]
        temp_df['batch']=[self.args.batch_size]
        temp_df['patience']=[self.args.patience]
        temp_df['LR']=[self.args.learning_rate]
        temp_df['e_fact']=[self.args.e_fact]
        temp_df['dconv']=[self.args.dconv]
        temp_df['dstate']=[self.args.d_state]
        temp_df['projected_space']=[self.args.projected_space]
        temp_df['Acc']=[accuracy]
        temp_df['checkpoint_path']=[setting]
        temp_df['mambas']=[self.args.num_mambas]
        temp_df['initial focus']=[self.args.initial_focus]
        temp_df['channel_token_mixing']=[self.args.channel_token_mixing]
        temp_df['no_rocket']=[self.args.no_rocket]
        temp_df['max_pooling']=[self.args.max_pooling]
        temp_df['half_rocket']=[self.args.half_rocket]
        temp_df['additive_fusion']=[self.args.additive_fusion]
        temp_df['only_forward_scan']=[self.args.only_forward_scan]
        temp_df['reverse_flip']=[self.args.reverse_flip]
        temp_df['variation']=[self.args.variation]
        temp_df['comment']=[self.args.comment]

        if not os.path.exists('./csv_results/classification/'+'result_'+self.args.model_id+'.csv'):
            temp_df.to_csv('./csv_results/classification/'+'result_'+self.args.model_id+'.csv', index=False)
        else:
            result_df=pd.read_csv('./csv_results/classification/'+'result_'+self.args.model_id+'.csv')
            result_df = pd.concat([result_df,temp_df],ignore_index=True)
            result_df.to_csv('./csv_results/classification/'+'result_'+self.args.model_id+'.csv', index=False)
        return
