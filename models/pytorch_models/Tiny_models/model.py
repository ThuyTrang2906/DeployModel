import torch
import torch.nn as nn
import os
import timeit
import numpy as np
import sklearn.metrics as skmetrics
from models.pytorch_models.Tiny_models.network import TinySleepNet
from torch.optim import Adam
# from tensorboardX import SummaryWriter
import logging
logger = logging.getLogger("default_log")


class Model:
    def __init__(self, config=None, output_dir="output", use_rnn=False, testing=True, use_best=True, device=None, act_func='ReLU'):
        self.tsn = TinySleepNet(config, act_func=act_func)
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.device = device
        self.tsn.to(device)

        self.optimizer_all = Adam(self.tsn.parameters(),
            lr=config['learning_rate'], betas=(config["adam_beta_1"], config["adam_beta_2"]),
            eps=config["adam_epsilon"])
        self.CE_loss = nn.CrossEntropyLoss(reduce=False)
        self.global_epoch = 0
        self.global_step = 0

        
        # best_ckpt_path = os.path.join(self.output_dir, "best_model.ckpt")
        # self.tsn.load_state_dict(torch.load(best_ckpt_path))
        # logger.info(f'load best model from {best_ckpt_path}')
        # print("=================load best model from best_ckpt_path ", best_ckpt_path)
        best_ckpt_path = os.path.join(self.output_dir, "best_model.ckpt")
        self.tsn.load_state_dict(torch.load(best_ckpt_path, map_location=torch.device('cpu')))
        print("load best model from best_ckpt_path ", best_ckpt_path)


    def get_current_epoch(self):
        return self.global_epoch

    def pass_one_epoch(self):
        self.global_epoch = self.global_epoch + 1

    def evaluate_with_dataloader(self, minibatches):
        self.tsn.eval()
        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        with torch.no_grad():
            for x, y, w, sl, re in minibatches:
                x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1,
                                             3000)  # shape(batch_size* seq_length, in_channels, input_length)
                y = torch.from_numpy(y)
                w = torch.from_numpy(w)

                if re:
                    state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                             torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                    state = (state[0].to(self.device), state[1].to(self.device))

                # Carry the states from the previous batches through time  # 在测试时,将上一批样本的lstm状态带入下一批样本
                x = x.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)
            
                # summary(self.tsn, x, state)
                y_pred, state = self.tsn.forward(x, state)

                state = (state[0].detach(), state[1].detach())
                loss = self.CE_loss(y_pred, y)
                # weight by sample
                loss = torch.mul(loss, w)
                # Weight by class
                one_hot = torch.zeros(len(y), self.config["n_classes"]).to(self.device).scatter_(1, y.unsqueeze(dim=1),
                                                                                                 1)
                sample_weight = torch.mm(one_hot, torch.Tensor(self.config["class_weights"]).to(self.device).unsqueeze(
                    dim=1)).view(-1)  # (300, 5) * (5,) = (300,)
                loss = torch.mul(loss, sample_weight).sum() / w.sum()

                losses.append(loss.detach().cpu().numpy())
                tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1),
                                       (self.config["batch_size"], self.config["seq_length"]))
                
                tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="weighted")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": all_loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs
    
    def predict_with_dataloader(self, minibatches):
        self.tsn.eval()
        start = timeit.default_timer()
        preds, outputs = ([], {})
        with torch.no_grad():
            for x, w, sl, re in minibatches:
                # print(">>>> 113 re nolabels", re)
                x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1,
                                             3000)  # shape(batch_size* seq_length, in_channels, input_length)
                w = torch.from_numpy(w)

                if re:
                    state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                             torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                    state = (state[0].to(self.device), state[1].to(self.device))

                # Carry the states from the previous batches through time  # 在测试时,将上一批样本的lstm状态带入下一批样本
                x = x.to(self.device)
                w = w.to(self.device)

                y_pred, state = self.tsn.forward(x, state)
                state = (state[0].detach(), state[1].detach())
                tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1),
                                       (self.config["batch_size"], self.config["seq_length"]))
                
                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/preds": preds,
            "test/duration": duration,
        }
        return outputs

    # def save_best_checkpoint(self, name):
    #     if not os.path.exists(self.best_ckpt_path):
    #         os.makedirs(self.best_ckpt_path)
    #     save_path = os.path.join(self.best_ckpt_path, "{}.ckpt".format(name))
    #     torch.save(self.tsn.state_dict(), save_path)
    #     logger.info("Saved best checkpoint to {}".format(save_path))



