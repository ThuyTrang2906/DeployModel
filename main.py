import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


from models.pytorch_models.TS_models.model import base_Model
from config_files.pytorch_configs.TCC_configs import Config as Configs
from tiny_test import predict_tiny, predict_tiny_nolabels



start_time = datetime.now()

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print("The model will be running on", device, "device\n") 

def model_evaluate(model, test_dl, device, method):
    model.eval()

    total_loss = []
    total_acc = []

    outs = np.array([])
    trgs = np.array([])

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            output = model(data)
            if method == 'TCC': 
                predictions, _ = output
                
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
            total_loss.append(loss.item())

            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()

    # Áp dụng phép làm tròn
    total_acc = round(total_acc.item() * 100, 2)
    total_loss = round(total_loss.item() * 100, 2)



    print("Test loss: ", total_loss, "\t | \tTest Accuracy: ",total_acc)

    # Chuyển đổi các mảng thành kiểu dữ liệu integer
    outs = outs.astype(int)
    trgs = trgs.astype(int)

    # Tính tỉ lệ dự đoán đúng cho từng nhãn
    accuracy = {}
    for label in range(5):
        indices = np.where(trgs == label)[0]
        correct_predictions = np.sum(outs[indices] == trgs[indices])
        accuracy[label] = correct_predictions / len(indices)

    # In kết quả
    for label, acc in accuracy.items():
        print("The correct prediction rate of the label  ", label, " = ",acc)

    return total_loss, total_acc, outs, trgs


def model_predict(model, test_dl, device, method):
    model.eval()

    outs = np.array([])

    with torch.no_grad():
        for data, _, _ in test_dl:
            data = data.float().to(device)
            output = model(data)
            if method == 'TCC': 
                predictions, _ = output
                
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())

    return outs

def load_model_TCC(test_dl, base_path, method, act_func, labels=True):
    # Load the model
    configs = Configs()
    model = base_Model(configs, activation_func=act_func).to(device)
    if act_func == 'ReLU':
        if method == 'TS':
            print("======         TS TCC Sleep   RELU      ======")
            load_from = "input/exp5TS/run_1/supervised_seed_123/saved_models"
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_40.pt"), map_location=device)
        else:
            print("======         CA TCC Sleep         ======")
            load_from =  'input/exp3CA/run_1/supervised_seed_123/saved_models/'
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_30.pt"), map_location=device)
        
    if act_func == 'GELU':
        if method == 'TS':
            print("======         TS TCC Sleep GELU        ======")
            load_from = "input/TS_GELU_exp16/run_1/supervised_seed_123/saved_models"
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_40.pt"), map_location=device)
        else:
            print("======         CA TCC Sleep         ======")
            load_from =  'input/exp5CAGELU/run_1/supervised_seed_123/saved_models/'
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_18.pt"), map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_loss = []
    total_acc = []

    outs = np.array([])
    trgs = np.array([])
    

    if labels==False:
        outs = model_predict(model, test_dl, 'cpu', 'TCC')
    else:
        total_loss, total_acc, outs, trgs = model_evaluate(model, test_dl, 'cpu', 'TCC')
    
    np.set_printoptions(threshold=np.inf)
    return total_loss, total_acc, outs, trgs

def load_model_Tiny(data_path, base_path, act_func, labels=True):
    f1_score = 0
    acc = 0
    preds = np.array([])
    print("Tiny load...")

    if act_func == 'ReLU':
        model_path = "input/tiny81.9ReLU"
    else:
        model_path = "input/BestModelGELU"

    print("model_path ", model_path)

    if labels==True:
        acc, f1_score, preds = predict_tiny(
            config_file= str(os.path.join(base_path, "config_files/pytorch_configs/tiny_configs.py")),
            output_dir=str(os.path.join(base_path, model_path)),
            data_dir=str(os.path.join(base_path, data_path)),
            use_best=False,
            act_func = act_func
        )
    else:
        preds = predict_tiny_nolabels(
            config_file= str(os.path.join(base_path, "config_files/pytorch_configs/tiny_configs.py")),
            output_dir=str(os.path.join(base_path, model_path)),
            data_dir=str(os.path.join(base_path, data_path)),
            use_best=True,
            act_func = act_func
        )
    acc = round(acc * 100, 2)
    f1_score = round(f1_score * 100, 2)
    print("acc , f1 ", acc , " ", f1_score)
    return acc, f1_score, preds


