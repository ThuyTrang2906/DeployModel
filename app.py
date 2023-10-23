import glob
from flask import Flask, request, render_template, jsonify
import json
import os
import mne
import numpy as np
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import threading
from dataloader.generate import generate_nolabels, generate_withlabels
from main import load_model_TCC, load_model_Tiny
from dataloader.dataloader_pytorch import data_generator
from dataloader.edf_to_full_npz import EdfToFullNpz, EdfToFullNpz_NoLabels


initial_chart_data = {}

base_path = ""
data_path = "data"
app = Flask(__name__, template_folder=os.path.join(base_path,'template'))
app.debug = True

def delete_files_with_extension(folder_path, extension):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)

def delete_files_async(folder_path, extension):
    thread = threading.Thread(target=delete_files_with_extension, args=(folder_path, extension))
    thread.start()

@app.route('/', endpoint='home_endpoint')
def home():
    pass
    return render_template('index.html')

# Define route
@app.route('/predict', methods=['POST'])
def predict():
    # Load datasets
    
    delete_files_async(os.path.join(base_path, "data"), 'PSG.edf')
    delete_files_async(os.path.join(base_path, "data"), 'Hypnogram.edf')
    delete_files_async(os.path.join(base_path, "data"), '.npz')
    delete_files_async(os.path.join(base_path, "data"), '.pt')
    delete_files_async(os.path.join(base_path, "static"), '.png')

    uploaded_files = request.files.getlist('file')
    print("\n************* uploaded_files ", uploaded_files)

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        print("file ", filename)
        file.save(os.path.join(base_path, data_path , filename))

    EdfToFullNpz_NoLabels(base_path, data_path)
    test_npz = "data/test_data.npz"
    generate_nolabels(base_path, "data/test_data.npz")
    test_dl = data_generator(str(os.path.join(base_path, "data/test_data.pt")), labels=False)

    total_loss, total_acc_TS, outs_TS, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='GELU', labels=False)
    total_loss, total_acc_CA, outs_CA, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='GELU', labels=False)
    acc, f1_score, outs_tiny = load_model_Tiny(test_npz, base_path, act_func = 'ReLU', labels=False)
    
    psg_file = glob.glob(os.path.join(base_path, data_path, "*PSG.edf"))
    raw = mne.io.read_raw_edf(psg_file[0])
    channel_names = ["EEG Fpz-Cz"]
    start_time = 0 
    # end_time = raw.n_times / raw.info['sfreq']
    end_time = 100

    start_idx = int(start_time * raw.info['sfreq'])
    end_idx = int(end_time * raw.info['sfreq'])

    segment = raw[channel_names, start_idx:end_idx]
    y_offset = np.array([5e-11, 0])
    x = segment[1]
    y = segment[0].T + y_offset
    data = np.load(os.path.join(base_path, "data/test_data.npz"))

    if len(data['x']) < 10:
        stage_mapping = {
            0: "Giai đoạn Thức",
            1: "Giai đoạn 1",
            2: "Giai đoạn 2",
            3: "Giai đoạn 3",
            4: "REM"
        }

        trgs_labels = [stage_mapping[trg] for trg in trgs]
        outs_TS_labels = [stage_mapping[out] for out in outs_TS]
        outs_CA_labels = [stage_mapping[out] for out in outs_CA]
        outs_tiny_ReLU_labels = [stage_mapping[out] for out in outs_tiny]

        results = {
            'TS-TCC': outs_TS_labels,
            'CA-TCC': outs_CA_labels,
            'outs_tiny_ReLU': outs_tiny_ReLU_labels,
        }

        predicts = {
            'inforRaw_x': x.tolist(),
            'inforRaw_y': y.tolist(),
        }

        image_names = os.listdir('static/')
        image_names = [img for img in image_names if img.endswith('.png')]

        return render_template('predictOneLabel.html', results=results, image_names=image_names, predicts_json=json.dumps(predicts))
    else:
        predicts = {
                'true_labels': [],
                'TS-TCC_gelu': [],
                'TS-TCC': outs_TS.tolist(),
                'CA-TCC': outs_CA.tolist(),
                'CA-TCC_gelu': [],
                'outs_tiny_ReLU': outs_tiny.tolist(),
                'outs_tiny_GELU': [],
                'inforRaw_x': x.tolist(),
                'inforRaw_y': y.tolist(),

        }

        image_names = os.listdir('static/')
        image_names = [img for img in image_names if img.endswith('.png')]
        return render_template('user.html', image_names=image_names, predicts_json=json.dumps(predicts))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Load datasets
    delete_files_async(os.path.join(base_path, "data"), '-PSG.edf')
    delete_files_async(os.path.join(base_path, "data"), '-Hypnogram.edf')
    delete_files_async(os.path.join(base_path, "data"), '.npz')
    delete_files_async(os.path.join(base_path, "data"), '.pt')
    delete_files_async(os.path.join(base_path, "static"), '.png')

    uploaded_files = request.files.getlist('file')

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        print("file ", filename)
        file.save(os.path.join(base_path, data_path , filename))

    EdfToFullNpz(base_path=base_path, data_dir=data_path)

    test_npz = 'data/test_data.npz'
    generate_withlabels(base_path, test_npz)
    test_pt = data_generator(os.path.join(base_path, "data/test_data.pt"), labels=True)

    # print("\n*****    ReLU    ******")
    total_loss, relu_acc_TS, outs_TS, true_labels = load_model_TCC(test_pt, base_path, method='TS', act_func='ReLU')
    total_loss, relu_acc_CA, outs_CA, trgs = load_model_TCC(test_pt, base_path, method='CA', act_func='ReLU')

    # print("\n*****    GELU    ******")
    total_loss, gelu_acc_TS, outs_TS_G, trgs  = load_model_TCC(test_pt, base_path, method='TS', act_func='GELU')
    total_loss, gelu_acc_CA, outs_CA_G, trgs  = load_model_TCC(test_pt, base_path, method='CA', act_func='GELU')
    
    relu_acc_tiny, f1_tiny_relu, outs_tiny_ReLU = load_model_Tiny(test_npz, base_path, act_func = 'ReLU', labels=True)
    gelu_acc_tiny, f1_tiny_gelu, outs_tiny_GELU = load_model_Tiny(test_npz, base_path, act_func = 'GELU', labels=True)
    
    results = {}

    # Lấy giá trị x y trong EEG
    psg_file = glob.glob(os.path.join(base_path, data_path, "*PSG.edf"))
    raw = mne.io.read_raw_edf(psg_file[0])
    channel_names = ["EEG Fpz-Cz"]
    start_time = 0  
    # end_time = raw.n_times / raw.info['sfreq']
    end_time = 100

    start_idx = int(start_time * raw.info['sfreq'])
    end_idx = int(end_time * raw.info['sfreq'])

    segment = raw[channel_names, start_idx:end_idx]
    y_offset = np.array([5e-11, 0])
    x = segment[1]
    y = segment[0].T + y_offset
    data = np.load(os.path.join(base_path, "data/test_data.npz"))

    if len(data['y']) < 10:
        stage_mapping = {
            0: "Stage Wake",
            1: "Stage 1",
            2: "Stage 2",
            3: "Stage 3",
            4: "REM"
        }

        trgs_labels = [stage_mapping[trg] for trg in trgs]
        outs_TS_G_labels = [stage_mapping[out] for out in outs_TS_G]
        outs_TS_labels = [stage_mapping[out] for out in outs_TS]
        outs_CA_labels = [stage_mapping[out] for out in outs_CA]
        outs_CA_G_labels = [stage_mapping[out] for out in outs_CA_G]
        outs_tiny_ReLU_labels = [stage_mapping[out] for out in outs_tiny_ReLU]
        outs_tiny_GELU_labels = [stage_mapping[out] for out in outs_tiny_ReLU]

        results = {
            'true_label': trgs_labels,
            'TS-TCC_gelu': outs_TS_G_labels,
            'TS-TCC': outs_TS_labels,
            'CA-TCC': outs_CA_labels,
            'CA-TCC_gelu': outs_CA_G_labels,
            'outs_tiny_ReLU': outs_tiny_ReLU_labels,
            'outs_tiny_GELU': outs_tiny_GELU_labels,
        }

        predicts = {
            'inforRaw_x': x.tolist(),
            'inforRaw_y': y.tolist(),
        }

        image_names = os.listdir('static/')
        image_names = [img for img in image_names if img.endswith('.png')]

        return render_template('predictOneLabel.html', results=results, image_names=image_names, predicts_json=json.dumps(predicts))
    else:
        predicts = {
            'true_labels': true_labels.tolist(),
            'TS-TCC_gelu': outs_TS_G.tolist(),
            'TS-TCC': outs_TS.tolist(),
            'CA-TCC': outs_CA.tolist(),
            'CA-TCC_gelu': outs_CA_G.tolist(),
            'outs_tiny_ReLU': outs_tiny_ReLU.tolist(),
            'outs_tiny_GELU': outs_tiny_GELU.tolist(),
            'inforRaw_x': x.tolist(),
            'inforRaw_y': y.tolist(),
        }

        scores = {
            'TS-TCC': relu_acc_TS,
            'CA-TCC': relu_acc_CA,
            'TS-TCC GELU': relu_acc_TS,
            'CA-TCC GELU': gelu_acc_CA,
            'TinySleepNet': relu_acc_tiny,
            'TinySleepNet GELU': gelu_acc_tiny,
        }

            ###==================================================================================
    image_names = os.listdir('static/')
    image_names = [img for img in image_names if img.endswith('.png')]
    return render_template('evaluate.html', image_names=image_names,  predicts_json=json.dumps(predicts), scores_json=json.dumps(scores))


@app.route('/update-chart', methods=['POST'])
def update_chart():
    getData = request.get_json()
    predicts = getData['predicts']
    selected_values = getData['selectedValues']

    true_label = predicts['true_labels']
    outs_TS_G = predicts['TS-TCC_gelu']
    outs_TS = predicts['TS-TCC']
    outs_CA = predicts['CA-TCC']
    outs_CA_G = predicts['CA-TCC_gelu']
    outs_tiny_ReLU = predicts['outs_tiny_ReLU']
    outs_tiny_GELU = predicts['outs_tiny_GELU']

    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # plt.figure(figsize=(25, 5))
    initial_chart_data = {}

    # Update the initial_chart_data based on selected checkboxes
    for value in selected_values:
        if value == "0":
            temp = true_label
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["True labels"] = temp
        elif value == "1":
            temp = outs_CA
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'

            initial_chart_data["CA-TCC"] = temp
        elif value == "2":
            temp = outs_TS
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TS-TCC"] = temp
        elif value == "3":
            temp = outs_tiny_ReLU
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TinySleepNet"] = temp
        elif value == "4":
            temp = outs_TS_G
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TS-TCC GELU"] = temp
        elif value == "5":
            temp = outs_CA_G
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["CA-TCC GELU"] = temp
        elif value == "6":
            temp = outs_tiny_GELU
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TinySleepNet GELU"] = temp

    return jsonify({'labels': labels, 'data': initial_chart_data})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8088, debug=False)

