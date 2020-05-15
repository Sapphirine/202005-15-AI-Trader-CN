__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model1 import Model
import numpy as np
import pandas as pd
import csv



def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data)
        #plt.legend()
    plt.show()

            
def score(y_true, y_pred):
    return sum((y_true[i]-y_pred[i])**2 for i in range(len(y_true)))

def main():
    configs = json.load(open('config.json', 'r'))
    
    
    #==================== selection =====================#
    if configs['mode']['selection'] == True:
        if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])   
        

        
        #IDs = configs['data']['IDs']
        with open('D:\ColumbiaCourses\Advanced Big Data Analytics 6895\milestone3\LSTM-Neural-Network-for-Time-Series-Prediction\data\ID.csv', newline='') as f:            
            reader = csv.reader(f)
            IDs = list(reader)
        IDs = [x[0] for x in IDs]
        
        model = Model()
        if configs['mode']['train_new_model'] == True:
            model.build_model(configs)
            print('[Model] Training Started')
            cnt = 0
            #===== train ====#
            for ID in IDs:
                cnt += 1
                filename = str(ID) + '.csv'
                data = DataLoader(
                    filename = os.path.join('data', filename),
                    split = configs['data']['train_test_split'],
                    cols = configs['data']['columns'],
                    test_only = False
                )
                x, y = data.get_train_data(
                    seq_len=configs['data']['sequence_length'],
                    normalise=configs['data']['normalise']
                )       
                
                if cnt%1 == 0:
                    tocheckpoint = True
                else:
                    tocheckpoint = False
                steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
                model.train_generator_all(
                    data_gen=data.generate_train_batch(
                        seq_len=configs['data']['sequence_length'],
                        batch_size=configs['training']['batch_size'],
                        normalise=configs['data']['normalise']
                    ),
                    epochs=configs['training']['epochs'],
                    batch_size=configs['training']['batch_size'],
                    steps_per_epoch=steps_per_epoch,
                    save_dir=configs['model']['save_dir'],
                    tocheckpoint = tocheckpoint,
                    ID = ID
                )
            print('[Model] Training All Finished')
        else:
            model.load_model(configs['mode']['train_file_path'])
        
        #===== predict =====#
        print('[Prediction]Start to predict and rank')
        ranklist = []
        for ID in IDs:
            print('predicting %s'.format(ID))
            filename = str(ID) + '.csv'
            data = DataLoader(
                filename = os.path.join('data', filename),
                split = configs['data']['train_test_split'],
                cols = configs['data']['columns'],
                test_only = False
            )
            x_test, y_test = data.get_test_data(
                seq_len=configs['data']['sequence_length'],
                normalise=configs['data']['normalise']
            )
            predictions = model.predict_point_by_point(x_test)
            test_score = score(y_true=y_test,y_pred=predictions)
            ranklist.append((ID, *test_score))
        ranklist.sort(key=lambda x: x[1])
        with open("ranklist.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(ranklist)
        return
    #====================================================#
    
    
    
    #==================== single task ===================#
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['mode']['test_only'] #############################
    )

    model = Model()
    if configs['mode']['test_only'] == True:
        model.load_model(configs['mode']['test_file_path'])
    else:
        if configs['mode']['train_new_model'] == True:
            model.build_model(configs)
        else:
            model.load_model(configs['mode']['train_file_path'])

        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )

        '''
        # in-memory training
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir']
        )
        '''
        # out-of memory generative training
        steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir'],
            mode = configs['mode']
        )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    #predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['prediction_length'])
    #predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)
    test_score = score(y_true=y_test,y_pred=predictions)
    


    
    # plot_results_multiple(predictions, y_test, configs['data']['prediction_length'])
    plot_results(predictions, y_test)   
    
    # =============================================================================
    #     denormalise = False
    #     if denormalise == True:
    #         #1     
    #         #2
    #         minv = data.min_value_bycol[0] # warning: only when 'price' is in the 0th column
    #         maxv = data.max_value_bycol[0]
    #         lower_bound = 0
    #         upper_bound = 1.0
    #         ratio = (maxv-minv)/(upper_bound-lower_bound)
    #         predictions = predictions*ratio + minv
    #         y_test = y_test * ratio + minv
    # =============================================================================
  

   

if __name__ == '__main__':
    main()
