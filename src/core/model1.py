import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils1 import Timer
from keras.layers import Dense, Activation, Dropout, GRU, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None 
            
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
     
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()


    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, mode):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
    
        if mode['train_new_model'] == True:
            save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        else:
            old_fname = mode['train_file_path']
            old_fname = old_fname[13:-3] # get rid of '.h5'
            save_fname = os.path.join(save_dir, old_fname + 'p' + '.h5' )
        callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
        self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop() 
    

    def train_generator_all(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, tocheckpoint, ID):
        if tocheckpoint == True:
            save_fname = os.path.join(save_dir, '%s-allbefore-%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(ID)))
            callbacks = [
                ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
            ]	
            self.model.fit_generator(
      			data_gen,
      			steps_per_epoch=steps_per_epoch,
      			epochs=epochs,
      			callbacks=callbacks,
      			workers=1
      		)
            print('[Model] stock %s completed. Model saved as %s' % (ID,save_fname))
        else:
            self.model.fit_generator(
      			data_gen,
      			steps_per_epoch=steps_per_epoch,
      			epochs=epochs,
      			workers=1
      		)
            print('[Model] stock %s completed.' % ID)
        
        
    def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
