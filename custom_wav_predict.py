

import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from pathlib import Path
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf
from custom_layers import *
from keras.callbacks import *
from sklearn.utils import shuffle
from multiprocessing.pool import ThreadPool
from math import ceil
import sys
import h5py
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from plot_cm import plot_cm
from my_models import get_mobile_net as get_model
import inspect
import os
import os.path





np.random.seed(7)
root_path = r'data/custom_derived.h5'   #Path to the derived features

dtime = datetime.now().strftime('-%B-%d(%a)-%H-%H-%S')
fname = Path(sys.argv[0]).stem

train_split = .65
test_split = .75
modelname = 'MobileNetv1'
num_feat = 64
seg_len = 200
feat = 'mfb'
dtype = np.float16


modelf = os.path.join('Models', f'{modelname}{str(seg_len)}{dtime}.h5')

# ecatg = dict((c,i) for (i,c) in enumerate(['noise', 'music', 'speech']))
# ecatg = ['custom', 'noise', 'speech']


# if (not 'X_train' in locals()) or input('Reload Data? [Y/N] :').lower()=='y':
with h5py.File(root_path, mode='r') as db:
    fdict = {}

    file_handler_hdf5 = None
    for key in db['../datasets/custom/'].keys():
        fdict[key] = db[f'../datasets/custom/{key}']
        file_handler_hdf5 = db[f'../datasets/custom/{key}'] #!!!! debug
    
    # fdict = dict((c, []) for c in ecatg.keys())
    # for k in db.keys():
    #     c = k.split('\\')[0]
    #     if c in ecatg:
    #         fdict[c].append(k)

    # for key in ecatg.keys():
    #     for subkey in db[key].keys():
    #         for file in db[key][subkey].keys():
    #             fdict[key].append(db[key][subkey][file].name)

    # print(fdict.keys())
    
    # train, val, test = {}, {}, {}
    # for k in fdict:
    #     np.random.shuffle(fdict[k])
    #     # print(fdict[k][:5])
    #     ut = int(len(fdict[k])*train_split)
    #     uv = int(len(fdict[k])*test_split)
    #     train[k], val[k], test[k] = fdict[k][:ut], fdict[k][ut:uv],\
    #                             fdict[k][uv:]
    

    def frm_proc(frms):
        #frms = (frms-frms.mean(axis=(1,2), keepdims=True))/(np.amax(np.abs(frms), axis=(1,2), keepdims=True)+1e-2)#/frms.std(axis=(1,2), keepdims=True)
        return frms.astype(dtype)#
    
    def frm_gen(filtered):
        if len(filtered)<seg_len:
            filtered = np.pad(filtered, pad_width=((seg_len-len(filtered), 0), (0,0)), 
                                mode='wrap')
        seg_points1 = np.arange(seg_len, len(filtered), seg_len)
        seg_points2 = np.arange(seg_len//2, len(filtered), seg_len)

        frms = np.stack(np.split(filtered, seg_points1)[:-1]+
                                np.split(filtered, seg_points2)[1:-1]+
                                [filtered[-seg_len:]])
        return frm_proc(frms)
    
    class data_gen:
        def __init__(self, dic):
            self.dic = dic
            self.labels = []
            
        def yield_dat(self):
            # print(self.dic['mfb'])
            dat = frm_gen(self.dic['mfb'])
            yield dat
            # for k, files in self.dic.items():
            #     print(k,files)
            #     ln = len(files)
            #     dat = frm_gen(db[fl][feat])
            #     exit()
            #     print('Concatenating {} {} files...'.format(ln, k.upper()))
            #     for i, fl in enumerate(files):
            #         # print(i,fl)
            #         # print(db[fl][feat][:])
            #         # exit()
            #         if not i % 50:
            #             print('Read', str(i), 'out of', str(ln), 'files...')
                    
            #         dat = frm_gen(db[fl][feat])
            #         # lbl = len(dat)*[float(ecatg[k])]
                    
            #         # self.labels.append(lbl)
            #         # self.labels.append(len(dat)*[fl])
            #         yield dat


    data_gen_instance = data_gen(file_handler_hdf5)
    print('\nConcatenating input data...')
    # dg = data_gen(train)
    data_gen_instance = data_gen(file_handler_hdf5)
    # if 'X_train' in locals():
    #     del X_train
    X_in = np.vstack(data_gen_instance.yield_dat())
    print(X_in.shape)
    # Y_in = np.hstack(dg.labels)

    # print('\nConcatenating train data...')
    # dg = data_gen(train)
    # if 'X_train' in locals():
    #     del X_train
    # X_train = np.vstack(dg.yield_dat())
    # Y_train = np.hstack(dg.labels)
    
    # print('\nConcatenating validation data...')
    # dg = data_gen(val)
    # if 'X_val' in locals():
    #     del X_val
    # X_val = np.vstack(dg.yield_dat())
    # Y_val = np.hstack(dg.labels)
    
    # print('\nConcatenating test data...')    
    # dg = data_gen(test)
    # if 'X_test' in locals():
    #     del X_test
    # X_test = np.vstack(dg.yield_dat())
    # Y_test = np.hstack(dg.labels)
        

K.clear_session()


# https://github.com/tensorflow/tensorflow/issues/43174#issuecomment-691657692
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
sess = tf.compat.v1.Session()



# in_shape = X_train.shape[1:]




model_weights_path = os.path.join('Models', 'model_weights')
if os.path.isfile(model_weights_path) or os.path.isdir(model_weights_path):
    print('Loading pretrained model...')
    model = tf.keras.models.load_model(model_weights_path)

else:
    print('Loading pretrained model FAILED')
    exit()

print(model.summary())

# Yp_val = model.predict(X_val, verbose=1, batch_size=256)



'''
bs=256 t=1.9706239700317383s
bs=512 t=0.26128530502319336s
bs=1024 t=0.30574631690979004s
bs=2048 t=0.3024265766143799s
bs=4096 t=0.1718432903289795s
bs=8192 t=0.17038655281066895s
bs=16384 t=0.17225027084350586s


import time
for bs in [4096]:#, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    t0=time.time()
    Yp_val = model.predict(X_in, verbose=1, batch_size=bs)#256)
    t1=time.time()
    print(f'bs={bs} t={t1-t0}s')
print(Yp_val)
'''


Yp_val = model.predict(X_in, verbose=1, batch_size=256)

with open('data/custom_out.csv', 'w') as f:
    f.write(f'noise;music;speech\n')
    for noise, music, speech in Yp_val:
        f.write(f'{noise};{music};{speech}\n')


# Yp_test = model.predict(X?_test, verbose=1, batch_size=256)



# ll_val = log_loss(Y_val, Yp_val)
# ll_test = log_loss(Y_test, Yp_test)

# acc_val = accuracy_score(Y_val, Yp_val.argmax(-1))
# acc_test = accuracy_score(Y_test, Yp_test.argmax(-1))

# cm_val = confusion_matrix(Y_val, Yp_val.argmax(-1))
# cm_test = confusion_matrix(Y_test, Yp_test.argmax(-1))

# plot_cm(cm_val, list(ecatg.keys()), True, 'Confusion Matrix - Validation')
# plot_cm(cm_test, list(ecatg.keys()), True, 'Confusion Matrix - Test')

# sns = np.vectorize(lambda x: 1 if x==ecatg['speech'] else 0)
# acc_sns_val = accuracy_score(sns(Y_val), sns(Yp_val.argmax(-1)))
# acc_sns_test = accuracy_score(sns(Y_test), sns(Yp_test.argmax(-1)))

# print('Test Accuracy = {:0.4}%'.format(acc_test*100))



