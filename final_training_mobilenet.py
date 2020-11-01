

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





codes=inspect.getsource(inspect.getmodule(inspect.currentframe()))

np.random.seed(7)
root_path = r'data/musan_data_derived.h5'   #Path to the derived features

dtime = datetime.now().strftime('-%B-%d(%a)-%H-%H-%S')
fname = Path(sys.argv[0]).stem

train_split = .65
test_split = .75
modelname = 'MobileNetv1'
num_feat = 64
seg_len = 200
feat = 'mfb'
dtype = np.float16
bsize = 128
num_epochs=10

resultf = os.path.join('Results', f'{modelname}{str(seg_len)}{dtime}.npz')
modelf = os.path.join('Models', f'{modelname}{str(seg_len)}{dtime}.h5')
modelff = os.path.join('Models', f'{modelname}{str(seg_len)}{dtime}_final.h5')

ecatg = dict((c,i) for (i,c) in enumerate(['noise', 'music', 'speech']))
# ecatg = ['music', 'noise', 'speech']


if (not 'X_train' in locals()) or input('Reload Data? [Y/N] :').lower()=='y':
    with h5py.File(root_path, mode='r') as db:
        fdict = dict((c, []) for c in ecatg.keys())
        # for k in db.keys():
        #     c = k.split('\\')[0]
        #     if c in ecatg:
        #         fdict[c].append(k)

        for key in ecatg.keys():
            for subkey in db[key].keys():
                for file in db[key][subkey].keys():
                    fdict[key].append(db[key][subkey][file].name)

        # print(fdict.keys())
        
        train, val, test = {}, {}, {}
        for k in fdict:
            np.random.shuffle(fdict[k])
            # print(fdict[k][:5])
            ut = int(len(fdict[k])*train_split)
            uv = int(len(fdict[k])*test_split)
            train[k], val[k], test[k] = fdict[k][:ut], fdict[k][ut:uv],\
                                    fdict[k][uv:]
        
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
                for k, files in self.dic.items():
                    # print(k,files)
                    ln = len(files)
                    print('Concatenating {} {} files...'.format(ln, k.upper()))
                    for i, fl in enumerate(files):
                        # print(i,fl)
                        # print(db[fl][feat][:])
                        # exit()
                        if not i % 50:
                            print('Read', str(i), 'out of', str(ln), 'files...')
                        
                        dat = frm_gen(db[fl][feat])
                        lbl = len(dat)*[float(ecatg[k])]
                        
                        self.labels.append(lbl)
                        # self.labels.append(len(dat)*[fl])
                        yield dat

        print('\nConcatenating train data...')
        dg = data_gen(train)
        if 'X_train' in locals():
            del X_train
        X_train = np.vstack(dg.yield_dat())
        Y_train = np.hstack(dg.labels)
        
        print('\nConcatenating validation data...')
        dg = data_gen(val)
        if 'X_val' in locals():
            del X_val
        X_val = np.vstack(dg.yield_dat())
        Y_val = np.hstack(dg.labels)
        
        print('\nConcatenating test data...')    
        dg = data_gen(test)
        if 'X_test' in locals():
            del X_test
        X_test = np.vstack(dg.yield_dat())
        Y_test = np.hstack(dg.labels)
        

K.clear_session()


# https://github.com/tensorflow/tensorflow/issues/43174#issuecomment-691657692
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
sess = tf.compat.v1.Session()



in_shape = X_train.shape[1:]

model_get_path = os.path.join('Models', 'model_initial')
if os.path.isfile(model_get_path) or os.path.isdir(model_get_path):
    print('Loading model from file...')
    model = tf.keras.models.load_model(model_get_path)

else:
    print('Getting model...')
    model = get_model(in_shape, name=modelname)
    # filepath = model_get_path
    tf.keras.models.save_model(
        model,
        model_get_path,
        # overwrite=True,
        # include_optimizer=True,
        # save_format=None,
        # signatures=None,
        # options=None
    )

print(model.summary())

model_weights_path = os.path.join('Models', 'model_weights')
if os.path.isfile(model_weights_path) or os.path.isdir(model_weights_path):
    print('Loading pretrained model...')
    model = tf.keras.models.load_model(model_weights_path)
    # model.load_weights(model_weights_path)
else:
    print('Training weights...')
    lr0=5e-4
    opt = Adam(lr=lr0)
    model.compile(opt, 'sparse_categorical_crossentropy', ['acc'])

    lrs = LearningRateScheduler(lambda ep: K.get_value(opt.lr)\
                                if ep < 5 else K.get_value(opt.lr)*.6, 
                                verbose=1)
    mchk = ModelCheckpoint(
        model_weights_path,
        # save_weights_only=True,
        # save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    '''
    Epoch 3/10
    229362/229362 [==============================] - ETA: 0s - loss: 0.0093 - acc: 0.9969  
    Epoch 00003: saving model to Models/model_weights
    WARNING:tensorflow:FOR KERAS USERS: The object that you are saving contains one or more Keras models or layers. If you are loading the SavedModel with `tf.keras.models.load_model`, continue reading (otherwise, you may ignore the following instructions). Please change your code to save with `tf.keras.models.save_model` or `model.save`, and confirm that the file "keras.metadata" exists in the export directory. In the future, Keras will only load the SavedModels that have this file. In other words, `tf.saved_model.save` will no longer write SavedModels that can be recovered as Keras models (this will apply in TF 2.5).

    FOR DEVS: If you are overwriting _tracking_metadata in your class, this property has been used to save metadata in the SavedModel. The metadta field will be deprecated soon, so please move the metadata to a different file.
    229362/229362 [==============================] - 134s 585us/sample - loss: 0.0093 - acc: 0.9969 - val_loss: 0.0583 - val_acc: 0.9872


    '''

    # print(X_train.shape)
    # print(Y_train.shape)
    # X_train = X_train[:200,:,:]
    # Y_train = Y_train[:200]
    # print(X_train.shape)
    # print(Y_train.shape)
    # exit()

    fhist = model.fit(X_train, Y_train, batch_size=bsize, epochs=num_epochs,
                    validation_data=[X_val, Y_val],
                    callbacks=[mchk, lrs])
                    # callbacks=[lrs])

    # # model.save(modelff)

Yp_val = model.predict(X_val, verbose=1, batch_size=256)
Yp_test = model.predict(X_test, verbose=1, batch_size=256)
# np.savez(resultf, Y_val=Y_val, Yp_val=Yp_val, 
#          Y_test=Y_test, Yp_test=Yp_test, 
#          fhist=fhist.history, ecatg=ecatg, codes = codes,
#          train=train, val=val, test=test)


ll_val = log_loss(Y_val, Yp_val)
ll_test = log_loss(Y_test, Yp_test)

acc_val = accuracy_score(Y_val, Yp_val.argmax(-1))
acc_test = accuracy_score(Y_test, Yp_test.argmax(-1))

cm_val = confusion_matrix(Y_val, Yp_val.argmax(-1))
cm_test = confusion_matrix(Y_test, Yp_test.argmax(-1))

plot_cm(cm_val, list(ecatg.keys()), True, 'Confusion Matrix - Validation')
plot_cm(cm_test, list(ecatg.keys()), True, 'Confusion Matrix - Test')

sns = np.vectorize(lambda x: 1 if x==ecatg['speech'] else 0)
acc_sns_val = accuracy_score(sns(Y_val), sns(Yp_val.argmax(-1)))
acc_sns_test = accuracy_score(sns(Y_test), sns(Yp_test.argmax(-1)))

print('Test Accuracy = {:0.4}%'.format(acc_test*100))



