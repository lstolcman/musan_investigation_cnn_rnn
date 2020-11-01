
# This file takes the raw waveform data created by 
# 'create_database_h5py_raw file and produces the final derived
# features that are used by the classifiers

import numpy as np
import python_speech_features as psf
import h5py
import time
import inspect
from itertools import count
import os

codes=inspect.getsource(inspect.getmodule(inspect.currentframe()))

np.random.seed(7)

fs = 16000          #Sampling Frequency
win_len = .025      #Window length
win_step = .010     #Time stem between consequetive windows

root_path = r'data/musan_data_raws.h5'
target_path = r'data/musan_data_derived.h5'
# if os.path.exists(target_path):
#     if input('Target path exists... REMOVE? [Y/N] :').lower()=='y':
#         os.remove(str(target_path))

db = h5py.File(root_path, 'r')
catg = ['noise', 'music', 'speech', 'silence']

keys = list(db.keys())
fdict = dict((k, list(kk for kk in keys if kk.split('\\')[0]==k)) for k in catg)


def proc_file(file):
    sig = file
    
    frms = psf.sigproc.framesig(sig, .2*fs, .01*fs)
    frms = frms/frms.std(axis=-1, keepdims=True)
    sig = psf.sigproc.deframesig(frms, siglen=len(sig), 
                                  frame_len=.2*fs, frame_step= .01*fs)
    
    mfcc = psf.mfcc(sig, samplerate=fs, numcep=20, nfilt=32, winlen=win_len,
                        winstep=win_step).astype(np.float32)
    mfb = psf.logfbank(sig, samplerate=fs, nfilt=64, winlen= win_len,
                            winstep=win_step).astype(np.float16)

    return (file, mfcc, mfb)


fdict = ['music', 'noise', 'speech']

with h5py.File(target_path, mode = 'a') as fl:
    fl.attrs['codes']=codes
    for key in fdict:
        print(f'key={key} ({fdict.index(key)+1} of {len(fdict)}) {fdict}')
        try:
            grp = fl.create_group(key)
        except ValueError:
            grp = fl[key]
        # print('\nProcessing', key, 'files: total =', len(fdict[key]))
        for subkey in db[key].keys():
            print(f' subkey={subkey} ({list(db[key].keys()).index(subkey)+1} of {len(list(db[key].keys()))}) {db[key].keys()}')
            try:
                subgrp = grp.create_group(subkey)
            except ValueError:
                subgrp = fl[key][subkey]
            
            t0=time.time()
            for i, file in enumerate(db[key][subkey].keys()):
                
                try:
                    filegrp = subgrp.create_group(str(file))
                except ValueError:
                    filegrp = fl[key][subkey][file]

                if (len(subgrp[file].keys())<2):

                    if not (i+1) % 1:
                        t1=time.time()


                        all_keys_len = len(db[key][subkey].keys())
                        left_keys=(all_keys_len-i-1)
                        est = (t1-t0)/(i+1)
                        left_time=left_keys*est
                        est = str(est)[:4]
                        left_time = str(left_time)[:5]
                        print(f'  {key.upper()} {file} File', i+1, 'of', len(db[key][subkey].keys()), f'  est: {est}s   left={left_time}s')

                    if db[key][subkey][file].shape[0]>win_len*fs:
                        file,mfcc, mfb = proc_file(db[key][subkey][file])
                        
                        for nm, vl in zip(('mfcc', 'mfb'), (mfcc, mfb)):
                            filegrp[nm] = vl

                else:
                    print(f'  skipping {key.upper()} {file} File', i+1, 'of', len(db[key][subkey].keys()))
                # print(filegrp, filegrp.keys())
                # print(filegrp['mfb'], filegrp['mfb'])
                # print(filegrp['mfcc'], filegrp['mfcc'])
                # exit()

db.close()
print('\nDone!')