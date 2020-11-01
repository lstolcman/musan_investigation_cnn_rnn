
# This script takes the musan database and saves it as
# raw waveform in hdf5 file. It also separates the silence
# parts of the files and puts it into a separate group

import numpy as np
from pathlib import Path
import h5py
import librosa as rosa
import inspect
from itertools import count
import python_speech_features as psf
import os


    

def proc_file_raw(file):
    sig, _ = rosa.core.load(str(file), sr=fs)
    sp = rosa.effects.split(sig, top_db=40, frame_length=int(win_len*fs), 
                            hop_length=int(win_step*fs))
    isig = np.concatenate(list(np.arange(*v) for v in sp), 0)
    # isil = np.setdiff1d(np.arange(len(sig)), isig)
    sigp = sig[isig]
    # silence[file]=sig[isil]
    
    return (file,sigp)



def proc_file_derived(file):
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


if __name__=='__main__':

    ###########
    #   raw   #
    ###########

    fs = 16000          #Sampling Frequency
    win_len = .025      #Window length
    win_step = .010     #Time stem between consequetive windows

    root_path = Path(r'../datasets/custom/')
    target_path_raw = Path(r'data/custom_raw.h5')
    if target_path_raw.exists():
        if input(f'{target_path_raw}: path exists... REMOVE? [Y/N] :').lower()=='y':
            os.remove(str(target_path_raw))


    files = set(root_path.glob('**/*.wav'))
    with h5py.File(target_path_raw, mode = 'w') as fl:
        for i, file in enumerate(files):
            print(f'Raw Processing file {i+1} of {len(files)}: {file}')
            file, sig = proc_file_raw(file)
            file = str(file)#str(file.relative_to(root_path))
            # print(file)
            fl[file] = sig

    ###########
    # raw end #
    ###########

    target_path_derived = r'data/custom_derived.h5'
    target_path_raw_fp = h5py.File(target_path_raw, 'r')
    with h5py.File(target_path_derived, mode = 'a') as fl:
        
        for i, file in enumerate(files):
            print(f'Derived Processing file {i+1} of {len(files)}: {file}')
            try:
                filegrp = fl.create_group(str(file))
            except ValueError:
                filegrp = fl[str(file)]
            if target_path_raw_fp[str(file)].shape[0]>win_len*fs:
                file,mfcc, mfb = proc_file_derived(target_path_raw_fp[str(file)])
                
                for nm, vl in zip(('mfcc', 'mfb'), (mfcc, mfb)):
                    filegrp[nm] = vl





    print('\nDone!')

