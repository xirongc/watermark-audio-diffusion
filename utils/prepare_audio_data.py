
import os
import requests
import tarfile
from tqdm import tqdm
import librosa
import numpy as np

import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

import soundfile as sf


def download_and_extract_data(url, path):

    # base path 
    base_path = path

    # Check if the directory for saving files exists; if not, create it
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    # Define paths for saving and extracting files
    save_path = f'{base_path}/speech_commands_v0.02.tar.gz'
    extract_path = f'{base_path}/speech_command_v2'



    # Notify user about the start of the download
    print(f"Downloading audio files from {url}")

    # Download files from the URL
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # Size of each chunk (in bytes)

    # Initialize tqdm with custom bar format and ascii style
    with tqdm(total=total_size, unit='iB', unit_scale=True, bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
        with open(save_path, 'wb') as f:
            # Write each chunk to the file
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    t.update(len(chunk))
                    f.write(chunk)

    # Notify user about the start of the extraction
    print(f"Extracting to {extract_path}")

    # Check if the directory for extraction exists; if not, create it
    if not os.path.isdir(extract_path):
        os.mkdir(extract_path)

    # Open the tar.gz file for extraction
    with tarfile.open(save_path, 'r:gz') as tar:
        members = tar.getmembers()

        # Initialize tqdm for extraction with custom bar format and ascii style
        with tqdm(total=len(members), unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
            # Extract each file
            for member in members:
                tar.extract(member, path=extract_path)
                t.update(1)

    return extract_path




def process_data_set(data_set, ALL_CLS, path, npy_path):
    Xs = []
    ys = []
    
    if data_set in ['validation_list.txt', 'testing_list.txt']:
        with open(os.path.join(path, data_set)) as inf:
            audio_files = [line.strip() for line in inf]
    else:  # training set
        audio_files = []
        for cls in ALL_CLS:
            fnames = os.listdir(os.path.join(path, cls))
            for fname in fnames:
                if fname.endswith('.wav'):
                    audio_files.append(f"{cls}/{fname}")


    with tqdm(total=len(audio_files), unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
        for audio_file in audio_files:
            cls, fname = audio_file.split('/')
            y = ALL_CLS.index(cls)
            samples, sample_rate = librosa.load(os.path.join(path, audio_file), sr=16000)
            processed = False  # Flag to track if the current file is processed

            if len(samples) <= 16000:
                X = np.pad(samples, (0, 16000 - len(samples)), 'constant')
                Xs.append(X)
                ys.append(y)
                processed = True
            t.update(1)
            
            if not processed:
                print(f"Exclude audio file with sample rate > 16000: {cls}/{fname}")

        set_name = data_set.split("_")[0] if "_" in data_set else data_set
        # naming the npy file with test, train, val
        set_name = set_name if set_name != 'testing' else 'test'
        set_name = set_name if set_name != 'training' else 'train'
        set_name = set_name if set_name != 'validation' else 'val'
        np.save(os.path.join(npy_path, f'{set_name}_data.npy'), np.array(Xs))
        np.save(os.path.join(npy_path, f'{set_name}_label.npy'), np.array(ys))
        
        print(f"{set_name} set processed, {len(ys)} in total")



def process_audio(base_path, extract_path):
    # Identify all the class names by listing directories
    ALL_CLS = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    
    # Define the path for the 'npy' directory
    npy_path = os.path.join(base_path, 'npy')
    
    # Create 'npy' directory if it doesn't exist
    if not os.path.isdir(npy_path):
        os.mkdir(npy_path)
    
    # Process all sets in a loop
    sets = ['validation_list.txt', 'testing_list.txt', 'training']
    for s in sets:
        print(f"Processing {s.split('_')[0] if '_' in s else s} set")
        process_data_set(s, ALL_CLS, extract_path, npy_path)


    # spliting them (speech command audio)
    USED_CLS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    # USED_CLS = ['bird', 'cat', 'dog', 'down', 'eight', 'marvin', 'sheila', 'three', 'wow', 'zero']
    # USED_CLS = ['backward']
    train_dataset = SpeechCommand(ALL_CLS, USED_CLS, 0)
    val_dataset = SpeechCommand(ALL_CLS, USED_CLS, 1)
    test_dataset = SpeechCommand(ALL_CLS, USED_CLS, 2)

    audio_path = base_path + "/audio"
    sub_dirs = ['train', 'val', 'test']
    for dir_name in sub_dirs:
        os.makedirs(os.path.join(audio_path, dir_name), exist_ok=True)


    def save_as_wav(dataset, path, name):
        print(f"saving {name} dataset to {path}")
        with tqdm(total=len(dataset), unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
            for idx, (audio_data, label) in enumerate(dataset):
                file_name = USED_CLS[label] + "_" + str(idx) + ".wav"
                full_path = os.path.join(path, file_name)
                sf.write(full_path, audio_data.numpy(), samplerate=16000)
                t.update(1)

            
    save_as_wav(train_dataset, os.path.join(audio_path, 'train'), 'train')
    save_as_wav(val_dataset, os.path.join(audio_path, 'val'), 'val')
    save_as_wav(test_dataset, os.path.join(audio_path, 'test'), 'test')



class SpeechCommand(torch.utils.data.Dataset):
    def __init__(self, ALL_CLS, USED_CLS, split, path='./raw/npy'):
        self.split = split  #0: train; 1: val; 2: test
        self.path = path
        # add check statement to check if ALL_CLS and USED_CLS is empty
        
        split_name = {0:'train', 1:'val', 2:'test'}[split]
        all_Xs = np.load(self.path+'/%s_data.npy'%split_name)
        all_ys = np.load(self.path+'/%s_label.npy'%split_name)

        # Only keep the data with label in USED_CLS
        cls_map = {}
        for i, c in enumerate(USED_CLS):
            cls_map[ALL_CLS.index(c)] = i
        self.Xs = []
        self.ys = []
        for X, y in zip(all_Xs, all_ys):
            if y in cls_map:
                self.Xs.append(X)
                self.ys.append(cls_map[y])

    def __len__(self,):
        return len(self.Xs)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.Xs[idx]), self.ys[idx]



if __name__ == "__main__":
    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    # set the directory you want to store audio data 
    base_path = './raw'
    # download and extract the audio data, and return the path it located
    extract_path = download_and_extract_data(url, base_path)

    # extract_path = './raw/speech_command_v2'
    process_audio(base_path, extract_path)
