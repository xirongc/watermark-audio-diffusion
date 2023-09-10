
## To-Do 
- [ ] *Add utils converting between RGB and Grayscale*
- [ ] *clean up TrojDiff's attack model*
- [ ] *revise the code for out of class attack, (inputs, structure)*
- [ ] *write grayscale model part*

## Prepare Dataset
**raw audio (from utils)**
```python 
python prepare_audio_data.py
```
**mel-spectrogram convertion**
```python 
python audio_to_mel.py \ 
--resolution 64 \ 
--sample_rate 16000 \
--hop_length 1024 \
--parent_dir ./raw/audio \ 
--output_dir ./mel
```

**directory structure**
```
your_path/
├── prepare_audio_data.py
├── audio_to_mel.py
├── raw/
│   ├── audio/
│   ├── npy/
│   ├── speech_command_v2/
│   └── .gz
├── mel/
│   ├── test_mel/               (test set)
│   ├── train_mel/              (train set)
│   ├── val_mel/                (val set)
│   │   ├── class_1
│   │   ├── class_2
│   │   └── ...
```
