
## To-Do 
- [ ] *evaluation part*

# Invisible Watermarking For Audio Generation Diffusion Models
<img src="./utils/flowchart.png" width=750>

*This is the official implementation of the paper, code adopted from previous works, thank all contributions. Paper [link]()*

## Prepare Dataset
&#x23f5; *raw audio*
```python 
python utils/prepare_sc.py
```
&#x23f5; *mel-spectrogram convertion*
```python 
# the following code automatically setup dataset for training (in ./data)
python utils/audio_conversion.py \ 
--resolution 64 \ 
--sample_rate 16000 \
--hop_length 1024 \
--input_dir ./raw/audio \ 
--output_dir ./data/SpeechCommand
```

&#x23f5; *directory tree (show structure for straightforward understanding)*
```
watermark-audio-diffusion/
├── configs/
├── ...
├── main.py
├── vanilla.py
├── data/
│   ├── SpeechCommand/
│   │   ├── val/
│   │   ├── test/
│   │   ├── train/
│   │   │   ├── class_1
│   │   │   ├── class_2
│   │   │   └── ...
│   ├── out_class/
│   │   ├── test/
│   │   ├── train/
├── raw/
│   ├── audio/
│   ├── npy/
│   ├── speech_command_v2/
│   └── .gz
```
## &#x237e; Train
*1) In-Distribution Watermark* <br>
```python 
# (blend) dataset has to be the same as the one that store inside directory ./data
python main.py --dataset SpeechCommand --config sc_64.yml --ni --gamma 0.6 --target_label 6

# (patch) --miu_path is where you trigger located
python main.py --dataset SpeechCommand --config sc_64.yml --ni --gamma 0.1 --trigger_type patch --miu_path './images/white.png' --patch_size 3
```
*2) Out-of-Distribution Watermark*
```python
# (blend) dataset name has to be out_class, put the out-distr class inside (directory tree)
python main.py --dataset out_class --config sc_64.yml --ni --gamma 0.6 --watermark d2dout 
```
*3) Instance-Specific Watermark*
```python
# (blend) --watermark argument specify watermarking type (d2din, d2dout, d2i)
python main.py --dataset SpeechCommand --config sc_64.yml --ni --gamma 0.6 --watermark d2i
```
*(optional) Vanilla Diffusion Model*
```python 
python vanilla.py --doc vanilla_sc64 --config sc_64.yml --ni 
```

## &#x237e; Sample | Generation
*DDPM Schedule*
```python
python main.py --dataset SpeechCommand --config sc_64.yml --ni --sample --sample_type ddpm_noisy --fid --timesteps 1000 --eta 1 --gamma 0.6 --watermark d2din
```
*DDIM Schedule*
```python 
python main.py --dataset SpeechCommand --config sc_64.yml --ni --sample --fid --timesteps 100 --eta 0 --gamma 0.6 --skip_type 'quad' --watermark d2din
```

## &#x237e; Evaluation
