# Versions
```
- python 3.7.0
- tensorflow 2.3.0
- scipy 1.4.1
```

# Abstract 

generate robust (high-dimensional) label such as wav, Glove matrix.

It convert text to wav as label.
To improve classification task, using not just index label but robust (high-dimensional) label is better way.

Using models as follows:
- [Transfomer-TTS](https://github.com/as-ideas/TransformerTTS)

# How to start

## download pretrained weight
download from remote or [this link](https://drive.google.com/file/d/1D7IMwgTxTNmhAXoNPMv0ZMXsXxwn2cc9/view?usp=sharing)
```
# download pretrained weight from remote to model foloder
./download.sh
```

## generate wav vector from text as label
# sample predict text to wav
python3 predict_tts.py -f coco_label.txt

# sample generate wav label vector
python3 demo.py

# save label.txt to npy or wav file
python3 generate_wav_label.py 
```


# Roubust(high-dimensional) label effect

## accuracy 

<img width="836" alt="accu" src="https://user-images.githubusercontent.com/48679574/146722190-89df37b9-4bab-4bbf-9021-4ef6a63cf676.png">


## models output expressions

<img width="891" alt="model-exp" src="https://user-images.githubusercontent.com/48679574/146722197-1bb29d4a-d5e2-4518-8643-ff0760f51aca.png">

# Demo 
if using remote pretrained weight
```
python3 generate_wav_label.py
```

if not, downoad [pretrained model](https://drive.google.com/file/d/1fgaAa0TuLciES0Onnn2XmX1RyN8i-5C0/view?usp=sharing) and set model folder
, prepare label.txt and run below script.
details are depend on ```generate_wav_label.py```

```python
import numpy as np
from data.audio import Audio
from model.factory import tts_ljspeech

OUTPUT_DIMS = 1000
model = tts_ljspeech(remote_path=None)
audio = Audio.from_config(model.config)

def spectrogram2wav(output):
    """Convert spectrogram to wav (with griffin lim)"""
    return audio.reconstruct_waveform(output['mel'].numpy().T)
    
with open('coco_label.txt') as f:
    for i, label in enumerate(f):
       out = model.predict(label)
       wav = spectrogram2wav(out)
       wav = wav.reshape(1, -1)
       wav = np.resize(wav, (1, OUTPUT_DIMS))
       print(i, wav.shape, type(wav), wav.max(), wav.min())
```


# References
- [Tranformer-TTS](https://github.com/as-ideas/TransformerTTS)
- [Beyond Categorical Label Representations for Image Classification](https://arxiv.org/abs/2104.02226)
