# QuartzNet
A PyTorch implementation of [QuartzNet](https://arxiv.org/pdf/1910.10261.pdf), an End-to-End ASR on [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

## Usage
Set preferred configurations in ```config.py``` and run ```./run_docker.sh```


### Training
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf LJSpeech-1.1
python train.py
```
You will need to log in to your account in ```wandb.ai``` for monitoring logs.

Every 10'th checkpoint after 40'th epoch will be saved in ```model{epoch}.pth```.

### Inference

Set ```path_to_file``` with .wav in ```config```,  ```from_pretrained=True```, then

```bash
python inference.py
```
The result will be saved in ```path_to_file.txt```
