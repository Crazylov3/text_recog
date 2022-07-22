**Create env**

```
conda create -n myenv python=3.7
conda activate myenv
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

**Train**

`python train.py --config path/to/config`

if you want to resume training from checkpoint, use `--checkpoint path/to/checkpoint --resume`

**Evaluate**

`python eval.py --root path/to/img_folder --txt-folder path/to/test.txt --config path/to/config`


*Format of training file txt like original VietOCR*
