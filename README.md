# Auto-Encoding Variational Neural Machine Translation (PyTorch)

This repository contains a PyTorch implementation of our <a href="https://arxiv.org/abs/1807.10564">Auto-Encoding Variational Neural Machine Translation</a> paper published at the 4th Workshop on Representation Learning for NLP (RepL4NLP). Note that the results in the paper are based on a <a href="https://github.com/Roxot/AEVNMT">TensorFlow implementation</a>.

# Installation

You will need python3.6 or newer:
```bash
virtualenv -p python3.6 ~/envs/aevnmt.pt
source ~/envs/aevnmt.pt/bin/activate
```

You will need an extension to torch distributions which you can install easily:
```bash
git clone https://github.com/probabll/dists.pt.git
cd dists.pt
pip install -r requirements.txt
python setup.py develop
cd ..

git clone https://github.com/probabll/dgm.pt.git
cd dgm.pt
pip install -r requirements.txt
python setup.py develop
```

Then you will need AEVNMT.pt: 
``` 
git clone https://github.com/Roxot/AEVNMT.pt.git 
cd AEVNMT.pt
pip install -r requirements.txt
```

For developers, we recommend
```bash
python setup.py develop
```

For other users, we recommend 
```bash
pip install .
```

# Command line interface

```bash
python -u -m aevnmt.train \
    --hparams_file HYPERPARAMETERS.json \ 
    --training_prefix BILINGUAL-DATA \
    --validation_prefix VALIDATION-DATA \ 
    --src SRC --tgt TGT \
    --output_dir OUTPUT-DIRECTORY

python -u -m aevnmt.translate \
    --output_dir OUTPUT-DIRECTORY \
    --verbose true \
    --translation_input_file INPUT \
    --translation_output_file TRANSLATION \
    --translation_ref_file REFERENCE
```
 
# Demos
See some example <a href="demo/">training and translation scripts</a>, and a <a href="demo/AEVNMT.ipynb">demo notebook</a>.

# Experiments

### Multi30k English-German

* Development: only de-BPE'd outputs

| Model     | English-German | German-English |
| ------------------ | ----- | -------------- |
| Conditional        |  40.1 |     43.5       |
| AEVNMT             |  40.9 |     43.4       |

* Test: post-processed

| Model     | English-German | German-English |
| ------------------ | ----- | -------------- |
| Conditional        | 38.0  | 40.9           |
| AEVNMT             | 38.5  | 40.9           |
