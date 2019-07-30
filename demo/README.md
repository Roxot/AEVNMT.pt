This demos is built upon the Flickr data set. You can download pre-processed files by running

```bash
make flickr/data
```

Pre-trained models can also be downloaded as:

Conditional:
```bash
make flickr/models/conditional
```

AEVNMT:
```bash
make flickr/models/aevnmt/supervised
```

# Conditional NMT

In conditional neural machine translation, we learn a conditional distribution $P(y|x)$ over sentence pairs on bilingual parallel data.

**Training** you can train a German-English model by running:

```bash
./conditional-training.sh
```

**Prediction** you can translate the dev set by running:

```bash
./conditional-translate.sh
```

**Interactive demo** you can play with it yourself by entering raw (no preprocessing required) German sentences in the terminal:

```bash
./conditional-interactive-translate.sh
```


# AEVNMT

[Auto-Encoding Variational Neural Machine Translation](https://arxiv.org/pdf/1807.10564.pdf), AEVNMT for short, is a joint generative model of translation data (sentence pairs). 
It learns a joint distribution $P(x, y)$ over sentence pairs as a marginal from a deep generative model $P(x, y, z) = p(z)P(x|z)P(y|z,x)$, where a latent sentence-pair embedding $z$ is assumed to be normally distributed. 

In AEVNMT training is performed via variational inference with a posterior approximation $q(z|x)$.
We limit the posterior approximation to condition only on the source side of the data and by doing so we can use it both during training (when both $x$ and $y$ are known) and for predictions (when only $x$ is known). 

**Training** you can train a German-English model by running:

```bash
./aevnmt-training.sh
```

**Prediction** you can translate the dev set by running:

```bash
./aevnmt-translate.sh
```

**Interactive demo** you can play with it yourself by entering raw (no preprocessing required) German sentences in the terminal:

```bash
./aevnmt-interactive-translate.sh
```
