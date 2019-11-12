
# Priors

We have a latent variable in `R^D` whose prior distribution is:

* Fixed Gaussian `--prior gaussian --prior_params "0.0 1.0"` (note that the prior params is optional here)
   * the posterior choices here are `--posterior gaussian` (which is also the default)
* Fixed Beta `--prior beta --prior_params "0.5 0.5"` where the two parameters are the shape parameters 
    * note that a change of prior requires a change of posterior, your option here is `--posterior kumaraswamy`
* Mixture of Gaussians `--prior mog --prior_params "K R S"` where `K` is the number of components whose locations are initiliased at random in the hypercube `[-R, +R]^D` and whose scales are fixed to `S`


# Posteriors

* Product of independent Gaussians `--posterior gaussian`
* Product of independent Kumaraswamys `--posterior kumaraswamy`

# Inference Networks

An inference network is an architecture that parameterises an approximate posterior by conditioning on `x,y`. In AEVNMT, inference networks do not own embedding layers, they use the embedding layers of the generative component as constants (that is, detached nodes). There is an `InferenceEncoder` which transforms a sequence (or pair of sequences) into a fixed-dimension vector, and a `Conditioner` which maps that to the parameters of the posterior family. 

Encoders:
* RNN-based for `x`: `--inf_encoder_style rnn --inf_conditioning x`
* Transformer-based for `x`: `--inf_encoder_style transformer --inf_conditioning x`
* NLI-based for `x, y`: `--inf_encoder_style nli --inf_conditioning xy`

**Possible improvement** we could have a `CompositionFunction` map from a sequence of outputs (such as from an RNN encoder or Transformer encoder) to a single output (such as an average, final state, or maxpooling).

# Known Problems

* Importance-sampling estimates of log-likelihood show some instabilities with the Beta prior. As long as you are selecting models on dev BLEU, this should not be a terrible problem, but we will look into fixing this soon. 
