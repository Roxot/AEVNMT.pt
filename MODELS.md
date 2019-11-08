
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


# Known Problems

* Importance-sampling estimates of log-likelihood show some instabilities with the Beta prior. As long as you are selecting models on dev BLEU, this should not be a terrible problem, but we will look into fixing this soon. 
