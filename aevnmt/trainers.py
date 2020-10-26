import torch
from torch.distributions import Categorical
from aevnmt.dist import get_named_params

class AEVNMTTrainer:
    def __init__(self, model, loss, histogram_every=100):
        self.model = model
        self.loss = loss
        self.histogram_every = histogram_every

    def _repeat_over_samples(self, tensor, num_samples, flatten=True):
        """
        Repeats given tensor with shape [B, ...] over num_samples, the resulting shape is:
        [num_samples * B, ...] if flatten,
        [num_samples, B, ...] otherwise.

        A flattened tensor can be unflattened with self._unflatten_samples.
        """
        if tensor is not None:
            tensor_exp = tensor.unsqueeze(0).repeat(num_samples, *[1]*len(tensor.shape))
            if flatten:
                tensor_exp = self._flatten_samples(tensor_exp)
            return tensor_exp

    def _flatten_samples(self, tensor):
        if tensor is not None:
            return tensor.view(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])

    def _unflatten_samples(self, tensor, num_samples):
        if tensor is not None:
            return tensor.view(num_samples, -1, *tensor.shape[1:])

    def step(self, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out,
             seq_mask_y, seq_len_y, noisy_y_in, step, summary_writer=None):
        q_z = self.model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
        p_z = self.model.prior()

        if self.loss.num_samples == 1:
            z = q_z.rsample()
        else:
            z = q_z.rsample([self.loss.num_samples])
            z = self._flatten_samples(z) # [K * B, ...]

            # Expand required inputs to [K * B, ...]
            noisy_x_in = self._repeat_over_samples(noisy_x_in, self.loss.num_samples, flatten=True)
            noisy_y_in = self._repeat_over_samples(noisy_y_in, self.loss.num_samples, flatten=True)
            seq_mask_x = self._repeat_over_samples(seq_mask_x, self.loss.num_samples, flatten=True)
            seq_len_x = self._repeat_over_samples(seq_len_x, self.loss.num_samples, flatten=True)

            # Expand targets
            x_out = self._repeat_over_samples(x_out, self.loss.num_samples, flatten=True)
            y_out = self._repeat_over_samples(y_out, self.loss.num_samples, flatten=True)

        tm_likelihood, lm_likelihood = self.model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in, z)

        loss_dict = self.loss(tm_likelihood, lm_likelihood, y_out, x_out, q_z, p_z, z, step, self.model, reduction='mean')

        if summary_writer and step % self.histogram_every == 0:
            # generate histograms for the posterior and prior
            summary_writer.add_histogram("posterior/z", z, step)
            for param_name, param_value in get_named_params(q_z):
                summary_writer.add_histogram("posterior/%s" % param_name, param_value, step)
            prior_sample = p_z.sample(torch.Size([z.size(0)]))
            summary_writer.add_histogram("prior/z", prior_sample, step)
            for param_name, param_value in get_named_params(p_z):
                summary_writer.add_histogram("prior/%s" % param_name, param_value, step)

        return loss_dict


class NMTTrainer:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

    def step(self, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out,
             seq_mask_y, seq_len_y, noisy_y_in, step, summary_writer=None):
        likelihood, _ = model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in)
        loss_dict = self.loss(likelihood, y_out, reduction="mean")
        return loss_dict