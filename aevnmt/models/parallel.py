import torch
from torch import nn
from aevnmt.dist import get_named_params, create_prior


class ParallelWrapper(nn.DataParallel):
    def __getattr__(self, name):
        """
        Give self access to the methods of the wrapped model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module.model, name)


class ParallelAEVNMT(nn.Module):

    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.hparams = hparams

    def forward(self, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in, step):

        # for arg in (x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in):
        #     print(type(arg))
        #     if isinstance(arg, torch.Tensor):
        #         print(arg.shape)
        # print()
        # Inf model
        qz = self.model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
        z = qz.rsample()

        # Gen model
        tm_likelihood, lm_likelihood, _, aux_lm_likelihoods, aux_tm_likelihoods = self.model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in, z)

        # Loss
        if self.hparams.kl.annealing_steps > 0:
            KL_weight = min(1., (1.0 / self.hparams.kl.annealing_steps) * step)
        else:
            KL_weight = 1.
            
        return_dict = self.model.loss(tm_likelihood, lm_likelihood, y_out, x_out, qz,
                                    free_nats=self.hparams.kl.free_nats,
                                    KL_weight=KL_weight,
                                    reduction="none",
                                    smoothing_x=self.hparams.gen.lm.label_smoothing,
                                    smoothing_y=self.hparams.gen.tm.label_smoothing,
                                    aux_lm_likelihoods=aux_lm_likelihoods,
                                    aux_tm_likelihoods=aux_tm_likelihoods,
                                    loss_cfg=None)

        # Add posterior values to return dict for Tensorboard
        return_dict['posterior/z'] = z.detach()
        for param_name, param_value in get_named_params(qz):
            return_dict["posterior/%s" % param_name] = param_value.detach()
        return return_dict


def aevnmt_train_parallel(model, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in,
                          hparams, step, summary_writer=None):
    return_dict = model(x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in, step)
    return_dict['loss'] = return_dict['loss'].mean()

    # To keep the summary a reasonable size, only save histograms every print_step
    if summary_writer and step % hparams.print_every == 0:
        for comp_name, comp_value in sorted(return_dict.items()):
            if comp_name.startswith('posterior/'):
                summary_writer.add_histogram(comp_name, comp_value, step)
        pz = model.module.model.prior()
        # This part is perhaps not necessary for a simple prior (e.g. Gaussian),
        #  but it's useful for more complex priors (e.g. mixtures and NFs)
        prior_sample = pz.sample(torch.Size([hparams.batch_size]))
        summary_writer.add_histogram("prior/z", prior_sample, step)
        for param_name, param_value in get_named_params(pz):
            summary_writer.add_histogram("prior/%s" % param_name, param_value, step)
    return return_dict


def nmt_train_parallel(*args, **kwargs):
    raise NotImplementedError()
