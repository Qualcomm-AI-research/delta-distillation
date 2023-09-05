#  ============================================================================
#  @@-COPYRIGHT-START-@@
#
# Adapted and modified from the code by Andreas Veit:
# https://github.com/andreasveit/convnet-aig/blob/master/gumbelmodule.py
# Gumbel Softmax Sampler
# Works for categorical and binary input
#
#  @@-COPYRIGHT-END-@@
#  ============================================================================
import torch
import torch.nn as nn


class HardSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        y_hard = input.clone()
        y_hard = y_hard.zero_()
        y_hard[input >= 0.5] = 1

        return y_hard

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None


class GumbelSoftmax(torch.nn.Module):
    def __init__(self):
        """
            Implementation of gumbel softmax for a binary case using gumbel sigmoid.
        """
        super(GumbelSoftmax, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))

        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature, inference=False):
        """ Adds noise to the logits and takes the sigmoid. No Gumbel noise during inference."""
        if not inference:
            gumble_samples_tensor = self.sample_gumbel_like(logits.data)
            gumble_trick_log_prob_samples = logits + gumble_samples_tensor.data
        else:
            gumble_trick_log_prob_samples = logits
        soft_samples = self.sigmoid(gumble_trick_log_prob_samples / temperature)

        return soft_samples

    def gumbel_softmax_sample_binary(self, logits, temperature, inference=False):
        """ Adds noise to the logits and takes the sigmoid. No Gumbel noise during inference."""
        if not inference:
            g1 = self.sample_gumbel_like(logits.data)
            g2 = self.sample_gumbel_like(logits.data)
            gumble_trick_log_prob_samples = logits + (g1 - g2).data
        else:
            gumble_trick_log_prob_samples = logits
        soft_samples = self.sigmoid(gumble_trick_log_prob_samples / temperature)

        return soft_samples

    def gumbel_softmax(self, logits, temperature=2 / 3, hard=False, inference=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
            Args:
                logits: [batch_size, n_class] unnormalized log-probs
                temperature: non-negative value
                hard: if True, take argmax, but differentiate w.r.t. soft sample y
                inference: if True, does not add the gumbel noise

            Returns:
                [batch_size, n_class] sample from the Gumbel-Softmax distribution.
                If hard=True, then the returned sample will be one-hot, otherwise it will
                be a probability distribution that sums to 1 across classes
                For sigmoid this is not the case since we have single node per gate
        """
        out = self.gumbel_softmax_sample_binary(logits, temperature, inference)
        if hard:
            out = HardSoftmax.apply(out)

        return out

    def forward(self, logits, force_hard=False, temperature=2 / 3):
        inference = not self.training

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=temperature, hard=False, inference=inference)
        else:
            return self.gumbel_softmax(logits, temperature=temperature, hard=True, inference=inference)
