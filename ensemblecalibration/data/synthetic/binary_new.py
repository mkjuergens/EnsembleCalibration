import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from numpy import random
from scipy.special import expit
import torch

from ensemblecalibration.utils.helpers import (
    multinomial_label_sampling,
)


class BinaryExperiment:
    """
    A unified experiment class for binary classification that can
    generate data in two ways:

    - 'gp': Using a Gaussian Process to sample the "true" function p(Y=1|x),
            then optionally shifting/adding noise for ensemble members.
    - 'logistic': Using a mixture of Gaussians for X conditioned on Y,
                  and a logistic link as the "true" function. Then create
                  ensemble predictions by random offsets, etc.

    Attributes
    ----------
    method : str
        Either "gp" or "logistic".
    n_samples : int
        Number of samples to generate.
    n_ens : int
        Number of ensemble members.
    scale_noise : float
        Controls the amplitude of random noise for ensemble predictions.
    kernel : callable
        Kernel function if method == "gp".
    n_classes : int
        Typically 2 for binary.

    After calling `generate_data()`, you get:
    - self.x_inst : (n_samples,) or (n_samples,1) array/tensor of inputs
    - self.p_true : (n_samples, 2) array with true probabilities
    - self.ens_preds : (n_samples, n_ens, 2) array with ensemble predictions
    - self.y_labels : (n_samples,) sampled labels
    """

    def __init__(
        self,
        method="gp",
        n_samples=1000,
        n_ens=5,
        scale_noise=0.5,
        kernel=rbf_kernel,
        # logistic mixture parameters:
        mixture_loc=(-1.0, +1.0),    # location of Gaussians for Y=0, Y=1
        mixture_std=1.0,
        # for GP kernel
        kernel_width=0.05
    ):
        self.method = method.lower()
        self.n_samples = n_samples
        self.n_ens = n_ens
        self.scale_noise = scale_noise
        self.kernel = kernel
        self.n_classes = 2

        # if logistic mixture approach:
        self.mixture_loc = mixture_loc
        self.mixture_std = mixture_std

        # extra param for GP kernel
        self.kernel_width = kernel_width

        # placeholders
        self.x_inst = None
        self.p_true = None
        self.ens_preds = None
        self.y_labels = None

    def generate_data(self, **kwargs):
        """
        Main entry: generate data according to 'method'.
        """
        if self.method == "gp":
            self._generate_data_gp(**kwargs)
        elif self.method == "logistic":
            self._generate_data_logistic(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    ##################################################
    # GP-based data generation (existing approach)
    ##################################################
    def _generate_data_gp(self, low=0.0, high=5.0, shift_fn=None):
        """
        Uses your existing approach: sample x in [low, high], 
        build a GP, sample logistic function => p_true, 
        then create ensemble with shifts/noise.
        """
        # 1) sample x
        self.x_inst = np.random.uniform(low, high, self.n_samples)
        # 2) generate ground truth p_true
        self.p_true = self._generate_gt_gp(self.x_inst)
        # 3) ensemble predictions
        if shift_fn is None:
            shift_fn = self._default_shift_fn
        self.ens_preds = self._generate_ens_preds_gp(self.x_inst, self.p_true[:, 0], shift_fn)
        # 4) sample labels
        self.y_labels = multinomial_label_sampling(self.p_true, tensor=True)
        # 5) store x_inst as a float tensor
        self.x_inst = torch.from_numpy(self.x_inst).float().view(-1, 1)

    def _generate_gt_gp(self, x_inst):
        """
        Create a GP sample -> logit -> p_true.
        """
        p_true = np.zeros((x_inst.shape[0], 2))
        # build kernel matrix
        dist = x_inst.reshape(-1, 1) - x_inst.reshape(1, -1)
        K = self.kernel(dist, dist, gamma=self.kernel_width)
        K += 1e-6 * np.eye(K.shape[0])

        # sample GP
        f_vals = np.random.multivariate_normal(np.zeros(x_inst.shape[0]), K)
        # apply sigmoid
        p_true_1 = expit(f_vals)
        p_true[:, 0] = p_true_1
        p_true[:, 1] = 1 - p_true_1
        return p_true

    def _default_shift_fn(self, x):
        return np.sin(np.pi * x)

    def _generate_ens_preds_gp(self, x_inst, p_true, shift_fn):
        """
        Create ensemble predictions by shifting the logit of p_true
        and adding correlated noise.
        """
        p_true_logit = np.log(p_true / (1 - p_true + 1e-12) + 1e-12)
        shift_vals = shift_fn(x_inst)
        m_logit = p_true_logit + shift_vals

        ens_preds = np.zeros((x_inst.shape[0], self.n_ens, self.n_classes))
        for m in range(self.n_ens):
            # random offset
            offset = np.random.uniform(-0.5, 0.5, size=x_inst.shape[0])
            # correlated noise from GP
            noise = self._sample_gp_noise(x_inst)
            noise_scaled = noise * self.scale_noise

            logit_m = m_logit + noise_scaled + offset
            z1 = 1.0 / (1.0 + np.exp(-logit_m))
            z2 = 1.0 - z1
            ens_preds[:, m, 0] = z1
            ens_preds[:, m, 1] = z2

        return ens_preds

    def _sample_gp_noise(self, x_inst):
        """
        Sample correlated noise from a GP (RBF).
        """
        dist = x_inst.reshape(-1, 1) - x_inst.reshape(1, -1)
        K = self.kernel(dist, dist, gamma=self.kernel_width)
        K += 1e-8 * np.eye(K.shape[0])
        noise = np.random.multivariate_normal(np.zeros(x_inst.shape[0]), K)
        return noise

    ##################################################
    # Logistic mixture approach (like a 2-class mixture of Gaussians for X)
    ##################################################
    def _generate_data_logistic(self):
        """
        Generate data from Y=0 or 1 w.p. 0.5, 
        then X ~ N(mixture_loc[y], mixture_std^2).
        The 'true' p_true is logistic in form, or direct from the ground truth.

        Then produce ensemble predictions by offsetting logistic parameters.
        """
        # 1) sample y in {0,1} (balanced)
        y_array = np.random.binomial(n=1, p=0.5, size=self.n_samples)

        # 2) sample x from N(mixture_loc[y], mixture_std^2)
        x_array = np.zeros_like(y_array, dtype=float)
        mask0 = (y_array == 0)
        mask1 = (y_array == 1)
        x_array[mask0] = np.random.normal(self.mixture_loc[0], self.mixture_std, mask0.sum())
        x_array[mask1] = np.random.normal(self.mixture_loc[1], self.mixture_std, mask1.sum())

        # 3) ground-truth p_true(Y=1|x). For logistic: p = 1/(1+exp(2x)) if you want that from eqn(13) 
        #    or define your own logistic function. Let's do the eqn(13) style: p= 1/(1+exp(2*x))
        p = 1.0 / (1.0 + np.exp(2.0 * x_array))  # shape (n_samples,)
        p_true = np.stack([p, 1 - p], axis=1)

        self.x_inst = x_array
        self.y_labels = torch.from_numpy(y_array).long()
        self.p_true = p_true

        # 4) ensemble predictions with random offsets around some logistic param
        self.ens_preds = self._generate_ens_preds_logistic(x_array, p)

        # store x_inst as torch
        self.x_inst = torch.from_numpy(x_array).float().view(-1,1)

    def _generate_ens_preds_logistic(self, x_array, p_array):
        """
        Example: we treat p_array as the 'center' logistic, then produce K members
        by adjusting the logit with random offsets. 
        """
        # logit(center) = - log(1/p -1)
        # or direct: logit_c = log(p/(1-p))
        eps = 1e-12
        logit_center = np.log(p_array + eps) - np.log(1 - p_array + eps)

        ens_preds = np.zeros((len(x_array), self.n_ens, self.n_classes))
        # sample location of offset (mean of normal)
        loc = np.random.normal(loc=2.0, scale=1.0, size=len(x_array))
        for k in range(self.n_ens):
            # random offset (like a small normal or uniform)
            offset = np.random.normal(loc=loc, scale=self.scale_noise, size=len(x_array))
            # apply offset
            logit_k = logit_center + offset
            z1 = 1.0 / (1.0 + np.exp(-logit_k))
            z2 = 1.0 - z1
            ens_preds[:,k,0] = z1
            ens_preds[:,k,1] = z2

        return ens_preds


##################################################
# Example usage
##################################################
if __name__ == "__main__":
    # 1) GP-based
    exp_gp = BinaryExperiment(
        method="gp",
        n_samples=1000,
        n_ens=5,
        scale_noise=0.5,
        kernel=rbf_kernel,
        kernel_width=0.05
    )
    exp_gp.generate_data()
    print("GP-based data shapes:")
    print("x_inst:", exp_gp.x_inst.shape)
    print("p_true:", exp_gp.p_true.shape)
    print("ens_preds:", exp_gp.ens_preds.shape)
    print("y_labels:", exp_gp.y_labels.shape)

    # 2) Logistic mixture
    exp_logistic = BinaryExperiment(
        method="logistic",
        n_samples=1000,
        n_ens=5,
        scale_noise=0.2,
        mixture_loc=(-1.0, +1.0),
        mixture_std=1.0
    )
    exp_logistic.generate_data()
    print("\nLogistic mixture data shapes:")
    print("x_inst:", exp_logistic.x_inst.shape)
    print("p_true:", exp_logistic.p_true.shape)
    print("ens_preds:", exp_logistic.ens_preds.shape)
    print("y_labels:", exp_logistic.y_labels.shape)    

        