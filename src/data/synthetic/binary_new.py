import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from numpy import random
from scipy.special import expit
import torch

from src.utils.helpers import (
    multinomial_label_sampling,
)

import numpy as np

# class SyntheticEnsembleGenerator:
#     """
#     Generator for synthetic binary classification data with ensemble predictions.
    
#     This class creates an ensemble of binary classifiers (predictors) with controlled 
#     differences, and can sample a synthetic dataset for experiments. Each ensemble 
#     member's prediction function is a perturbed version of a base logistic function.
    
#     The perturbation can be done in two ways:
#     - "logit": Add structured noise to the logits (linear outputs) of the base function.
#     - "gp": Sample functions from a Gaussian Process prior (approximated for simplicity).
    
#     By adjusting parameters, users can control how much ensemble members disagree with 
#     each other (ensemble spread) and introduce systematic biases for each predictor.
#     """
    
#     def __init__(self, ensemble_size=5, perturbation_method="logit", 
#                  perturbation_scale=1.0, bias_scale=None, random_state=None):
#         """
#         Initialize the synthetic data generator.
        
#         Parameters:
#         - ensemble_size (int): Number of ensemble members (predictors) to generate.
#         - perturbation_method (str): Method for perturbing the base function. 
#           Options are "logit" (default) for direct logit perturbation, or "gp" for 
#           Gaussian Process-based perturbation.
#         - perturbation_scale (float): Scaling factor (standard deviation) for the 
#           perturbations that control ensemble disagreement. Higher values produce 
#           more variance in ensemble logits (wider spread in predictions). For GP, 
#           this corresponds to the GP prior standard deviation. 
#         - bias_scale (float or None): Standard deviation for the random bias term 
#           added to each predictor's logit. If None, it defaults to the same value 
#           as perturbation_scale. This bias causes each ensemble member to be 
#           systematically shifted up or down.
#         - random_state (int or np.random.RandomState or None): Seed or RandomState 
#           for reproducible randomness. If an int is provided, it will be used as 
#           the seed. If None, the random draws will be different each time.
#         """
#         self.ensemble_size = ensemble_size
#         self.method = perturbation_method.lower()
#         if self.method not in ("logit", "gp"):
#             raise ValueError("perturbation_method must be 'logit' or 'gp'")
        
#         # Scale for perturbations and biases
#         self.perturbation_scale = perturbation_scale
#         # If bias_scale not provided, use the same as perturbation_scale
#         self.bias_scale = perturbation_scale if bias_scale is None else bias_scale
        
#         # Set up random state for reproducibility
#         if isinstance(random_state, np.random.RandomState):
#             self.rng = random_state
#         else:
#             # If random_state is None or int, create a RandomState
#             self.rng = np.random.RandomState(random_state)
        
#         # Placeholders for ensemble parameters (to ensure functions are consistent across calls)
#         self.base_weights = None   # will hold base function weights (if base function is linear)
#         self.base_bias = None      # base function bias
#         self.ensemble_biases = None  # biases for each ensemble member
#         # For "logit" method
#         self.ensemble_weights = None  # weight vectors for each ensemble member's perturbation (if using linear perturbation)
#         # For "gp" method
#         self.random_freq = None    # random frequencies for RFF (if using GP approximation)
#         self.random_phase = None   # random phase for RFF
#         self.ensemble_coeffs = None  # coefficient for each ensemble member on random features (GP mode)
        
#     def generate(self, X):
#         """
#         Generate synthetic binary classification labels and ensemble predictions for given instances.
        
#         Parameters:
#         - X (ndarray of shape (n_samples, n_features)): Feature matrix for the instances where we want to 
#           generate labels and ensemble predictions.
          
#         Returns:
#         - y_true (ndarray of shape (n_samples,)): The true binary labels (0 or 1) for each instance, 
#           sampled from the base logistic function.
#         - ensemble_probs (ndarray of shape (n_samples, ensemble_size)): The predicted probability of the positive 
#           class for each instance from each ensemble member. Each column corresponds to one ensemble member's 
#           predictions across all instances.
#         """
#         X = np.asarray(X)
#         n_samples, n_features = X.shape
        
#         # --- Initialize base function (ground truth) if not already set ---
#         if self.base_weights is None:
#             # For simplicity, define a base logistic function as a random linear model
#             # We draw base_weights from N(0,1) and normalize to unit length (to control scale), 
#             # then we can scale it if needed (here we keep scale = 1 for base).
#             base_w = self.rng.normal(loc=0.0, scale=1.0, size=n_features)
#             # Normalize the weight vector to unit L2 norm to standardize effect size
#             norm = np.linalg.norm(base_w)
#             if norm != 0:
#                 base_w = base_w / norm
#             self.base_weights = base_w
#             # Set base bias to 0 (center decision boundary roughly at 0 logit)
#             self.base_bias = 0.0
        
#         # Compute base logits for all instances: this is the underlying "true" logit function
#         base_logit = X.dot(self.base_weights) + self.base_bias
        
#         # Compute true probability of class 1 via logistic sigmoid
#         p_true = 1.0 / (1.0 + np.exp(-base_logit))
        
#         # Sample true labels from Bernoulli(p_true) for each instance
#         # This introduces some label noise according to the probability. 
#         # If deterministic labels are desired, one could threshold p_true at 0.5 instead.
#         y_true = (self.rng.rand(n_samples) < p_true).astype(int)
        
#         # --- Initialize ensemble parameters if not already done ---
#         if self.ensemble_biases is None:
#             # Sample a fixed bias for each ensemble member from N(0, bias_scale)
#             self.ensemble_biases = self.rng.normal(loc=0.0, scale=self.bias_scale, size=self.ensemble_size)
        
#         # Prepare array for ensemble logits: shape (n_samples, ensemble_size)
#         ensemble_logits = np.empty((n_samples, self.ensemble_size))
        
#         if self.method == "logit":
#             # If using logit perturbation method:
#             if self.ensemble_weights is None:
#                 # Sample random weight vectors for each ensemble member's perturbation.
#                 # Draw from N(0, 1) and scale them such that the resulting logit perturbation has desired scale.
#                 # We use 1/sqrt(n_features) to moderate the scale if features are roughly standardized.
#                 self.ensemble_weights = self.rng.normal(loc=0.0, scale=1.0, size=(self.ensemble_size, n_features))
#                 # Scale the weights down to control the variance of perturbation.
#                 # We use perturbation_scale in combination with feature count:
#                 self.ensemble_weights *= (self.perturbation_scale / np.sqrt(n_features))
            
#             # Compute perturbation for each ensemble member: X dot weight^T gives (n_samples, ensemble_size)
#             # Each column j is the structured offset for predictor j across all instances.
#             linear_offset = X.dot(self.ensemble_weights.T)  # shape (n_samples, ensemble_size)
#             # Add each predictor's bias to its column
#             ensemble_logits = linear_offset + self.ensemble_biases  # broadcasting bias (shape (ensemble_size,))
#             # Now add the base logit to all ensemble logits
#             ensemble_logits = ensemble_logits + base_logit[:, None]  # base_logit shape (n_samples,) -> (n_samples,1) for broadcasting
            
#         elif self.method == "gp":
#             # If using GP-based perturbation:
#             if self.random_freq is None or self.random_phase is None or self.ensemble_coeffs is None:
#                 # Initialize random Fourier features for RBF kernel approximation.
#                 # Choose number of random features (for simplicity, 100 or 2*n_features, whichever is larger)
#                 num_features = max(100, 2 * n_features)
#                 # Draw random frequencies from a normal distribution scaled by desired length-scale.
#                 # Here we use length_scale = 1.0 by default. (Users can modify if needed by altering this section.)
#                 length_scale = 1.0
#                 self.random_freq = self.rng.normal(loc=0.0, scale=1.0/length_scale, size=(num_features, n_features))
#                 # Draw random phase shifts uniformly from 0 to 2pi
#                 self.random_phase = self.rng.uniform(low=0.0, high=2*np.pi, size=num_features)
#                 # Sample random coefficients for each ensemble member for these features
#                 # Coefficients drawn from N(0, perturbation_scale) to match desired logit variance
#                 self.ensemble_coeffs = self.rng.normal(loc=0.0, scale=self.perturbation_scale, 
#                                                        size=(self.ensemble_size, num_features))
            
#             # Compute the random feature mapping for each instance:
#             # feature_map[i] = sqrt(2/num_features) * cos(random_freq[i] dot x + random_phase[i])
#             # We'll vectorize this computation for all instances and all random features.
#             # X shape: (n_samples, n_features), random_freq: (num_features, n_features)
#             # Compute random_freq * X^T to get shape (num_features, n_samples)
#             RFX = self.random_freq.dot(X.T)  # shape (num_features, n_samples)
#             # Add phase and take cosine
#             RFX = RFX + self.random_phase[:, None]  # add phase (broadcasted over samples)
#             phi = np.sqrt(2.0 / self.random_freq.shape[0]) * np.cos(RFX)  # shape (num_features, n_samples)
            
#             # Now phi is (num_features, n_samples). We need ensemble logits:
#             # Each ensemble j: coeffs[j] dot phi (coeffs[j] shape (num_features,))
#             # We can do this via matrix multiplication as well.
#             # First, compute phi transpose (n_samples, num_features) for convenience:
#             phi_T = phi.T  # shape (n_samples, num_features)
#             # Ensemble coefficients shape: (ensemble_size, num_features)
#             # So ensemble_logits_perturb = phi_T dot ensemble_coeffs^T -> shape (n_samples, ensemble_size)
#             perturbation = phi_T.dot(self.ensemble_coeffs.T)  # shape (n_samples, ensemble_size)
#             # Add biases (each column j gets bias_j)
#             perturbation += self.ensemble_biases  # bias broadcast across n_samples
#             # Add base logit to all ensemble logits
#             ensemble_logits = perturbation + base_logit[:, None]
        
#         # Convert ensemble logits to probabilities via sigmoid for output
#         ensemble_probs = 1.0 / (1.0 + np.exp(-ensemble_logits))
        
#         return y_true, ensemble_probs


class BinaryExperiment:
    """
    A unified experiment class for binary classification that generates synthetic data 
    and ensemble predictions via two methods: 'gp' and 'logistic'.
    
    For the "gp" method:
      1) x is sampled uniformly in [low, high].
      2) The ground truth logit is computed as:
             f(x) = a * (x - center) + f_gp(x)
         where center = (low+high)/2 and f_gp(x) is sampled from a multivariate normal with 
         an RBF kernel: K(i,j) = exp(- (x_i - x_j)^2 / (2*kernel_width^2)).
         The ground truth probability is p_true = sigmoid(f(x)).
      3) Ensemble predictions are generated deterministically. For each ensemble member m:
             f_m(x) = f(x) + bias_m + A_m * sin(π * x / L + C_m)
         where bias_m is sampled individually from offset_range, L = high - low, and A_m and C_m 
         are random amplitude and phase terms (with A_m scaled by scale_noise). The ensemble 
         probability is p_m(x) = sigmoid(f_m(x)).
         
    For the "logistic" method:
      1) x is sampled from a Gaussian mixture conditioned on Y.
      2) The base probability is computed using a logistic function (e.g., p = 1/(1+exp(2*x))) 
         and the corresponding base logit is derived.
      3) Ensemble predictions are generated analogously:
             f_m(x) = base_logit(x) + bias_m + A_m * sin(π * x / L + C_m)
         ensuring each ensemble member deviates from the ground truth function.
         
    Attributes
    ----------
    method : str
        Either "gp" or "logistic". 
    n_samples : int
        Number of samples to generate.
    n_ens : int
        Number of ensemble members.
    scale_noise : float
        Standard deviation of the Gaussian noise used as amplitude for the sine perturbation. 
        Controls how much each ensemble function deviates from the base function.
    offset_range : list of two floats
        Range from which an individual bias is sampled for each ensemble member.
    n_classes : int
        Typically 2 for binary classification.
    
    After calling generate_data(), the following attributes are set:
      - self.x_inst : torch.FloatTensor of shape (n_samples, 1) with input instances.
      - self.p_true : np.ndarray of shape (n_samples, 2) with ground truth probabilities.
      - self.ens_preds : np.ndarray of shape (n_samples, n_ens, 2) with ensemble predictions.
      - self.y_labels : torch.LongTensor of shape (n_samples,) with sampled labels.
    """
    
    def __init__(self,
                 method="gp",
                 n_samples=1000,
                 n_ens=5,
                 scale_noise=0.5,
                 offset_range=[-0.0, 4.0],
                 # Parameters for GP method:
                 low=0.0,
                 high=5.0,
                 a=1.0,
                 kernel_width=1.0,
                 # Parameters for logistic method:
                 mixture_loc=(-1.0, +1.0),
                 mixture_std=1.0):
        self.method = method.lower()
        self.n_samples = n_samples
        self.n_ens = n_ens
        self.scale_noise = scale_noise
        self.offset_range = offset_range
        self.n_classes = 2

        # GP parameters
        self.low = low
        self.high = high
        self.a = a
        self.kernel_width = kernel_width

        # Logistic method parameters
        self.mixture_loc = mixture_loc
        self.mixture_std = mixture_std

        # Placeholders for generated data
        self.x_inst = None
        self.p_true = None
        self.ens_preds = None
        self.y_labels = None

    def generate_data(self, **kwargs):
        """
        Main entry: generate synthetic data and ensemble predictions.
        
        Sets:
          - self.x_inst : torch.FloatTensor of shape (n_samples, 1)
          - self.p_true : np.ndarray of shape (n_samples, 2)
          - self.ens_preds : np.ndarray of shape (n_samples, n_ens, 2)
          - self.y_labels : torch.LongTensor of shape (n_samples,)
        """
        if self.method == "gp":
            self._generate_data_gp(**kwargs)
        elif self.method == "logistic":
            self._generate_data_logistic(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    ##################################################
    # GP-based data generation with GP ground truth
    ##################################################
    def _generate_data_gp(self):
        """
        Generate data using a GP-based approach.
        
        1) Sample x uniformly in [low, high].
        2) Compute the ground truth logit:
               f(x) = a * (x - center) + f_gp(x)
           where center = (low+high)/2 and f_gp(x) is sampled from a multivariate normal with 
           an RBF kernel: K(i,j) = exp(- (x_i - x_j)^2 / (2*kernel_width^2)).
        3) Compute p_true = sigmoid(f(x)).
        4) Sample binary labels from p_true.
        5) Generate ensemble predictions as deterministic functions:
             For each ensemble member m:
                 f_m(x) = f(x) + bias_m + A_m * sin(π * x / L + C_m)
             where each bias_m is individually sampled from offset_range, and A_m and C_m are 
             sampled such that the sine perturbation has amplitude determined by scale_noise.
             Then, p_m(x) = sigmoid(f_m(x)).
        6) Convert x_inst to a torch.FloatTensor.
        """
        # 1) Sample x uniformly in [low, high]
        x_inst = np.random.uniform(self.low, self.high, self.n_samples)
        
        # 2) Compute base function f(x)
        center = (self.low + self.high) / 2.0
        # Build RBF kernel matrix for x_inst
        diff = x_inst.reshape(-1, 1) - x_inst.reshape(1, -1)
        K = np.exp(- (diff ** 2) / (2 * self.kernel_width ** 2))
        K += 1e-6 * np.eye(self.n_samples)  # for numerical stability
        # Sample f_gp ~ N(0, K)
        f_gp = np.random.multivariate_normal(mean=np.zeros(self.n_samples), cov=K)
        # Base function: linear trend plus GP sample
        f_true = self.a * (x_inst - center) + f_gp
        
        # 3) Ground truth probability
        p_true_1 = expit(f_true)
        self.p_true = np.stack([p_true_1, 1 - p_true_1], axis=1)
        
        # 4) Sample labels from p_true
        self.y_labels = multinomial_label_sampling(self.p_true, tensor=True)
        
        # 5) Generate ensemble predictions as deterministic functions.
        L = self.high - self.low  # length of the interval
        ens_preds = np.zeros((self.n_samples, self.n_ens, self.n_classes))
        for m in range(self.n_ens):
            # Sample an individual bias for ensemble member m
            bias_m = np.random.uniform(self.offset_range[0], self.offset_range[1])
            # Sample amplitude and phase for a sine perturbation
            A_m = np.random.uniform(-self.scale_noise, self.scale_noise)
            C_m = np.random.uniform(0, 2 * np.pi)
            perturbation = A_m * np.sin(np.pi * x_inst / L + C_m)
            # Compute ensemble logit for member m
            f_m = f_true + bias_m + perturbation
            p_m = expit(f_m)
            ens_preds[:, m, 0] = p_m
            ens_preds[:, m, 1] = 1 - p_m
        self.ens_preds = ens_preds
        
        # 6) Convert x_inst to a torch tensor
        self.x_inst = torch.from_numpy(x_inst).float().reshape(-1, 1)

    ##################################################
    # Logistic mixture data generation with deterministic ensemble functions
    ##################################################
    def _generate_data_logistic(self):
        """
        Generate data using a logistic mixture approach.
        
        1) Sample balanced binary labels with p=0.5.
        2) For each label, sample x from a Gaussian:
               x ~ N(mixture_loc[y], mixture_std^2).
        3) Compute the base probability using a logistic function:
               p = 1/(1+exp(2*x)), and derive base logit = log(p/(1-p)).
        4) Sample labels from p.
        5) Generate ensemble predictions deterministically:
               For each ensemble member m:
                   f_m(x) = base_logit + bias_m + A_m * sin(π * x / L + C_m)
               where bias_m is individually sampled from offset_range and A_m, C_m are random.
               Then, p_m(x) = sigmoid(f_m(x)).
        6) Convert x_inst to a torch.FloatTensor.
        """
        # 1) Sample balanced binary labels
        y_array = np.random.binomial(n=1, p=0.5, size=self.n_samples)
        
        # 2) Sample x from a Gaussian conditioned on y
        x_array = np.zeros(self.n_samples, dtype=float)
        mask0 = (y_array == 0)
        mask1 = (y_array == 1)
        x_array[mask0] = np.random.normal(self.mixture_loc[0], self.mixture_std, mask0.sum())
        x_array[mask1] = np.random.normal(self.mixture_loc[1], self.mixture_std, mask1.sum())
        
        # 3) Compute base logistic probability and logit
        p = 1.0 / (1.0 + np.exp(2.0 * x_array))
        eps = 1e-12
        base_logit = np.log(p + eps) - np.log(1 - p + eps)
        self.p_true = np.stack([p, 1 - p], axis=1)
        
        # 4) Sample labels from p_true
        self.y_labels = multinomial_label_sampling(self.p_true, tensor=True)
        
        # 5) Generate ensemble predictions as deterministic functions
        L = np.max(x_array) - np.min(x_array)
        ens_preds = np.zeros((self.n_samples, self.n_ens, self.n_classes))
        for m in range(self.n_ens):
            # Sample an individual bias for member m
            bias_m = np.random.uniform(self.offset_range[0], self.offset_range[1])
            A_m = np.random.uniform(-self.scale_noise, self.scale_noise)
            C_m = np.random.uniform(0, 2 * np.pi)
            perturbation = A_m * np.sin(np.pi * x_array / L + C_m)
            f_m = base_logit + bias_m + perturbation
            p_m = expit(f_m)
            ens_preds[:, m, 0] = p_m
            ens_preds[:, m, 1] = 1 - p_m
        self.ens_preds = ens_preds
        
        # 6) Convert x_inst to a torch tensor
        self.x_inst = torch.from_numpy(x_array).float().reshape(-1, 1)




# class BinaryExperiment:
#     """
#     A unified experiment class for binary classification that generates synthetic data 
#     and ensemble predictions via two methods: 'gp' and 'logistic'.
    
#     For the "gp" method:
#       1) x is sampled uniformly in [low, high].
#       2) The ground truth logit is computed as:
#              f(x) = a * (x - center) + f_gp(x)
#          where center = (low + high)/2, and f_gp(x) is sampled from a Gaussian Process
#          with an RBF kernel (parameterized by kernel_width). The ground truth probability 
#          is p_true = sigmoid(f(x)).
#       3) Ensemble predictions are generated deterministically by setting, for each ensemble 
#          member m:
#              f_m(x) = f(x) + offset_common + A_m * sin( π * x / L + C_m )
#          with a common offset (offset_common) sampled once from offset_range, L = high - low, 
#          and A_m (amplitude) and C_m (phase) randomly sampled for each member. The ensemble 
#          prediction is p_m(x) = sigmoid(f_m(x)).
         
#     For the "logistic" method:
#       1) x is sampled from a Gaussian mixture conditioned on the label.
#       2) The base logit is computed from a logistic function (here, using p = 1/(1+exp(2*x))).
#       3) Ensemble predictions are generated similarly as in the "gp" method, by perturbing the 
#          base logit with a common offset and a sine function.

#     Attributes
#     ----------
#     method : str
#         Either "gp" or "logistic".
#     n_samples : int
#         Number of samples to generate.
#     n_ens : int
#         Number of ensemble members.
#     scale_noise : float
#         Standard deviation (amplitude) of the deterministic perturbation (in logit space) 
#         that controls the spread of ensemble predictions.
#     offset_range : list of two floats
#         Range from which a common offset is sampled for the ensemble predictions.
#     n_classes : int
#         Number of classes (2 for binary classification).
    
#     After calling generate_data(), the following attributes are set:
#       - self.x_inst : torch.FloatTensor of shape (n_samples, 1) with input instances.
#       - self.p_true : np.ndarray of shape (n_samples, 2) with the ground truth probabilities.
#       - self.ens_preds : np.ndarray of shape (n_samples, n_ens, 2) with ensemble predictions.
#       - self.y_labels : torch.LongTensor of shape (n_samples,) with sampled labels.
#     """
    
#     def __init__(self,
#                  method="gp",
#                  n_samples=1000,
#                  n_ens=5,
#                  scale_noise=0.5,
#                  offset_range=[-0.0, 4.0],
#                  # Parameters for GP method:
#                  low=0.0,
#                  high=5.0,
#                  a=1.0,
#                  kernel_width=0.5,
#                  # Parameters for logistic method:
#                  mixture_loc=(-1.0, +1.0),
#                  mixture_std=1.0):
#         self.method = method.lower()
#         self.n_samples = n_samples
#         self.n_ens = n_ens
#         self.scale_noise = scale_noise
#         self.offset_range = offset_range
#         self.n_classes = 2

#         # GP parameters
#         self.low = low
#         self.high = high
#         self.a = a
#         self.kernel_width = kernel_width

#         # Logistic method parameters
#         self.mixture_loc = mixture_loc
#         self.mixture_std = mixture_std

#         # Placeholders for generated data
#         self.x_inst = None
#         self.p_true = None
#         self.ens_preds = None
#         self.y_labels = None

#     def generate_data(self, **kwargs):
#         """
#         Main entry: generate synthetic data and ensemble predictions.
        
#         Sets:
#           - self.x_inst : torch.FloatTensor of shape (n_samples, 1)
#           - self.p_true : np.ndarray of shape (n_samples, 2)
#           - self.ens_preds : np.ndarray of shape (n_samples, n_ens, 2)
#           - self.y_labels : torch.LongTensor of shape (n_samples,)
#         """
#         if self.method == "gp":
#             self._generate_data_gp(**kwargs)
#         elif self.method == "logistic":
#             self._generate_data_logistic(**kwargs)
#         else:
#             raise ValueError(f"Unknown method: {self.method}")

#     ##################################################
#     # GP-based data generation with GP ground truth
#     ##################################################
#     def _generate_data_gp(self):
#         """
#         Generate data using a GP-based approach.
        
#         1) Sample x uniformly in [low, high].
#         2) Compute the ground truth logit:
#                f(x) = a * (x - center) + f_gp(x)
#            where center = (low + high)/2 and f_gp(x) is sampled from a multivariate normal with an RBF kernel:
#                K(i,j) = exp( - (x_i - x_j)^2 / (2*kernel_width^2) ).
#         3) Compute the ground truth probability: p_true = sigmoid(f(x)).
#         4) Sample binary labels from p_true.
#         5) Generate ensemble predictions deterministically:
#                For each ensemble member m:
#                  f_m(x) = f(x) + offset_common + A_m * sin( π * x / L + C_m )
#            where offset_common is sampled once from offset_range, L = high - low, and A_m, C_m are randomly sampled.
#            Ensemble probability p_m(x) = sigmoid(f_m(x)).
#         6) Convert x_inst to a torch.FloatTensor.
#         """
#         # 1) Sample x uniformly in [low, high]
#         x_inst = np.random.uniform(self.low, self.high, self.n_samples)
        
#         # 2) Compute base function f(x)
#         center = (self.low + self.high) / 2.0
#         # Compute RBF kernel matrix
#         diff = x_inst.reshape(-1, 1) - x_inst.reshape(1, -1)
#         K = np.exp(- (diff ** 2) / (2 * self.kernel_width ** 2))
#         K += 1e-6 * np.eye(self.n_samples)  # for numerical stability
#         # Sample f_gp ~ N(0, K)
#         f_gp = np.random.multivariate_normal(mean=np.zeros(self.n_samples), cov=K)
#         # Base function: add linear trend and GP sample
#         f_true = self.a * (x_inst - center) + f_gp
        
#         # 3) Ground truth probability p_true = sigmoid(f_true)
#         p_true_1 = expit(f_true)
#         self.p_true = np.stack([p_true_1, 1 - p_true_1], axis=1)
        
#         # 4) Sample labels from p_true
#         self.y_labels = multinomial_label_sampling(self.p_true, tensor=True)
        
#         # 5) Generate ensemble predictions as deterministic functions of x.
#         offset_common = np.random.uniform(self.offset_range[0], self.offset_range[1])
#         L = self.high - self.low  # length of the interval
#         ens_preds = np.zeros((self.n_samples, self.n_ens, self.n_classes))
#         for m in range(self.n_ens):
#             # For each ensemble member, sample an amplitude A_m and phase C_m.
#             A_m = np.random.uniform(-self.scale_noise, self.scale_noise)
#             C_m = np.random.uniform(0, 2 * np.pi)
#             perturbation = A_m * np.sin(np.pi * x_inst / L + C_m)
#             f_m = f_true + offset_common + perturbation
#             p_m = expit(f_m)
#             ens_preds[:, m, 0] = p_m
#             ens_preds[:, m, 1] = 1 - p_m
#         self.ens_preds = ens_preds
        
#         # 6) Store x_inst as a torch tensor of shape (n_samples, 1)
#         self.x_inst = torch.from_numpy(x_inst).float().reshape(-1, 1)

#     ##################################################
#     # Logistic mixture data generation with deterministic ensemble functions
#     ##################################################
#     def _generate_data_logistic(self):
#         """
#         Generate data using a logistic mixture approach.
        
#         1) Sample balanced binary labels (0 or 1) with p=0.5.
#         2) For each label, sample x from a Gaussian:
#                x ~ N(mixture_loc[y], mixture_std^2).
#         3) Compute base probability with a logistic function:
#                p = 1/(1+exp(2*x))
#            and compute the base logit as: base_logit = log(p/(1-p)).
#         4) Sample labels from p.
#         5) Generate ensemble predictions deterministically:
#                For each ensemble member m:
#                  f_m(x) = base_logit + offset_common + A_m * sin( π * x / L + C_m )
#            where offset_common is sampled once from offset_range, L is computed as (max(x)-min(x)), 
#            and A_m, C_m are random parameters.
#            Ensemble probability p_m(x) = sigmoid(f_m(x)).
#         6) Convert x_inst to a torch.FloatTensor.
#         """
#         # 1) Sample balanced binary labels
#         y_array = np.random.binomial(n=1, p=0.5, size=self.n_samples)
        
#         # 2) Sample x from a Gaussian conditioned on y
#         x_array = np.zeros(self.n_samples, dtype=float)
#         mask0 = (y_array == 0)
#         mask1 = (y_array == 1)
#         x_array[mask0] = np.random.normal(self.mixture_loc[0], self.mixture_std, mask0.sum())
#         x_array[mask1] = np.random.normal(self.mixture_loc[1], self.mixture_std, mask1.sum())
        
#         # 3) Compute base logistic probability and logit
#         p = 1.0 / (1.0 + np.exp(2.0 * x_array))
#         eps = 1e-12
#         base_logit = np.log(p + eps) - np.log(1 - p + eps)
#         self.p_true = np.stack([p, 1 - p], axis=1)
        
#         # 4) Sample labels from p_true
#         self.y_labels = multinomial_label_sampling(self.p_true, tensor=True)
        
#         # 5) Generate ensemble predictions as deterministic functions
#         L = np.max(x_array) - np.min(x_array)
#         offset_common = np.random.uniform(self.offset_range[0], self.offset_range[1])
#         ens_preds = np.zeros((self.n_samples, self.n_ens, self.n_classes))
#         for m in range(self.n_ens):
#             A_m = np.random.uniform(-self.scale_noise, self.scale_noise)
#             C_m = np.random.uniform(0, 2 * np.pi)
#             perturbation = A_m * np.sin(np.pi * x_array / L + C_m)
#             f_m = base_logit + offset_common + perturbation
#             p_m = expit(f_m)
#             ens_preds[:, m, 0] = p_m
#             ens_preds[:, m, 1] = 1 - p_m
#         self.ens_preds = ens_preds
        
#         # 6) Store x_inst as a torch tensor
#         self.x_inst = torch.from_numpy(x_array).float().reshape(-1, 1)

# class BinaryExperiment:
#     """
#     A unified experiment class for binary classification that generates synthetic data 
#     and ensemble predictions using two methods: 'gp' and 'logistic'.
    
#     In both cases a base function is defined to represent the underlying "true" probability 
#     distribution p(Y=1|x) (stored in self.p_true), and ensemble predictions (self.ens_preds) 
#     are obtained by perturbing the base function's logits with two components:
    
#       - A random bias (drawn from a user–specified offset_range), and 
#       - Gaussian noise in logit space (with standard deviation scale_noise).
    
#     This approach gives you direct control over the spread (disagreement) of the ensemble 
#     predictions in terms of the variance of the logits.
    
#     Attributes
#     ----------
#     method : str
#         Either "gp" or "logistic". 
#           - "gp": Samples x uniformly in [low, high] and defines a base logistic function 
#                   p_true = sigmoid(a * (x - center)) (with center = (low+high)/2). 
#                   Ensemble predictions are produced by adding a random bias and Gaussian noise 
#                   (in logit space) to the base logit.
#           - "logistic": Samples x from a Gaussian mixture conditioned on Y; 
#                   p_true is given by a logistic function (e.g. p=1/(1+exp(2*x))).
#     n_samples : int
#         Number of samples to generate.
#     n_ens : int
#         Number of ensemble members.
#     scale_noise : float
#         Standard deviation of the Gaussian noise added to the base logits. This controls 
#         how far the ensemble members deviate from the base (ground truth) function.
#     offset_range : list of two floats
#         The range from which a constant bias is drawn for each ensemble member.
#     n_classes : int
#         Number of classes (2 for binary classification).
    
#     After calling generate_data(), the following attributes are set:
#       - self.x_inst : torch.FloatTensor of shape (n_samples, 1) with input instances.
#       - self.p_true : np.ndarray of shape (n_samples, 2) with the ground truth probability distribution.
#       - self.ens_preds : np.ndarray of shape (n_samples, n_ens, 2) with ensemble predictions.
#       - self.y_labels : torch.LongTensor of shape (n_samples,) with sampled labels.
#     """
    
#     def __init__(self,
#                  method="gp",
#                  n_samples=1000,
#                  n_ens=5,
#                  scale_noise=0.5,
#                  offset_range=[-0.0, 4.0],
#                  # Parameters for logistic mixture method:
#                  mixture_loc=(-1.0, +1.0),
#                  mixture_std=1.0):
#         self.method = method.lower()
#         self.n_samples = n_samples
#         self.n_ens = n_ens
#         self.scale_noise = scale_noise
#         self.offset_range = offset_range
#         self.n_classes = 2
        
#         # Parameters for the logistic method (if used)
#         self.mixture_loc = mixture_loc
#         self.mixture_std = mixture_std
        
#         # Placeholders for generated data
#         self.x_inst = None
#         self.p_true = None
#         self.ens_preds = None
#         self.y_labels = None
    
#     def generate_data(self, **kwargs):
#         """
#         Main entry point: generate synthetic data and ensemble predictions according 
#         to the specified method.
        
#         Sets the following attributes:
#           - self.x_inst: torch.FloatTensor of shape (n_samples, 1)
#           - self.p_true: np.ndarray of shape (n_samples, 2)
#           - self.ens_preds: np.ndarray of shape (n_samples, n_ens, 2)
#           - self.y_labels: torch.LongTensor of shape (n_samples,)
#         """
#         if self.method == "gp":
#             self._generate_data_gp(**kwargs)
#         elif self.method == "logistic":
#             self._generate_data_logistic(**kwargs)
#         else:
#             raise ValueError(f"Unknown method: {self.method}")
    
#     ##################################################
#     # Simplified GP-based data generation via logit perturbation
#     ##################################################
#     def _generate_data_gp(self, low=0.0, high=5.0, a=1.0):
#         """
#         Generate data using a simplified GP-based approach.
        
#         1) Sample x uniformly in [low, high].
#         2) Define the base logit function as: base_logit = a * (x - center) where center = (low+high)/2.
#            Compute the ground truth probability as: p_true = sigmoid(base_logit).
#         3) For each ensemble member, add a constant bias (sampled from offset_range) and 
#            Gaussian noise (with standard deviation scale_noise) to the base_logit, and apply sigmoid 
#            to obtain perturbed probabilities.
#         4) Sample binary labels from p_true.
#         5) Convert x_inst to a torch.FloatTensor of shape (n_samples, 1).
#         """
#         # 1) Sample x uniformly
#         x_inst = np.random.uniform(low, high, self.n_samples)
#         self.x_inst = x_inst  # temporarily as numpy array
        
#         # 2) Compute base function (ground truth)
#         center = (low + high) / 2.0
#         base_logit = a * (x_inst - center)
#         p_true_1 = expit(base_logit)  # probability for class 1
#         self.p_true = np.stack([p_true_1, 1 - p_true_1], axis=1)
        
#         # 3) Generate ensemble predictions by perturbing the base logits
#         ens_preds = np.zeros((self.n_samples, self.n_ens, self.n_classes))
#         for m in range(self.n_ens):
#             # Sample a constant bias for ensemble member m
#             bias = np.random.uniform(self.offset_range[0], self.offset_range[1])
#             # Sample Gaussian noise for each instance
#             noise = np.random.normal(loc=0.0, scale=self.scale_noise, size=self.n_samples)
#             # Perturb the base logit
#             perturbed_logit = base_logit + bias + noise
#             p_member = expit(perturbed_logit)
#             ens_preds[:, m, 0] = p_member
#             ens_preds[:, m, 1] = 1 - p_member
#         self.ens_preds = ens_preds
        
#         # 4) Sample labels from the ground truth probabilities
#         self.y_labels = multinomial_label_sampling(self.p_true, tensor=True)
        
#         # 5) Store x_inst as a torch tensor of shape (n_samples, 1)
#         self.x_inst = torch.from_numpy(x_inst).float().reshape(-1, 1)
    
#     ##################################################
#     # Logistic mixture data generation
#     ##################################################
#     def _generate_data_logistic(self):
#         """
#         Generate data using a logistic mixture approach.
        
#         1) Sample balanced binary labels (0 or 1) with p=0.5.
#         2) For each label, sample x from a Gaussian: 
#              x ~ N(mixture_loc[y], mixture_std^2).
#         3) Compute the ground truth probability using a logistic function. 
#            Example: p = 1 / (1 + exp(2 * x)).
#         4) For each ensemble member, perturb the base logit (computed from p_true) by 
#            adding a constant bias and Gaussian noise (scale given by scale_noise), and then 
#            apply the sigmoid to obtain ensemble predictions.
#         5) Convert x_inst to a torch.FloatTensor.
#         """
#         # 1) Sample balanced binary labels
#         y_array = np.random.binomial(n=1, p=0.5, size=self.n_samples)
        
#         # 2) Sample x conditioned on y from a Gaussian
#         x_array = np.zeros(self.n_samples, dtype=float)
#         mask0 = (y_array == 0)
#         mask1 = (y_array == 1)
#         x_array[mask0] = np.random.normal(self.mixture_loc[0], self.mixture_std, mask0.sum())
#         x_array[mask1] = np.random.normal(self.mixture_loc[1], self.mixture_std, mask1.sum())
        
#         # 3) Compute ground truth probability using a logistic function
#         p = 1.0 / (1.0 + np.exp(2.0 * x_array))
#         self.p_true = np.stack([p, 1 - p], axis=1)
        
#         # 4) Generate ensemble predictions by perturbing the logit
#         # Compute base logit: logit = log(p/(1-p))
#         eps = 1e-12
#         base_logit = np.log(p + eps) - np.log(1 - p + eps)
#         ens_preds = np.zeros((self.n_samples, self.n_ens, self.n_classes))
#         for m in range(self.n_ens):
#             offset = np.random.uniform(self.offset_range[0], self.offset_range[1])
#             noise = np.random.normal(loc=0.0, scale=self.scale_noise, size=self.n_samples)
#             perturbed_logit = base_logit + offset + noise
#             p_member = 1.0 / (1.0 + np.exp(-perturbed_logit))
#             ens_preds[:, m, 0] = p_member
#             ens_preds[:, m, 1] = 1 - p_member
#         self.ens_preds = ens_preds
        
#         # 5) Convert labels and x to torch tensors
#         self.y_labels = torch.from_numpy(y_array).long()
#         self.x_inst = torch.from_numpy(x_array).float().reshape(-1, 1)

# class BinaryExperiment:
#     """
#     A unified experiment class for binary classification that can
#     generate data in two ways:

#     - 'gp': Using a Gaussian Process to sample the "true" function p(Y=1|x),
#             then optionally shifting/adding noise for ensemble members.
#     - 'logistic': Using a mixture of Gaussians for X conditioned on Y,
#                   and a logistic link as the "true" function. Then create
#                   ensemble predictions by random offsets, etc.

#     Attributes
#     ----------
#     method : str
#         Either "gp" or "logistic".
#     n_samples : int
#         Number of samples to generate.
#     n_ens : int
#         Number of ensemble members.
#     scale_noise : float
#         Controls the amplitude of random noise for ensemble predictions.
#     kernel : callable
#         Kernel function if method == "gp".
#     n_classes : int
#         Typically 2 for binary.

#     After calling `generate_data()`, you get:
#     - self.x_inst : (n_samples,) or (n_samples,1) array/tensor of inputs
#     - self.p_true : (n_samples, 2) array with true probabilities
#     - self.ens_preds : (n_samples, n_ens, 2) array with ensemble predictions
#     - self.y_labels : (n_samples,) sampled labels
#     """

#     def __init__(
#         self,
#         method="gp",
#         n_samples=1000,
#         n_ens=5,
#         scale_noise=0.5,
#         kernel=rbf_kernel,
#         # logistic mixture parameters:
#         mixture_loc=(-1.0, +1.0),    # location of Gaussians for Y=0, Y=1
#         mixture_std=1.0,
#         offset_range : list = [-0.0, 4.0],
#         # for GP kernel
#         kernel_width=0.05
#     ):
#         self.method = method.lower()
#         self.n_samples = n_samples
#         self.n_ens = n_ens
#         self.scale_noise = scale_noise
#         self.kernel = kernel
#         self.n_classes = 2

#         # if logistic mixture approach:
#         self.mixture_loc = mixture_loc
#         self.mixture_std = mixture_std
#         self.offset_range = offset_range
#         # extra param for GP kernel
#         self.kernel_width = kernel_width
#         self.deterministic_ens = True

#         # placeholders
#         self.x_inst = None
#         self.p_true = None
#         self.ens_preds = None
#         self.y_labels = None

#     def generate_data(self, **kwargs):
#         """
#         Main entry: generate data according to 'method'.
#         """
#         if self.method == "gp":
#             self._generate_data_gp(**kwargs)
#         elif self.method == "logistic":
#             self._generate_data_logistic(**kwargs)
#         else:
#             raise ValueError(f"Unknown method: {self.method}")

#     ##################################################
#     # GP-based data generation (existing approach)
#     ##################################################
#     def _generate_data_gp(self, low=0.0, high=5.0, shift_fn=None):
#         """
#         Uses your existing approach: sample x in [low, high], 
#         build a GP, sample logistic function => p_true, 
#         then create ensemble with shifts/noise.
#         """
#         # 1) sample x
#         self.x_inst = np.random.uniform(low, high, self.n_samples)
#         # 2) generate ground truth p_true
#         self.p_true = self._generate_gt_gp(self.x_inst)
#         # 3) ensemble predictions
#         if shift_fn is None:
#             shift_fn = self._default_shift_fn
#         self.ens_preds = self._generate_ens_preds_gp(self.x_inst, self.p_true[:, 0], shift_fn)
#         # 4) sample labels
#         self.y_labels = multinomial_label_sampling(self.p_true, tensor=True)
#         # 5) store x_inst as a float tensor
#         self.x_inst = torch.from_numpy(self.x_inst).float().view(-1, 1)

#     def _generate_gt_gp(self, x_inst):
#         """
#         Create a GP sample -> logit -> p_true.
#         """
#         p_true = np.zeros((x_inst.shape[0], 2))
#         # build kernel matrix
#         dist = x_inst.reshape(-1, 1) - x_inst.reshape(1, -1)
#         K = self.kernel(dist, dist, gamma=self.kernel_width)
#         K += 1e-6 * np.eye(K.shape[0])

#         # sample GP
#         f_vals = np.random.multivariate_normal(np.zeros(x_inst.shape[0]), K)
#         # apply sigmoid
#         p_true_1 = expit(f_vals)
#         p_true[:, 0] = p_true_1
#         p_true[:, 1] = 1 - p_true_1
#         return p_true

#     def _default_shift_fn(self, x):
#         return .5* np.sin(np.pi * x) + .5

#     def _generate_ens_preds_gp(self, x_inst, p_true, shift_fn):
#         """
#         Create ensemble predictions by shifting the logit of p_true
#         and adding correlated noise.
#         """
#         p_true_logit = np.log(p_true / (1 - p_true + 1e-12) + 1e-12)
#         shift_vals = shift_fn(x_inst)
#         m_logit = p_true_logit #+ shift_vals

#         ens_preds = np.zeros((x_inst.shape[0], self.n_ens, self.n_classes))
#         for m in range(self.n_ens):
#             # random offset
#             offset = np.random.uniform(self.offset_range[0], self.offset_range[1])
#             # repeat for all samples
#             # correlated noise from GP
#             noise = self._sample_gp_noise(x_inst)
#             noise_scaled = noise * self.scale_noise

#             logit_m = m_logit + noise_scaled + offset
#             z1 = 1.0 / (1.0 + np.exp(-logit_m))
#             z2 = 1.0 - z1
#             ens_preds[:, m, 0] = z1
#             ens_preds[:, m, 1] = z2

#         return ens_preds

#     def _sample_gp_noise(self, x_inst):
#         """
#         Sample correlated noise from a GP (RBF).
#         """
#         dist = x_inst.reshape(-1, 1) - x_inst.reshape(1, -1)
#         K = self.kernel(dist, dist, gamma=self.kernel_width)
#         K += 1e-8 * np.eye(K.shape[0])
#         noise = np.random.multivariate_normal(np.zeros(x_inst.shape[0]), K)
#         return noise

#     ##################################################
#     # Logistic mixture approach (like a 2-class mixture of Gaussians for X)
#     ##################################################
#     def _generate_data_logistic(self):
#         """
#         Generate data from Y=0 or 1 w.p. 0.5, 
#         then X ~ N(mixture_loc[y], mixture_std^2).
#         The 'true' p_true is logistic in form, or direct from the ground truth.

#         Then produce ensemble predictions by offsetting logistic parameters.
#         """
#         # 1) sample y in {0,1} (balanced)
#         y_array = np.random.binomial(n=1, p=0.5, size=self.n_samples)

#         # 2) sample x from N(mixture_loc[y], mixture_std^2)
#         x_array = np.zeros_like(y_array, dtype=float)
#         mask0 = (y_array == 0)
#         mask1 = (y_array == 1)
#         x_array[mask0] = np.random.normal(self.mixture_loc[0], self.mixture_std, mask0.sum())
#         x_array[mask1] = np.random.normal(self.mixture_loc[1], self.mixture_std, mask1.sum())

#         # 3) ground-truth p_true(Y=1|x). For logistic: p = 1/(1+exp(2x)) if you want that from eqn(13) 
#         #    or define your own logistic function. Let's do the eqn(13) style: p= 1/(1+exp(2*x))
#         p = 1.0 / (1.0 + np.exp(2.0 * x_array))  # shape (n_samples,)
#         p_true = np.stack([p, 1 - p], axis=1)

#         self.x_inst = x_array
#         self.y_labels = torch.from_numpy(y_array).long()
#         self.p_true = p_true

#         # 4) ensemble predictions with random offsets around some logistic param
#         if self.deterministic_ens:
#             self.ens_preds = self._generate_ens_preds_logistic_more_deterministic(x_array)
#         else:
#             self.ens_preds = self._generate_ens_preds_logistic(x_array, p)

#         # store x_inst as torch
#         self.x_inst = torch.from_numpy(x_array).float().view(-1,1)

#     def _generate_ens_preds_logistic(self, x_array, p_array):
#         """
#         Example: we treat p_array as the 'center' logistic, then produce K members
#         by adjusting the logit with random offsets. 
#         """
#         # logit(center) = - log(1/p -1)
#         # or direct: logit_c = log(p/(1-p))
#         eps = 1e-12
#         logit_center = np.log(p_array + eps) - np.log(1 - p_array + eps)

#         ens_preds = np.zeros((len(x_array), self.n_ens, self.n_classes))
#         # sample location of offset (mean of normal)
#         loc = np.random.normal(loc=2.0, scale=1.0, size=len(x_array))
#         for k in range(self.n_ens):
#             # random offset (like a small normal or uniform)
#             offset = np.random.normal(loc=loc, scale=self.scale_noise, size=len(x_array))
#             # apply offset
#             logit_k = logit_center + offset
#             z1 = 1.0 / (1.0 + np.exp(-logit_k))
#             z2 = 1.0 - z1
#             ens_preds[:,k,0] = z1
#             ens_preds[:,k,1] = z2

#         return ens_preds
    
#     def _generate_ens_preds_logistic_more_deterministic(self, x_array):
#         """
#         Creates K logistic-based ensemble members, each with a distinct but fixed
#         slope/intercept. 
#         """
#         N = len(x_array)
#         ens_preds = np.zeros((N, self.n_ens, self.n_classes), dtype=np.float32)

#         # 1) For each ensemble member, sample once:
#         #    (beta0_k, beta1_k).
#         #    For instance, sample slope from ~ Unif(1,3), intercept from ~ Unif(-2,2).
#         beta0 = np.random.uniform(low=-2.0, high=5.0, size=self.n_ens)
#         beta1 = np.random.uniform(low=.5, high=6.0, size=self.n_ens)

#         # 2) For each x_i, compute f_k(x_i) = beta0_k + beta1_k * x
#         for k in range(self.n_ens):
#             f_k = beta0[k] + beta1[k] * x_array  # shape (N,)
#             # 3) logistic transform
#             z1 = 1.0 / (1.0 + np.exp(f_k))
#             z2 = 1.0 - z1
#             ens_preds[:, k, 0] = z1
#             ens_preds[:, k, 1] = z2

#         return ens_preds



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

        