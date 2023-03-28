import torch

def tv_distance_tensor(p_1: torch.Tensor, p_2: torch.Tensor):
    """total variation distance between two point predictions.

    Parameters
    ----------
    p_1 : torch.Tensor
        point estimate of shape (n_classes,)
    p_2 : torch.Tensor
        second point estimate of shape (n_classes,)

    Returns
    -------
    float
        variation distance
    """

    return 0.5*torch.sum(torch.abs(p_1-p_2))

def l2_distance(p_1: torch.Tensor, p_2: torch.Tensor):
    """L" distance between two point predictions given as torch.Tensors.

    Parameters
    ----------
    p_1 : torch.Tensor
        first point prediction
    p_2 : torch.Tensor
        second point porediction

    Returns
    -------
    float
        distance
    """

    return torch.sqrt(torch.sum((p_1 - p_2)**2))

def tensor_kernel(p: torch.Tensor, q: torch.Tensor, dist_fct=tv_distance_tensor, sigma: float = 2.0):
    """returns the matrix-valued kernel evaluated at two point predictions 

    Parameters
    ----------
    p : torch.Tensor
        first point prediction 
    q : torch.Tensor
        second point prediction
    sigma : float
        bandwidth 
    dist_fct : _type_
        distance measure. Options: {tv_distance, l2_distance}

    Returns
    -------
    torch.Tensor
        _description_
    """
    p = p.squeeze()
    q = q.squeeze()

    assert len(p) == len(q), "vectors need to be of the same length"
    id_k = torch.eye(len(p)) # identity matrix
    return torch.exp((-1/sigma)* (dist_fct(p, q)**2)* id_k)


def tensor_h_ij(p_i: torch.Tensor, p_j: torch.Tensor, y_i: torch.Tensor, y_j: torch.Tensor, dist_fct,
        sigma: float=2.0):
    """calculates the entries h_ij which are summed over in the expression of the calibration

    Parameters
    ----------
    p_i : torch.Tensor

        first point prediction
    p_j : torch.Tensor
        second point prediction
    y_i : torch.Tensor
        one hot encoding of labels for sample j
    y_j : torch.Tensor
        one hot encoding of labels for sample j
    dist_fct : 
        function used as a distance measure in the matrix valued kernel
    sigma : float, optional
        bandwidth, by default 2.0

    Returns
    -------
    torch.Tensor

    """
    gamma_ij = tensor_kernel(p_i, p_j, dist_fct=dist_fct, sigma=sigma).float()
    y_ii = (y_i - p_i).float()
    y_jj = (y_j - p_j).float()

    h_ij = torch.matmul(y_ii, torch.matmul(gamma_ij,y_jj))

    return h_ij

def skce_ul_tensor(p_bar: torch.Tensor, y: torch.Tensor, dist_fct= tv_distance_tensor, sigma: float = 2.0):
    """calculates the skce_ul calibration error used as a test statistic in Mortier et  al, 2022.

    Parameters
    ----------
    P_bar :  torch.Tensor of shape (n_predictors, n_classes)
        matrix containing probabilistic predictions for each instance 
    y : torch.Tensor
        vector with class labels of shape
    dist_fct : [tv_distance, l2_distance]
        distance function to be used
    sigma : float
        bandwidth used in the matrix valued kernel

    Returns
    -------
    torch.Tensor
        _description_
    """

    n = round(p_bar.shape[0]/2)
    # transform y to one-hot encoded labels
    yoh = torch.eye(p_bar.shape[1])[y,:]
    stats = torch.zeros(n)
    for i in range(0,n):
        stats[i] = tensor_h_ij(p_bar[(2*i),:], p_bar[(2*i)+1,:], yoh[(2*i),:], 
                               yoh[(2*i)+1,:], dist_fct=dist_fct, sigma=sigma)

    return stats

def skce_uq_tensor(p_bar: torch.Tensor, y: torch.Tensor, dist_fct=tv_distance_tensor, sigma: float = 2.0):
    """calculates the SKCEuq miscalibration measure introduced in Widman et al, 2019.

    Parameters
    ----------
    p_bar : torch.Tensor
        tensor containing all 
    y : torch.Tensor
        _description_
    dist_fct : _type_, optional
        _description_, by default tv_distance_tensor
    sigma : float, optional
        _description_, by default 2.0

    Returns
    -------
    _type_
        _description_
    """
    
    N, M = p_bar.shape[0], p_bar.shape[1] # p is of shape (n_samples, m_predictors, n_classes)
    # one-hot encoding
    y_one_hot =torch.eye(M)[y, :]

    # binomial coefficient n over 2
    stats = torch.zeros(int((N*(N-1))/2))
    count=0
    for j in range(1, N):
        for i in range(j):
            stats[count] = tensor_h_ij(p_bar[i, :], p_bar[j, :], y_one_hot[i, :], y_one_hot[j,:],
                                 dist_fct=dist_fct, sigma=sigma)
            count+=1

    return stats


def median_heuristic(p_hat: torch.Tensor, y_labels: torch.Tensor):
    """calculates the optimal bandwidth of the kernel used in the SKCE using a median heuristic,
    where the pairwise distances of the predicted labels and the real labels are calculated and the 
    median is taken as a reference bandwidth.

    Parameters
    ----------
    p_hat: torch.Tensor
        tensor of predicted probabilities
    y_labels : torch.Tensor
        tensor of real labels

    Returns
    -------
    float
        bandwidth
    """
    # get predictions of the model
    y_pred = torch.argmax(p_hat, dim=1)
    print(y_pred)
    # reshape to two dimensions
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_labels = y_labels.view(y_labels.shape[0], -1)
    dist = torch.nn.functional.pairwise_distance(y_pred, y_labels)
    sigma_bw = dist.median() / 2

    return sigma_bw