import torch
import numpy as np
from torch import nn 
from ensemblecalibration.calibration.test_objectives import calculate_pbar, skce_ul_arr
from ensemblecalibration.nn_training.distances import skce_ul_tensor, tv_distance, median_heuristic
from ensemblecalibration.nn_training.helpers import calculate_pbar_torch                                                  

class SKCEuqLoss(nn.Module):

    def __init__(self, use_median_bw: bool = False, bw: float = 2.0, dist_fct = tv_distance) -> None:
        """_summary_

        Parameters
        ----------
        use_median_bw : bool, optional
            _description_, by default False
        bw : float, optional
            _description_, by default 2.0
        dist_fct : _type_, optional
            _description_, by default tv_distance
        """
        super().__init__()
        self.bw = bw
        self.use_median_bw = use_median_bw
        self.dist_fct = dist_fct

    def forward(self, p_preds: torch.Tensor, weights_l: torch.Tensor, y: torch.Tensor):
        # calculate matrix of point predictions of the convex combination
        p_bar = calculate_pbar_torch(weights_l=weights_l, p_preds=p_preds, reshape=False)
        if self.use_median_bw:
            bw = median_heuristic(p_hat=p_bar, y_labels=y)
        else:
            bw = self.bw
        # get tensor of SKCE values which is to be summed over
        hat_skce_ul = skce_ul_tensor(p_bar=p_bar, y=y, dist_fct=self.dist_fct, sigma=bw)
        # calculate mean 
        loss = torch.mean(hat_skce_ul)

        return loss
    
if __name__ == "__main__":
    loss = SKCEuqLoss()
    p = torch.from_numpy(np.random.dirichlet([1]*10, size=(100,10)))
    lambdas = torch.from_numpy(np.random.dirichlet([1]*10, size=100))
    y = torch.randint(7, size=(100,))
    print(y)
    out = loss(p, lambdas, y)
    print(out)
        

