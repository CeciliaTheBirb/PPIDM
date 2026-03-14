import torch
import torch.nn.functional as F

#import einops as ein
        
class ResidualsADR:
    def __init__(self, model, fd_acc, pixels_per_dim, pixels_at_boundary, reverse_d1, device = 'cpu', bcs = 'none', domain_length = 1., residual_grad_guidance = False, use_ddim_x0 = False, ddim_steps = 0):
        """
        Initialize the residual evaluation.

        :param model: The neural network model to compute the residuals for.
        :param n_steps: Number of steps for time discretization.
        """
        self.gov_eqs = 'adr'
        self.model = model
        self.pixels_at_boundary = pixels_at_boundary
        self.periodic = False
        self.input_dim = 2
 
    def compute_residual(self, f_pred, input_all, dx=1, dy=1, dt=3):
        """
        Compute the residual of the advection equation as a loss term.
        
        """
        d = input_all[:,0]
        u = input_all[:,1]
        v = input_all[:,2]
        def compute(data, u, v, dt=3):
            f_pad = F.pad(data, (1, 1, 1, 1, 0, 0), mode='replicate')

            f_x = (f_pad[:, :, :, 1:-1, 2:] - f_pad[:, :, :, 1:-1, :-2]) / (2 * dx)
            f_y = (f_pad[:, :, :, 2:, 1:-1] - f_pad[:, :, :, :-2, 1:-1]) / (2 * dy)
        
            f_t = torch.zeros_like(data)
            f_t[:, :, 0, :, :] = (data[:, :, 1, :, :] - data[:, :, 0, :, :]) / dt  
            f_t[:, :, 1:-1, :, :] = (data[:, :, 2:, :, :] - data[:, :, :-2, :, :]) / (2 * dt)  
            f_t[:, :, -1, :, :] = (data[:, :, -1, :, :] - data[:, :, -2, :, :]) / dt  
            residual = f_t + u * f_x + v * f_y
            
            return residual
        
        theory = compute(d, u, v)
        real = compute(f_pred, u, v)
        return (real-theory)**2

        