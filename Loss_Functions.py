import torch

"""
Repository of Loss Functions for LSTM Hydrology Models

Includes:
- Kling Gupta Efficiency: fit indicator common in hydrological sciences that combines the correlation, variability, 
    and bias of predictions, 1 indicates a perfect fit
- Nash Sutcliffe Efficiency: ratio of prediction variance to observation variance, 1 indicates perfect fit
- NSE combined with mean absolute error: to better handle base flow, loss is alpha*NSE + (1-alpha)*MAE
- KGE combined with mean absolute error: to better handle base flow, loss is alpha*KGE + (1-alpha)*MAE

"""

# Kling Gupta Efficiency Loss
class KlingGuptaEfficiencyLoss(torch.nn.Module):
    def __init__(self):
        super(KlingGuptaEfficiencyLoss, self).__init__()

    def forward(self, predictions, targets, alpha):
        predictions = predictions.view(-1)  # Flatten predictions
        targets = targets.view(-1)  # Flatten targets

        mean_pred = torch.mean(predictions)
        mean_obs = torch.mean(targets)
        
        std_pred = torch.std(predictions)
        std_obs = torch.std(targets)
        
        # Calculate correlation manually
        covariance = torch.mean((predictions - mean_pred) * (targets - mean_obs))
        correlation = covariance / (std_pred * std_obs + 1e-8)  # Add small constant for numerical stability
        
        beta = mean_pred / mean_obs
        gamma = (std_pred / mean_pred) / (std_obs / mean_obs)
        
        kge = 1 - torch.sqrt((correlation - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
        return 1 - kge  # Return as loss (to minimize)
    
# Nash-Sutcliffe Efficiency Loss
class NashSutcliffeEfficiencyLoss(torch.nn.Module):
    def __init__(self):
        super(NashSutcliffeEfficiencyLoss, self).__init__()

    def forward(self, predictions, targets, alpha):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        numerator = torch.sum((targets - predictions) ** 2)
        denominator = torch.sum((targets - torch.mean(targets)) ** 2)

        nse = 1 - (numerator / (denominator + 1e-8))  # Small constant to avoid division by zero
        return 1 - nse  # Return 1- NSE for minimization
    
# Weighted NSE MAE Loss
class NSE_MAELoss(torch.nn.Module):
    def __init__(self):
        super(NSE_MAELoss, self).__init__()

    def forward(self, predictions, targets, alpha):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        numerator = torch.sum((targets - predictions) ** 2)
        denominator = torch.sum((targets - torch.mean(targets)) ** 2)

        nse = 1 - (numerator / (denominator + 1e-8))  # Small constant to avoid division by zero

        mae = abs((torch.sum((targets - predictions)))/len(targets))

        combined = alpha * (1-nse) + (1-alpha) * (mae)

        return combined
    
# Weighted KGE MAE Loss
class KGE_MSELoss(torch.nn.Module):
    def __init__(self):
        super(KGE_MSELoss, self).__init__()

    def forward(self, predictions, targets, alpha):
        predictions = predictions.view(-1)  # Flatten predictions
        targets = targets.view(-1)  # Flatten targets

        mean_pred = torch.mean(predictions)
        mean_obs = torch.mean(targets)
        
        std_pred = torch.std(predictions)
        std_obs = torch.std(targets)
        
        # Calculate correlation manually
        covariance = torch.mean((predictions - mean_pred) * (targets - mean_obs))
        correlation = covariance / (std_pred * std_obs + 1e-8)  # Add small constant for numerical stability
        
        beta = mean_pred / mean_obs
        gamma = (std_pred / mean_pred) / (std_obs / mean_obs)
        
        kge = 1 - torch.sqrt((correlation - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

        mae = abs(torch.sum((targets - predictions))/len(targets))

        combined = alpha * (1-kge) + (1-alpha) * (mae) 

        return combined 