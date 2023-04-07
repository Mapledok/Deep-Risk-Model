"""
@Time : 2023/3/20 21:12
@Author : 十三
@Email : mapledok@outlook.com
@File : Temporal_Transformation.py
@Project : My_python
"""
import torch
import torch.nn as nn


class TtRiskFactor(nn.Module):
    """
        Generate K/2 risk factors from historical sequential characteristics.
        Args:

        Inputs:
            batch_returns  (batch_size, period_size, stock_size, risk_factor_size): batch arrays

        Returns:
            torch.Tensor (batch_size, stock_size, risk_factor_size): The K/2 risk factors of stocks
    """

    def __init__(self,
                 features_size,
                 hidden_size,
                 risk_factor_size,
                 num_layers,
                 stock_size,
                 ):
        super().__init__()
        self.features_size = features_size
        self.hidden_size = hidden_size
        self.risk_factor_size = risk_factor_size
        self.num_layers = num_layers
        self.stock_size = stock_size

        self.gru = nn.GRU(input_size=features_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.FC = nn.Linear(in_features=hidden_size,
                            out_features=risk_factor_size)
        self.norm = nn.BatchNorm1d(num_features=stock_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        period_size = input_tensor.shape[1]
        # ''' (batch_size, period_size, stock_size, features_size) -->
        #     (batch_size, stock_size, period_size, features_size) -->
        #     (batch_size*stock_size, period_size, features_size) '''
        input_tensor_ = input_tensor.permute((0, 2, 1, 3)).contiguous().view(-1,
                                                                             period_size,
                                                                             self.features_size)
        # ''' (batch_size*stock_size, period_size, features_size) -->
        #     hidden: (num_layers, batch_size*stock_size, hidden_size) '''
        _, hidden = self.gru(input_tensor_)
        # ''' (num_layers, batch_size*stock_size, hidden_size) -->
        #     (batch_size, stock_size, hidden_size) -->
        #     (batch_size, stock_size, risk_factor_size) '''
        hidden_ = self.FC(hidden.view(-1, self.stock_size, self.hidden_size))
        # ''' (batch_size, stock_size, risk_factor_size) -->
        #     (batch_size, stock_size, risk_factor_size) '''
        Tt_risk_factors = self.norm(hidden_)

        return Tt_risk_factors


# test_data = torch.randn(size=(16, 60, 35, 10))
#
# TRF = TtRiskFactor(
#     features_size=10,
#     hidden_size=32,
#     risk_factor_size=3,
#     num_layers=1,
#     stock_size=35,
# )
#
# factors = TRF(test_data)
# print(factors.size())
