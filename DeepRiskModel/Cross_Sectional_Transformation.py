"""
@Time : 2023/3/21 12:32
@Author : 十三
@Email : mapledok@outlook.com
@File : Cross_Sectional_Transformation.py
@Project : deepquant
"""
import torch
import torch.nn as nn
from DeepRiskModel.Graph_Attention_Networks import GraphAttentionNetworks
from DeepRiskModel.Temporal_Transformation import TtRiskFactor


class CStRiskFactor(nn.Module):
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
                 dropout
                 ):
        super().__init__()
        self.features_size = features_size
        self.hidden_size = hidden_size
        self.risk_factor_size = risk_factor_size
        self.num_layers = num_layers
        self.stock_size = stock_size

        self.gat = GraphAttentionNetworks(features_size=features_size,
                                          dropout=dropout)
        self.gru = TtRiskFactor(features_size=features_size,
                                hidden_size=hidden_size,
                                risk_factor_size=risk_factor_size,
                                num_layers=num_layers,
                                stock_size=stock_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # ''' (batch_size, period_size, stock_size, features_size) -->
        #     gat_output: (batch_size, period_size, stock_size, features_size) '''
        gat_output, _ = self.gat(input_tensor, input_tensor, input_tensor)
        # ''' (batch_size, period_size, stock_size, features_size) -
        #     (batch_size, period_size, stock_size, features_size) -->
        #     (batch_size, period_size, stock_size, features_size) '''
        input_tensor_ = input_tensor - gat_output
        # ''' (batch_size, period_size, stock_size, features_size) -->
        #     (batch_size, period_size, risk_factor_size) '''
        CSt_risk_factors = self.gru(input_tensor_)

        return CSt_risk_factors


# test_data = torch.randn(size=(16, 60, 35, 10))
#
# CST = CStRiskFactor(
#     features_size=10,
#     hidden_size=32,
#     risk_factor_size=3,
#     num_layers=1,
#     stock_size=35,
#     dropout=0.5
# )
#
# factors = CST(test_data)
# print(factors.size())

