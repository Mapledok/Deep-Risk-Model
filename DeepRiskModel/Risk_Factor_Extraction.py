"""
@Time : 2023/3/21 15:39
@Author : 十三
@Email : mapledok@outlook.com
@File : Risk_Factor_Extraction.py
@Project : deepquant
"""
import torch
import torch.nn as nn
from DeepRiskModel.Temporal_Transformation import TtRiskFactor
from DeepRiskModel.Cross_Sectional_Transformation import CStRiskFactor


class RiskFactorExtraction(nn.Module):
    """
        Generate K risk factors from historical sequential characteristics.
        Args:

        Inputs:
            batch_returns  (batch_size, period_size, stock_size, risk_factor_size): batch arrays

        Returns:
            torch.Tensor (batch_size, stock_size, risk_factor_size*2): The K risk factors of stocks
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

        self.TtLayer = TtRiskFactor(features_size=features_size,
                                    hidden_size=hidden_size,
                                    risk_factor_size=risk_factor_size,
                                    num_layers=num_layers,
                                    stock_size=stock_size)
        self.CStLayer = CStRiskFactor(features_size=features_size,
                                      hidden_size=hidden_size,
                                      risk_factor_size=risk_factor_size,
                                      num_layers=num_layers,
                                      stock_size=stock_size,
                                      dropout=dropout)
        self.proj = nn.Linear(in_features=risk_factor_size,
                              out_features=risk_factor_size,
                              bias=False)
        self.norm = nn.BatchNorm1d(num_features=stock_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # ''' (batch_size, period_size, stock_size, features_size) -->
        #     (batch_size, period_size, risk_factor_size) -->
        #     (batch_size, period_size, risk_factor_size) -->
        #     (batch_size, period_size, risk_factor_size) '''
        Tt_Risk_Factors = self.norm(self.proj(self.TtLayer(input_tensor)))
        CSt_risk_Factors = self.norm(self.proj(self.CStLayer(input_tensor)))
        # ''' (batch_size, period_size, risk_factor_size) ||
        #     (batch_size, period_size, risk_factor_size) -->
        #     (batch_size, period_size, risk_factor_size*2) '''
        risk_factors = torch.cat((Tt_Risk_Factors, CSt_risk_Factors), dim=-1)

        return risk_factors


# test_data = torch.randn(size=(16, 20, 35, 10))
#
# RFE = RiskFactorExtraction(
#     features_size=10,
#     hidden_size=32,
#     risk_factor_size=3,
#     num_layers=1,
#     stock_size=35,
#     dropout=0.5
# )
#
# factors = RFE(test_data)
# print(factors.size())
