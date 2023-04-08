"""
@Time : 2023/3/21 13:03
@Author : 十三
@Email : mapledok@outlook.com
@File : Graph_Attention_Networks.py
@Project : Deep Risk Model
"""
import torch
import torch.nn as nn
from typing import Tuple


class GraphAttentionNetworks(nn.Module):
    """
        Generate aggregated information and attention distribution from historical sequential characteristics.
        Args:

        Inputs:
            batch_returns (batch_size, period_size, stock_size, risk_factor_size): batch arrays

        Returns:
            torch.Tensor (batch_size, period_size, stock_size, risk_factor_size): aggregated information
            torch.Tensor (batch_size, period_size, stock_size, risk_factor_size): attention distribution
    """

    def __init__(self,
                 features_size,
                 dropout,
                 ):
        super().__init__()
        self.features_size = features_size
        self.dropout = dropout

        self.proj = nn.Linear(in_features=features_size,
                              out_features=features_size,
                              bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.Tensor([features_size]))
        self.leakyRelu = nn.LeakyReLU()

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, stock_size = query.shape[0], query.shape[2]
        # ''' (batch_size, period_size, stock_size, features_size) -->
        #     (batch_size, period_size, stock_size, features_size) '''
        Q, K, V = self.proj(query), self.proj(key), self.proj(value)
        # ''' (batch_size, period_size, stock_size, features_size) *
        #     (batch_size, period_size, features_size, stock_size) -->
        #     (batch_size, period_size, stock_size, stock_size) '''
        a = torch.matmul(Q, K.permute(0, 1, 3, 2))/self.scale  # energy为注意力打分(此处为缩放点积模型)
        if mask is not None:
            a = a.masked_fill(mask == 0, -1e10)
        # ''' (batch_size, period_size, stock_size, stock_size) -->
        #     (batch_size, period_size, stock_size, stock_size) '''
        e = self.leakyRelu(a)
        # ''' (batch_size, period_size, stock_size, stock_size) -->
        #     (batch_size, period_size, stock_size, stock_size) '''
        attention_distribution = torch.softmax(e, dim=-1)
        # ''' (batch_size, period_size, stock_size, stock_size) *
        #     (batch_size, period_size, stock_size, features_size) -->
        #     (batch_size, period_size, stock_size, features_size) '''
        output = self.leakyRelu(torch.matmul(self.dropout(attention_distribution), V))

        return output, attention_distribution


# test_data = torch.randn(size=(16, 60, 35, 10))
#
# GAT = GraphAttentionNetworks(
#     features_size=10,
#     dropout=0.5
# )
#
# x, attention = GAT(test_data, test_data, test_data)
# print(x.size())
