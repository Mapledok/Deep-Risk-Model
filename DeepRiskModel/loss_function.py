"""
@Time : 2023/3/21 18:04
@Author : 十三
@Email : mapledok@outlook.com
@File : loss_function.py
@Project : deepquant
"""
import torch
import torchmetrics
import operator
from typing import *
from functools import reduce
from itertools import accumulate
from torch import mean, linalg, square, matmul, trace

torch.manual_seed(seed=13)


class FactorLoss(torchmetrics.Metric):
    higher_is_better = False
    full_state_update = False

    def __init__(self, gamma):
        super().__init__()
        self.add_state(name='explained_variance',
                       default=torch.tensor(0.0),  # Can either be a Tensor or an empty list.
                       dist_reduce_fx='sum')  # 'sum', 'mean', 'cat', 'min' or 'max'
        self.add_state(name='vif',
                       default=torch.tensor(0.0),
                       dist_reduce_fx='sum')
        self.gamma = gamma

    def update(self, factors: torch.Tensor, returns: torch.Tensor):
        T, H = returns.shape[0:2]

        def explained_ratio(n):
            matrix_list = [factors,
                           linalg.inv(matmul(factors.permute(0, 2, 1), factors)),
                           factors.permute(0, 2, 1),
                           returns.permute(1, 0, 2, 3)[n]]
            dev = returns.permute(1, 0, 2, 3)[n] - reduce(operator.matmul, matrix_list)

            return mean(square(linalg.norm(dev, ord=2, dim=(1, 2)))) / \
                mean(square(linalg.norm(matrix_list[-1], ord=2, dim=(1, 2))))

        self.explained_variance += reduce(operator.add, map(lambda h: explained_ratio(h), range(H))) / H

        self.vif += reduce(operator.add, map(lambda t: trace(
            linalg.inv(matmul(factors.permute(0, 2, 1), factors))[t]), range(T))) / T

    def compute(self) -> torch.Tensor:
        return self.explained_variance + self.gamma * self.vif


# stock_returns = torch.randn(16, 20, 35, 10)
# risk_factors = torch.randn(16, 35, 5)
# loss_fun = FactorLoss(gamma=0.01)
# print(loss_fun(risk_factors, stock_returns))


