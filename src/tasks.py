"""Generate the 25 budgetary decision tasks per subject.

Following Chen et al. (2023) and Choi et al. (2007):
  - Prices (M, N) are drawn from the unit interval such that M, N in [0.1, 1]
    and max(M, N) >= 0.5, kept to two decimals.
  - Each subject completes 25 such (M, N) tasks per preference domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class BudgetTask:
    round_idx: int          # 1..25
    M: float                # commodity-A price parameter
    N: float                # commodity-B price parameter

    @property
    def pA(self) -> float:
        # 论文里 commodity 价格定义为 1/M 与 1/N (花 1/M 点能换 1 美元 →
        # 每点价格 = M 美元的"反价格"，预算分析里 pA = 1/M, pB = 1/N，
        # 这样 pA*xA + pB*xB = 100 等价于 xA*M + xB*N = total payoff)。
        return 1.0 / self.M

    @property
    def pB(self) -> float:
        return 1.0 / self.N


def generate_tasks(num_subjects: int,
                   rounds_per_subject: int = 25,
                   seed: int = 20231212) -> List[List[BudgetTask]]:
    """Return a list of `num_subjects` task sets, each of length 25.

    The same seed reproduces the same task sequences, which lets different
    LLMs face an identical questionnaire.
    """
    rng = np.random.default_rng(seed)
    all_subjects: List[List[BudgetTask]] = []
    for _ in range(num_subjects):
        tasks: List[BudgetTask] = []
        for r in range(1, rounds_per_subject + 1):
            while True:
                M = round(float(rng.uniform(0.1, 1.0)), 2)
                N = round(float(rng.uniform(0.1, 1.0)), 2)
                if max(M, N) >= 0.5:
                    break
            tasks.append(BudgetTask(round_idx=r, M=M, N=N))
        all_subjects.append(tasks)
    return all_subjects
