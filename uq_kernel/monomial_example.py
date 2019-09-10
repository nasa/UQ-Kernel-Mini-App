from time import sleep
from typing import List, Tuple
import numpy as np

from uq_kernel.model import UQModel


class MonomialModel(UQModel):
    """
    An exemplary implementation of the `UQModel` that can be used for testing
    """

    def __init__(self, exponent: int, approx_cost: float, cost_std: float):
        """
        Initialize monomial model state

        :param exponent: the exponent to raise the input
        :param approx_cost: the approximate time for evaluation of the model
        :param cost_std: the standard deviation of the evaluation time
        """
        self._approximate_cost = approx_cost
        self._cost_std = cost_std
        self.exponent = exponent

    @property
    def cost(self) -> float:
        """the approximate evaluation time"""
        return self._approximate_cost

    def evaluate(self, random_input: np.ndarray) -> np.ndarray:
        """
        Evaluate the monomial model at a given random input.  The models output
        is the input raised to an exponent, y = x^e

        :param random_input: array of inputs, x
        :return: corresponding output, x^e
        """
        actual_cost = np.random.normal(self._approximate_cost,
                                       self._cost_std)
        sleep(actual_cost)
        output = np.array(random_input ** self.exponent)
        return output


def create_monomial_models(num_models: int, max_cost: float,
                           model_cost_ratio: float = 0.1,
                           cost_std_ratio: float = 0.1) -> List[MonomialModel]:
    models = []
    for i in range(num_models):
        exponent = num_models - i
        approx_cost = max_cost * model_cost_ratio**i
        cost_std = approx_cost*cost_std_ratio
        models.append(MonomialModel(exponent, approx_cost, cost_std))
    return models


def get_monomial_model_inputs(models: List[MonomialModel],
                              target_cost: float) -> List[np.ndarray]:
    stdevs, correlation = _get_monomial_std_and_correlation(models)
    costs = [model.cost for model in models]
    num_samples = _calculate_optimal_num_samples(stdevs, correlation, costs,
                                                 target_cost)
    full_rand_inputs = np.random.random(num_samples[-1])
    rand_inputs = [full_rand_inputs[:i] for i in num_samples]
    return rand_inputs


def _get_monomial_std_and_correlation(models: List[MonomialModel]
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    num_models = len(models)
    cov = np.empty((num_models, num_models))
    for i, model_i in enumerate(models):
        p_i = model_i.exponent
        for j, model_j in enumerate(models):
            p_j = model_j.exponent
            cov[i, j] = 1.0 / (p_i + p_j + 1) - 1.0 / ((p_i + 1) * (p_j + 1))
    stdevs = np.sqrt(np.diag(cov))
    cov /= stdevs
    cov /= stdevs.reshape((-1, 1))
    return stdevs, cov


def _calculate_optimal_num_samples(stdevs: np.ndarray, correlation: np.ndarray,
                                   costs: List[float], target_cost: float
                                   ) -> List[int]:
    num_models = len(costs)
    sample_ratios = [1]
    for i in range(1, num_models):
        corr_i = correlation[0, i]
        corr_i_plus_1 = 0 if i == num_models - 1 else correlation[0, i + 1]
        ratio = np.sqrt(costs[0] * (corr_i**2 - corr_i_plus_1**2) /
                        (costs[i] * (1 - correlation[0, 1]**2)))
        sample_ratios.append(ratio)

    sample_nums = [target_cost / np.dot(costs, sample_ratios)]
    for ratio in sample_ratios[1:]:
        sample_nums.append(sample_nums[0]*ratio)

    alpha_star = [0] + [correlation[0, i] * stdevs[0] / stdevs[i]
                        for i in range(1, len(stdevs))]
    estimator_variance = stdevs[0]**2 / sample_nums[0]
    for i in range(1, num_models):
        estimator_variance += (1/sample_nums[i-1] - 1/sample_nums[i]) \
                              * (alpha_star[i]**2 * stdevs[i]**2
                                 - 2 * alpha_star[i] * correlation[0, i]
                                 * stdevs[0] * stdevs[i])
    print(estimator_variance)
    return [int(num) for num in sample_nums]


def plate():
    stdevs = np.sqrt([1.4968e-2,
                      1.4967e-2,
                      #1.4975e-2,
                      #1.4946e-2,
                      #1.7429e-2,
                      #1.4298e-2
                      ])
    corr = np.array([[1.0,
                      0.999999986,
                      #0.999999238,
                      #0.999948132,
                      #0.998580437,
                      #0.996563656
                      ]])
    costs = [4.0894e-1,
             4.9890e-3,
             #1.3264e-3,
             #2.9550e-4,
             #2.2260e-5,
             #1.5048e-6
             ]
    target_cost = 1
    _calculate_optimal_num_samples(stdevs, corr, costs, target_cost)

