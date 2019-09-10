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
    """
    Creates monomial models to be used in model execution tests

    :param num_models: number of models to create
    :param max_cost: cost of the highest cost model, models[0]
    :param model_cost_ratio: multiplicative in model costs for each successive
        model
    :param cost_std_ratio: amount of variation in evaluation time as a fraction
        of evaluation time
    :return: a list of `MonomialModel`s
    """
    models = []
    for i in range(num_models):
        exponent = num_models - i
        approx_cost = max_cost * model_cost_ratio**i
        cost_std = approx_cost*cost_std_ratio
        models.append(MonomialModel(exponent, approx_cost, cost_std))
    return models


def get_monomial_model_inputs(models: List[MonomialModel],
                              target_cost: float) -> List[np.ndarray]:
    """
    Gets the arrays of inputs for monomial models acording to multi-fidelity
    monte carlo

    :param models: the `MonomialModel`s
    :param target_cost: target total cost of all model evaluations
    :return: list of arrays of inputs for the `MonomialModel`
    """
    _, correlation = _get_monomial_std_and_correlation(models)
    costs = [model.cost for model in models]
    num_samples = _calculate_optimal_num_samples(correlation, costs,
                                                 target_cost)
    full_rand_inputs = np.random.random(num_samples[-1])
    rand_inputs = [full_rand_inputs[:i].reshape((-1, 1)) for i in num_samples]
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


def _calculate_optimal_num_samples(correlation: np.ndarray,
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

    return [int(num) for num in sample_nums]


def is_monomial_output_correct(models: List[MonomialModel],
                               model_input: List[np.ndarray],
                               model_output: List[np.ndarray]) -> bool:
    """
    Checks whether output from execution matches the expected output of
    monomial models

    :param models: `MonomialModel`s that were evaluated
    :param model_input: The inputs for each `MonomialModel`
    :param model_output: The output from execution of the models
    :return: whether the output is correct
    """
    correct_output = _get_correct_output(models, model_input)
    try:
        for actual, expected in zip(model_output, correct_output):
            np.testing.assert_array_almost_equal(actual, expected)
        return True
    except AssertionError:
        return False


def _get_correct_output(models: List[MonomialModel],
                        model_input: List[np.ndarray],
                        ) -> List[np.ndarray]:
    return [inp**mod.exponent for mod, inp in zip(models, model_input)]