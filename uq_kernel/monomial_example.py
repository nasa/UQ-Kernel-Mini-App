from time import sleep, time
from scipy.stats import norm as normal_distribution
import numpy as np

from uq_kernel.model import UQModel


class MonomialModel(UQModel):
    """
    An exemplary implementation of the `UQModel` that can be used for testing
    """

    def __init__(self, exponent, approx_cost, cost_std):
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
    def cost(self):
        """the approximate evaluation time"""
        return self._approximate_cost

    def evaluate(self, inputs):
        """
        Evaluate the monomial model at a given random input.  The models output
        is the input raised to an exponent, y = x^e

        :param random_input: array of input, x, in the interval (0, 1)
        :return: corresponding output, x^e
        """
        t_0 = time()
        actual_cost = self.calculate_actual_cost(inputs)[0]
        output = np.array(inputs ** self.exponent)
        actual_cost -= time() - t_0
        if actual_cost > 0:
            sleep(actual_cost)
        return output

    def calculate_actual_cost(self, inputs):
        z = normal_distribution.ppf(inputs)
        actual_cost = self._approximate_cost + z * self._cost_std
        return actual_cost


def create_models(num_models, max_cost, model_cost_ratio=0.1, 
                  cost_std_ratio=0.1):
    """
    Creates monomial models to be used in model execution tests

    :param num_models (int): number of models to create
    :param max_cost (float): cost of the highest cost model, models[0]
    :param model_cost_ratio (float): multiplicative model costs for each 
        successive model
    :param cost_std_ratio (float): amount of variation in evaluation time as a 
        fraction of evaluation time
    :return: a list of `MonomialModel`s 
    """
    models = []
    for i in range(num_models):
        exponent = num_models - i
        approx_cost = max_cost * model_cost_ratio**i
        cost_std = approx_cost*cost_std_ratio
        models.append(MonomialModel(exponent, approx_cost, cost_std))
    return models


def get_model_inputs(models, target_cost):
    """
    Gets the arrays of inputs for monomial models according to multi-fidelity
    monte carlo

    :param models (list): the `MonomialModel`s 
    :param target_cost (float): target total cost of all model evaluations
    :return: list of arrays of inputs for the `MonomialModel`
    """
    _, correlation = _get_monomial_std_and_correlation(models)
    costs = [model.cost for model in models]
    num_samples = _calculate_optimal_num_samples(correlation, costs,
                                                 target_cost)
    full_rand_inputs = np.random.random(num_samples[-1])
    rand_inputs = [full_rand_inputs[:i].reshape((-1, 1)) for i in num_samples]
    return rand_inputs


def _get_monomial_std_and_correlation(models):
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


def _calculate_optimal_num_samples(correlation, costs, target_cost):
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


def is_output_correct(models, model_input, model_output):
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


def _get_correct_output(models, model_input):
    return [inp**mod.exponent for mod, inp in zip(models, model_input)]


def get_execution_time(models, model_input):
    """
    Calculate the theoretical execution time of a group of models with given
    inputs.

    :param models: `MonomialModel`s that were evaluated
    :param model_input: The inputs for each `MonomialModel`
    :return: theoretical execution time
    """
    theoretical_time = 0
    for model, inpts in zip(models, model_input):
        theoretical_time += np.sum(model.calculate_actual_cost(inpts))
    return theoretical_time
