import pytest
import time
import numpy as np

from uq_kernel.monomial_example import MonomialModel


@pytest.mark.parametrize("approx_cost", [0.3, 0.2, 0.1])
def test_setting_cost(approx_cost):
    model = MonomialModel(1, approx_cost, 0.01)
    assert model.cost == approx_cost


@pytest.mark.parametrize("approx_cost", [0.1, 0.05])
def test_evaluation_time(approx_cost):
    model = MonomialModel(1, approx_cost, 0.00)
    random_input = np.zeros(1)

    t_0 = time.time()
    for _ in range(5):
        _ = model.evaluate(random_input)
    eval_time = (time.time() - t_0)/5
    assert eval_time == pytest.approx(approx_cost, abs=0.005)


@pytest.mark.parametrize("cost_std", [0.009, 0.015])
def test_variation_in_evaluation_time(cost_std):
    model = MonomialModel(1, 0.04, cost_std)
    random_input = np.zeros(1)

    eval_times = []
    for _ in range(40):
        t_0 = time.time()
        _ = model.evaluate(random_input)
        t_1 = time.time()
        eval_times.append(t_1 - t_0)
    eval_std = np.std(eval_times)
    assert eval_std == pytest.approx(cost_std, rel=0.2)
