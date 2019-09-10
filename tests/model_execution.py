import numpy as np

from uq_kernel.monomial_example import create_monomial_models, \
    get_monomial_model_inputs


def check_model_execution(num_models: int, max_cost: float,
                          model_cost_ratio: float, cost_std_ratio: float,
                          target_cost: float):

    models = create_monomial_models(num_models, max_cost, model_cost_ratio,
                                    cost_std_ratio)
    model_inputs = get_monomial_model_inputs(models, target_cost)


if __name__ == "__main__":
    NUM_MODELS = 5          # reasonable range [2, 10]
    MAX_COST = 1.0          # time to run longest model (sec)
    MODEL_COST_RATIO = 0.1  # reasonable range [0.001, 0.5]
    COST_STD_RATIO = 0.05   # reasonable range [0.01, 0.2]
    TARGET_COST = 10        # reasonable range [5, 1000]*MAX_COST

    check_model_execution(NUM_MODELS, MAX_COST, MODEL_COST_RATIO,
                          COST_STD_RATIO, TARGET_COST)
