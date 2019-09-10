import time
import numpy as np

from uq_kernel.monomial_example import create_monomial_models, \
    get_monomial_model_inputs, is_monomial_output_correct


def simple_serial_model_evaluation(models, model_inputs):
    model_outputs = []
    for model, multiple_inputs in zip(models, model_inputs):
        multiple_outputs = [model.evaluate(inpt) for inpt in multiple_inputs]
        model_outputs.append(np.array(multiple_outputs))
    return model_outputs


def check_model_execution(num_models: int, max_cost: float,
                          model_cost_ratio: float, cost_std_ratio: float,
                          target_cost: float):

    models = create_monomial_models(num_models, max_cost, model_cost_ratio,
                                    cost_std_ratio)
    model_inputs = get_monomial_model_inputs(models, target_cost)

    t_start = time.time()
    model_outputs = simple_serial_model_evaluation(models, model_inputs)
    t_end = time.time()

    success = is_monomial_output_correct(models, model_inputs, model_outputs)

    print("Model Execution Results:")
    print("  Output is", "Correct!" if success else "Incorrect!")
    print("  Target run time was", target_cost)
    print("  Actual run time was", t_end - t_start)


if __name__ == "__main__":
    NUM_MODELS = 5          # reasonable range [2, 10]
    MAX_COST = 1.0          # time to run longest model (sec)
    MODEL_COST_RATIO = 0.1  # reasonable range [0.001, 0.5]
    COST_STD_RATIO = 0.05   # reasonable range [0.01, 0.2]
    TARGET_COST = 10        # reasonable range [5, 1000]*MAX_COST

    check_model_execution(NUM_MODELS, MAX_COST, MODEL_COST_RATIO,
                          COST_STD_RATIO, TARGET_COST)
