import time
from typing import List, Callable
import numpy as np

from uq_kernel.model import UQModel
from uq_kernel import monomial_example

EngineFunction = Callable[[List[UQModel], List[np.ndarray]],
                          List[np.ndarray]]


def simple_serial_engine(models: List[UQModel],
                         model_inputs: List[np.ndarray]
                         ) -> List[np.ndarray]:
    """
    Evaluates each of the models at their given inputs one after another.
    Creating an alternative to this function which executes in parallel and
    balances loads is one of the goals of the project.

    :param models: a list of models to evaluate
    :param model_inputs: a list of inputs corresponding to the models
    :return: the model outputs at the given inputs
    """
    model_outputs = []
    for model, multiple_inputs in zip(models, model_inputs):
        multiple_outputs = [model.evaluate(inpt) for inpt in multiple_inputs]
        model_outputs.append(np.array(multiple_outputs))
    return model_outputs


def check_model_execution(engine_function: EngineFunction,
                          num_models: int, max_cost: float,
                          model_cost_ratio: float, cost_std_ratio: float,
                          target_cost: float, num_processes: int = 1):
    """
    Runs an engine function on a monomial model test case.

    The parameters max_cost, model_cost_ratio, an cost_std_ratio in this
    function allow for the adjustment of model run times which can be useful in
    exploring load balancing.

    :param engine_function: The function which executes the models
    :param num_models: The number of models to use in execution
    :param max_cost: Cost of the highest cost model, models[0]
    :param model_cost_ratio: Multiplicative model costs for each successive
        model
    :param cost_std_ratio: Amount of variation in evaluation time as a fraction
        of evaluation time
    :param target_cost: The approximate total run cost of the models
    :param num_processes: the number of processes used in execution
    """

    models = monomial_example.create_models(num_models, max_cost,
                                            model_cost_ratio,
                                            cost_std_ratio)
    model_inputs = monomial_example.get_model_inputs(models,
                                                     target_cost)

    t_start = time.time()
    model_outputs = engine_function(models, model_inputs)
    t_end = time.time()

    success = monomial_example.is_output_correct(models, model_inputs,
                                                 model_outputs)
    theoretical_time = monomial_example.get_execution_time(models,
                                                           model_inputs)
    theoretical_time /= num_processes
    actual_time = t_end - t_start
    efficiency = theoretical_time / actual_time

    print("Engine Check Results:")
    print("  Output is", "Correct!" if success else "Incorrect!")
    print("  Theoretical execution time was", theoretical_time)
    print("  Actual execution time was", actual_time)
    print("  Efficiency was {:.4f}".format(efficiency))


if __name__ == "__main__":
    NUM_MODELS = 5          # reasonable range [2, 10]
    MAX_COST = 1.0          # time to run longest model (sec)
    MODEL_COST_RATIO = 0.1  # reasonable range [0.001, 0.5]
    COST_STD_RATIO = 0.05   # reasonable range [0.01, 0.2]
    TARGET_COST = 10        # reasonable range [5, 1000]*MAX_COST
    EXECUTION_FUNCTION = simple_serial_engine

    np.random.seed(0)
    check_model_execution(EXECUTION_FUNCTION, NUM_MODELS, MAX_COST,
                          MODEL_COST_RATIO, COST_STD_RATIO, TARGET_COST)
