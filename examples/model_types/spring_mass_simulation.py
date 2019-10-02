import numpy as np
import sys
from scipy.integrate import odeint

class SpringMassSimulation():
    """
    Defines Spring Mass model with 1 free param (stiffness of spring, k). The
    quantity of interest that is returned by the evaluate() function is the
    maximum displacement over the specified time interval.

    The implementation for this model is done completely in the Python code
    contained in this class
    """

    def __init__(self, mass=1.5, gravity=9.8, time_step=None, cost=None,
                 state0=None):

        self._mass = mass
        self._gravity = gravity

        # Give default initial conditions & time grid if not specified
        if state0 is None:
            state0 = [0.0, 0.0]
        if time_step is None:
            time_grid = np.arange(0.0, 10.0, 0.1)
        else:
            time_grid = np.arange(0.0, 10.0, time_step)

        self._state0 = state0
        self._t = time_grid
        self._cost = cost

    @property
    def cost(self):
        return cost

    def evaluate(self, inputs):
        """
        Returns the max displacement over the course of the simulation.
        MLMCPy convention is that evaluated takes in an array and returns an
        array (even for 1D examples like this one).
        """
        stiffness = inputs[0]
        state = self.simulate(stiffness)
        return np.array([max(state[:, 0])])

    def simulate(self, stiffness):
        """
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid
        """
        return odeint(self._integration_func, self._state0, self._t,
                      args=(stiffness, self._mass, self._gravity))

    @staticmethod 
    def _integration_func(state, t, k, m, g):
        """
        Return velocity/acceleration given velocity/position and values for
        stiffness and mass. Helper function for numerical integrator
        """

        # unpack the state vector
        x = state[0]
        xd = state[1]

        # compute acceleration xdd
        xdd = ((-k * x) / m) + g

        # return the two state derivatives
        return [xd, xdd]


def read_inputs_from_file(filename):
    '''
    Assumes input file lists inputs as:
    <inputname1>=<inputvalue1>
    <inputname2>=<inputvalue2>
    ...
    ignores any lines that start with "#" (comments)

    '''

    with open(filename) as fid:
        lines = [line.strip('\n') for line in fid.readlines() if line[0]!="#"]
        input_dict = {}
        for line in lines:
            key, val = line.split("=")
            input_dict[key.strip()] = float(val)
    return input_dict


if __name__ == "__main__":
    
    #Read commmand line arguments
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    #Parse parameters from input file (must provide "mass", "gravity", 
    #"stiffness", "time_step", and "cost"
    inputs = read_inputs_from_file(inputfile)

    gravity = inputs["gravity"]
    mass = inputs["mass"]
    time_step = inputs["time_step"]
    cost = inputs["cost"]
    stiffness = inputs["stiffness"]

    #Initialize / evaluate model
    model = SpringMassSimulation(mass=mass, gravity=gravity,
                                  time_step=time_step)

    max_disp = model.evaluate([stiffness])
 
    #Write max. displacement to output file
    np.savetxt(outputfile, max_disp)

