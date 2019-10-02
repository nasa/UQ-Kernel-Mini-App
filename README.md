# uq kernel miniapp

A kernel for uncertainty quantification (UQ) codes at NASA.

# Getting Started

1) Clone or download this repository
* It is best to work with and modify the source code using Git ([install/update Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)). Using Git, clone the repository to your computer using `git clone https://github.com/nasa/UQ-Kernel-Mini-App.git`. Otherwise, the repository can be downloaded manually with the "Clone or download" button above. 
2) Add the repository to your Python path
* On Mac/Linux, add `export PYTHONPATH=$PYTHONPATH:Path/To/Your/Repo/` to your ~/.bashrc file (click [here](https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath) for more details). For Windows, see this [link](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages). This makes it possible to run the code in this repository from anywhere on your computer.
3) Make sure you have the required Python modules
* The repository uses a few external Python modules (currently just `numpy` and `scipy`). Either install them manually, or using pip: `pip install -r requirements.txt` or Anaconda: `conda install --yes --file requirements.txt` from the top directory of the repository.
4) Test that everything is working correctly
* Navigate to the `tests/` directory in the repository and type `python engine_check.py`. If things are working as expected, you should see an output similar to:
```
Engine Check Results:
  Output is Correct!
  Theoretical execution time was 9.40658189601207
  Actual execution time was 9.961344003677368
  Efficiency was 0.9443
```

# VT Capstone project

## Challenge
Devise  an  interface  (front end)  and  computational  engine  (back end)  
that  increases  the  usability, efficiency, and scalability of NASA 
open-source uncertainty quantification software.

### Sub-challenge 1: interface
 How can we best allow users from all disciplines to “plug” their own 
 computational model into our Python-based UQ software?

#### Interface Requirements

Currently, NASA UQ software requires that a user develops a Python class
for the model they'd like to analyze that has a standardized interface. 
The primary requirement of this interface is that their Python class has
an `evaluate` function that receives an array of input parameters for their
model and returns an array of outputs:

 ```python
def evaluate(self, inputs):
	<Python code to implement an internal simulation or execute an external simulation>
	return output
```

Secondarily, all Python models must have a `cost` member variable that contains
the approximate time to execute the model. The base interface for a `UQModel` 
is defined in `uq_kernel/model.py`.  It illustrates these parts of a model that 
must be exposed to the UQ framework. 

#### Challenge problem focus

The challenge problem should focus on the most general type of `UQModel` 
and how best to streamline the process of creating one to use with 
NASA UQ software. This is the situation where a user has a simulation
executable (developed in any programming language) that can be run
on the command line by providing an input file and, upon completion, writes 
an output file with any simulation output data of interest. 
 
In this case, the `UQModel` a user develops to use the NASA UQ
software will always follow the general form:
```python
def evaluate(self, inputs):
	self.write_input_file_to_disc(inputs)
	self.execute_model()
	output = self.parse_output_file()
	return output
```
where the `write_input_file_to_disc` function generates a problem-specific 
input file based on the `inputs` array and stores it, `execute_model` will
make a system call in Python to execute the user's simulation for the 
input file written previously and write an output file when it has completed,
and `parse_output_file` will load the output file and extract the relevant 
outputs to return. 

In general, a user will know the command to execute their simulation at the
command line and the format of their input file along with where the particular 
parametersin the `inputs` array should go within it. From here, they need to 
wrap this functionality inside of a `UQModel` Python class. We want to make 
this process as easy as possible by capitalizing on any code, procedures, patterns
that are shared between most or all applications and simulations.
 
#### UQModel Example 

There is a simple example of a simulation and corresponding Python `UQModel`
in the `examples/spring_mass_uq_model/` to make this challenge more concrete.

<img src="/imgs/spring_mass.png" width="200">

The `spring_mass_simulation.py` implements a simple spring-mass system and 
calculates the maximum displacement in the system for given parameters like
mass and acceleration due to gravity. For our purposes, we will assume this
is a "blackbox" executable, we don't have access to the code inside,
we just know we can run it on the command line using:

`python spring_mass_simulation.py <name-of-input-file> <name-of-output-file>`

where the input file contains values for mass, stiffness, etc. and the output
file will contain a single number defining the maximum displacement. An example
input file is provided (`spring_mass_inputs.txt`) so that the code can be 
tested by typing `python spring_mass_simulation.py spring_mass_inputs.txt output.txt`.

Now let's say that the user wants to use NASA UQ software to study how 
uncertainty in the spring stiffness effects their prediction of the 
maximum displacement. They need to define a `UQModel` class with and
`evaluate` function whose `inputs` array contains one value for stiffness 
and the `output` array contains one value for the maximum displacement.
An example of such a class can be seen in the `spring_mass_UQ_model.py`.
A sample script that initializes and evaluates the model for a few
different stiffness values is provided in `run_spring_mass_uq_model.py`
for illustration.

How can we help a user produce code/classes like inside of 
`spring_mass_UQ_model.py` to "plug" their simulation into the NASA
UQ software? The user will have information about how to execute their
simulation from the command line, the structure of their input file, and the
names of pertinant inputs/outputs. Note that it is important to distinguish 
between the parameters that will change from simulation to simulation (stiffness 
in this example) versus those that will remain fixed (gravity, mass, etc.).
The main procedure of 1) writing input file, 2) executing model, and 3) parsing 
output file / returning output(s) of interest will generally be followed, but
these individual steps will require customization depending on the user's 
simulation, input/output files, parameters of interest, etc. 

 
### Implementation
A starting interface for a `UQModel` is defined in `uq_kernel/model.py`.  It 
illustrates the parts of a model that must be exposed to the UQ framework. 
Defining implementations of this interface to account for different possible 
models is one option to pursue this challenge.  Alternative solutions are 
encouraged as well.  Everything, including the basic interface in 
`uq_kernel/model.py` can be modified to suit the chosen approach.

### Sub-challenge 2: engine
**A load balancing issue for parallel computing:** How can we devise a strategy 
to execute a set of computational models with *varying run times* an arbitrary 
number of times each in an efficient and scalable manner.

#### Problem Definition
**Given**
 * A set of Python models (like the ones developed in sub-challenge 1)
 * The estimated cost (run time) for each model
 * The number of times to execute each model (with the inputs to the models 
 for each evaluation)
 * Number of processors to run on

**Determine**

A strategy for spreading the executions of all models across the processors 
such that all processors have a similar amount of work to do (minimize idle 
time for processors)

#### Implementation

Develop an engine function with an interface that accepts a list of `UQmodel` 
objects and a list of `numpy` arrays defining the inputs for each model,

<img src="/imgs/inputs.png" width="600">

note that in general the individual numpy arrays will have different lengths
(number of inputs). This function returns a list of numpy arrays defining
the outputs

<img src="/imgs/outputs.png" width="600">

where each array has been generated by evaluating the appropriate `UQmodel`,
for example: 

<img src="/imgs/run_model.png" width="600">

assuming that `model-1` has `N1` inputs to evaluate. 

If the approximate cost of each model is `C1`, ..., `CM`, then the total time 
to run all of the models on one processor is `T-serial = N1 * C1 + N2 * C2 + ... NM * CM`. 
The specific goal is to write an engine function that has an execution time that 
is as close to `T-parallel = T1 / P` as possible when run on `P` processors. An example of 
a serial engine function can be seen in the `tests/engine_check.py` in the 
`simple_serial_engine` function. 

The implementation of the engine is completely open; however, a solution that 
works in a distributed memory system is preferred. `mpi4py` is one example of a 
distributed processing tool that could be helpful, though others exist as well.

A script for checking implementations of the engine is included in the miniapp: 
`tests/engine_check.py`. The script runs the engine, checks its outputs, and 
computes its efficiency.  The script is configurable to test many different 
possibilities of model number, run times, etc.  Note that it is likely that the 
script will need to be modified to work in the context of the engine you 
develop.  It should prove useful, nonetheless.

#### Running a model in parallel with mpi4py

An example script demonstrating basic usage of `mpi4py` to evaluate a Python model
in parallel is given in `examples/parallel/run_model_with_mpi.py`. In order to run
this script, `mpi4py` must be installed, which requires a working installation of MPI on your computer (look for [online resources](https://mpi4py.readthedocs.io/en/stable/install.html) if there are issues). Once installed properly, the script can be run with `mpirun -np <number of processors> python run_model_with_mpi.py` or `mpiexec -n <number of processors> python run_model_with_mpi.py` depending on your version of MPI. 
