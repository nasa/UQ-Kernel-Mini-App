# uq kernel miniapp

A kernel for uncertainty quantification (UQ) codes at NASA.

# VT Capstone project

## Challenge
Devise  an  interface  (front end)  and  computational  engine  (back end)  
that  increases  the  usability,efficiency, and scalability of NASA open-source 
uncertainty quantification software.

### Sub-challenge 1: interface
 How can we best allow users from all disciplines to “plug” their own 
 computational model into our Python-based UQ software?

#### Examples of possible types of models
 * Model implemented entirely/explicitly with Python code
 ```python
def evaluate(self, inputs):
	<a bunch of python code to calculate output>
	return output
```
 * Model is a Python wrapper that calls a simulation executable
```python
def evaluate(self, inputs):
	self.write_input_file_to_disc(inputs)
	self.execute_model()
	output = self.parse_output_file()
	return output
```
 * Model created from input/output dataset – initialized from dataset;  
 evaluate() function finds and returns output for given input
 * Model loads a previously-trained machine learning model from file (a Python 
 pickle file) and uses it to make a prediction in evaluate()
 
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

The implementation of the engine is completely open; however, a solution that 
works in a distributed memory system is preferred. MPI4py is one example of a 
distributed processing tool that could be helpful, though others exist as well.

A script for checking implementations of the engine is included in the miniapp: 
`tests/engine_check.py`. The script runs the engine, checks its outputs, and 
computes its efficiency.  The script is configurable to test many different 
possibilities of model number, run times, etc.  Note that it is likely that the 
script will need to be modified to work in the context of the engine you 
develop.  It should prove useful, nonetheless.