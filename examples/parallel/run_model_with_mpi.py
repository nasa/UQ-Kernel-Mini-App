'''
Simple demo script for running a UQModel object in parallel using mpi4py. 
'''
from mpi4py import MPI
import numpy as np
import time

from uq_kernel.model import UQModel

# Set up MPI communicator for running in parallel
comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()   #this processors number/identifier (int) 
size = comm.Get_size()   #total number of processors

#--------------- Simple UQModel that squares the input----------------
class SquareItModel(UQModel):
    
    def __init__(self, cost=0.001):
        self._cost = cost

    @property
    def cost(self):
        """
        The evaluation time - the evaluate() func will sleep for this amount 
        of time (in seconds) each time it is called
        """
        return self._cost

    def evaluate(self, inputs):
        time.sleep(self._cost)
        return inputs[0]**2
#----------------------------------------------------------------------

num_inputs = 1000
model_cost = 1e-3

if rank==0:
    theoretical_time = num_inputs*model_cost / size
    print("Theoretical run time:\t ", theoretical_time)

model = SquareItModel(model_cost)
np.random.seed(1)
all_proc_inputs = np.random.randn(num_inputs)  #random array of inputs
start_time = time.time()

#Split up inputs among processors (could do this smarter with "scatter"):
num_inputs_per_proc = num_inputs//size
start_index = rank*num_inputs_per_proc
if rank == size-1:
    end_index = num_inputs   #last processor gets remainder of inputs if uneven
else:
    end_index = (rank+1)*num_inputs_per_proc
    
this_proc_inputs = all_proc_inputs[start_index:end_index]
this_proc_outputs = np.zeros(len(this_proc_inputs))

#Evaluate model in parallel for each proc's inputs:
for i, input_ in enumerate(this_proc_inputs):
    this_proc_outputs[i] = model.evaluate([input_])

#Collect outputs from all procs into one output array on processor #0
all_proc_outputs = comm.gather(this_proc_outputs, root=0)

if rank==0:
    parallel_time = time.time() - start_time
    print("Actual run time:\t ", parallel_time)
    #Stacks all proc outputs into one 1D numpy array
    all_proc_outputs = np.concatenate(all_proc_outputs, axis=0)

#Generate the same result on one processor for comparison:
if rank==0:
    start_time = time.time()
    outputs_serial = np.zeros(num_inputs)
    for i, input_ in enumerate(all_proc_inputs):
        outputs_serial[i] = model.evaluate([input_])
    serial_time = time.time() - start_time
    output_diff = np.linalg.norm(all_proc_outputs - outputs_serial)

    print("Serial run time:\t ", serial_time)
    print("Serial vs parallel output difference: ", output_diff)



