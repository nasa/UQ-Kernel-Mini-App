
from spring_mass_UQ_model import SpringMassUQModel

mass = 1.5
gravity = 9.8
time_step = 0.1
cost = 1.0

model_UQModel = SpringMassUQModel(mass, gravity, time_step, cost)

stiffnesses = [1.0, 2.5, 5.0]

for stiffness in stiffnesses:
    max_disp = model_UQModel.evaluate([stiffness])
    print("Stiffness = ", stiffness, " max displacement = ", max_disp)


