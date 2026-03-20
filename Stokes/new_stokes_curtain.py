from firedrake import *
from firedrake.adjoint import *
import numpy as np

continue_annotation()

M = Mesh('DomainThin.msh')

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
T = FunctionSpace(M, "CG", 1)
Z = V * W

x, y = SpatialCoordinate(M)

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

t = Function(T, name="polutant concentration")
l = TestFunction(T)

Re = Constant(1e5)
Diff_coef = Constant(1)
#Produce = Constant(100)

R = FunctionSpace(M, 'R', 0)
JetIn = Function(R).interpolate(1)

a = ((1/Re)*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx
#bcs = [DirichletBC(Z.sub(0), as_vector([0, JetIn*((x-7)*(x-5))]), (19,)),
Ubdry = Function(V).interpolate(as_vector([0, JetIn*((x-7)*(x-5))]))
bcs = [DirichletBC(Z.sub(0), Ubdry, (19,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (21,22))]
solve(a==0, up, bcs=bcs)

pol = (Diff_coef*inner(grad(t),grad(l)) + dot(u, grad(t))*l)*dx - 100*exp(-10 * ((x - 8.5)**2 + (y - 0)**2))*l*dx
bcp = DirichletBC(T, Constant(0), (19))
solve(pol==0, t, bcs=bcp)

#myfile.write(u_, p_, t)

#bcp = [DirichletBC(T, Produce, (22,))]
#       DirichletBC(T, Constant(0), (19))]
J = assemble(conditional(le(x,7), t**2, Constant(0)) * dx) + 0.1*assemble(JetIn**2*ds(19))
Jhat = ReducedFunctional(J, [Control(JetIn)])
stop_annotating()

"""for JetIn_ in [1, 2, 10, 20]:
    JetIn.interpolate(JetIn_)
    print(f"Inlet vel = {float(JetIn):1.2f}, J = {float(Jhat(JetIn)):1.2e}")
"""

u_, p_ = up.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")
File = VTKFile("Stokes_opt/test2.pvd")
File.write(u_, p_, t)

print(f"Inlet vel = {float(JetIn):1.2f}, J = {float(Jhat(JetIn)):1.2e}")

get_working_tape().progress_bar = ProgressBar
#bounds=[0.0, 100.0]
#optval = minimize(Jhat, options={"maxiter":10}, bounds=bounds)
optval = minimize(Jhat)

# plot final pollution concetration
JetIn.interpolate(optval)
print(f"Inlet vel = {float(JetIn):1.2f}, J = {float(Jhat(JetIn)):1.2e}")
Ubdry.interpolate(as_vector([0, JetIn*((x-7)*(x-5))]))
solve(a==0, up, bcs=bcs)
solve(pol==0, t, bcs=bcp)
File.write(u_, p_, t)
