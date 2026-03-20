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
Produce = Constant(100)

R = FunctionSpace(M, 'R', 0)
JetIn = Function(R).assign(500)

a = ((1/Re)*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx

pol = (Diff_coef*inner(grad(t),grad(l)) + dot(u, grad(t))*l)*dx + exp(-10 * ((x - 8.5)**2 + (y - 0)**2))*l*dx 

bcs = [DirichletBC(Z.sub(0), as_vector([0, JetIn*((x-7)*(x-5))]), (19,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (21,22))]

bcp = [DirichletBC(T, Produce, (22,))]
#       DirichletBC(T, Constant(0), (19))]

solve(a==0, up, bcs=bcs)
bounds=[0.0, 10000.0]

solve (pol==0, t)

#Z1 = V * W * T
#up1 = Function(Z1)
#u1, p1, t1 = split(up1)
#v1, q1, l1 = TestFunctions(Z1)

J = assemble(conditional(le(x,7), t**2, Constant(0)) * dx )

u_, p_ = up.subfunctions

u_.rename("Mean Velocity")
p_.rename("Pressure")

File = VTKFile("Stokes_opt/test2.pvd")
File.write(u_, p_, t, time=1)

Jhat = ReducedFunctional(J, [Control(JetIn)])


stop_annotating()
get_working_tape().progress_bar = ProgressBar

optval = minimize(Jhat, options={"maxiter":10}, bounds=bounds)

JetIn.assign(optval)

print(f"Inlet vel = {float(JetIn):1.2f}")
print("final solve opt")
solve(a==0, up, bcs=bcs)
solve(pol==0, t)
File.write(u_, p_, t, time=2)
