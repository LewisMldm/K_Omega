from firedrake import *

M = Mesh('BFS_ComparisonFile.msh')

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Z = V * W

x, y = SpatialCoordinate(M)

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

Re = Constant(1)

a = ((1/Re)*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx

bcs = [DirichletBC(Z.sub(0), as_vector([-(10*y - 1)*(10*y - 5)/4, 0]), (1,)),
       DirichletBC(Z.sub(0), as_vector([0, 0]), (2, 4, 5, 6))]

u, p = up.subfunctions
u.rename("Mean Velocity")
p.rename("Pressure")

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

up.assign(0)
ConstRe = 1

File = VTKFile("Stokes2.pvd")

for ii in range(100):
    print("Re = ", ConstRe)
    solve(a == 0, up, bcs=bcs, nullspace=nullspace)
    File.write(u, p, time = ConstRe)

    if (ConstRe == 5100):
        break

    ConstRe = min(ConstRe * 2, 5100)
    Re.assign(ConstRe)



