from firedrake import *

M = Mesh('BFS_ComparisonFile.msh')

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

x, y = SpatialCoordinate(M)

Re = Constant(1.0)

F = (
    1.0 / Re * inner(grad(u), grad(v)) * dx +
    inner(dot(grad(u), u), v) * dx -
    p * div(v) * dx +
    div(u) * q * dx
)

bcs = [DirichletBC(Z.sub(0), as_vector([-(10*y - 1)*(10*y - 5)/4, 0]), (1,)),
       DirichletBC(Z.sub(0), as_vector([0, 0]), (2, 4, 5, 6))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"Re": Re, "velocity_space": 0}

parameters = {
        "mat_type": "matfree",
        "pc_type": "fieldsplit",
        "pc_python_type": "ilu",
        "pc_fieldsplit_type": "schur",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "lu",
        "snes_type": "newtonls",
        "snes_rtol": "1e-10",
        "snes_atol": "1e-10",
        "snes_divtol": "1e5",
        "snes_max_it": 5000,
        "ksp_type": "fgmres",
        "ksp_rtol": "1e-10",
        "ksp_atol": "1e-10",
        "ksp_divtol": "1e5",
        "ksp_max_it": "5000",
        "ksp_gmres_modifiedgramschmidt": True,
        "ksp_monitor_true_residual": None,
        "snes_monitor": None,
        }

up.assign(0)

u, p = up.subfunctions
u.rename("Velocity")
p.rename("Pressure")

ConstRe = 1

File = VTKFile("navier_stokes.pvd")

for ii in range(100):
    print("Re = ", ConstRe)
    solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
      appctx=appctx)
    File.write(u, p, time = ConstRe)

    if (ConstRe == 5100):
        break

    ConstRe = min(ConstRe * 2, 5100)
    Re.assign(ConstRe)


