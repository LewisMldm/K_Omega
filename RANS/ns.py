from firedrake import *

mesh = Mesh('backward-facing-step.msh')

# Taylor hood elements
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
Z = V*Q

# Functions and test functions
z = Function(Z)
u, p = split(z)
v, q = TestFunctions(Z)

# fluid constants
Re = Constant(1)

def StrT(u):
    "Symmetric stress tensor"
    return 0.5*(grad(u) + grad(u).T)

z.assign(0.8)

# weak form rans
F1 = (inner(dot(grad(u), u), v)*dx - p*div(v)*dx + q*div(u)*dx
      + 2*(1/Re)*inner(StrT(u), StrT(v))*dx
      )

x, y = SpatialCoordinate(mesh)
bcu = [DirichletBC(Z.sub(0), Constant((1, 0)), 16),
       DirichletBC(Z.sub(0), Constant((0, 0)), 18)]

# plotting tools
u_, p_ = z.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
parameters = {
        # should specify options from fieldsplit 1
        "snes_type": "newtonls",
        "mat_type": "matfree",
        "ksp_type": "gmres", #"fgmres",
        "snes_monitor": "",  # Monitor iterations during the solve
        #"ksp_monitor": "",
        #"snes_view": None,
        #"ksp_monitor_true_residual": None,
        #"ksp_view": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
        #"pc_fieldsplit_type": "schur",  # additive or multiplicative is more robust, look also into the pure Stokes example
        #"pc_fieldsplit_schur_fact_type": "diag",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "lu",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
        "fieldsplit_1_Mp_mat_type": "aij",
        "fieldsplit_1_Mp_pc_type": "lu",
        #"ksp_gmres_modifiedgramschmidt": True,
        }
NVP1 = NonlinearVariationalProblem(F1, z, bcs=bcu)
NVS1 = NonlinearVariationalSolver(NVP1, nullspace=nullspace, solver_parameters=parameters)

File = VTKFile("ns.pvd")

for ConstRe in range(1, 100, 1):
    Re.assign(ConstRe)
    print("Solve Navier_Stokes for ConstRe = ", ConstRe)
    NVS1.solve()
    File.write(u_, p_)
    #ConstRe = min(ConstRe * 5, 5100)
