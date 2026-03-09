from firedrake import *
from firedrake.adjoint import *
import numpy as np

continue_annotation()

M = Mesh('DomainThin.msh')

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Z = V * W
T = FunctionSpace(M, "CG", 1)

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)
t = Function(T)
l = TestFunction(T)

x, y = SpatialCoordinate(M)

Re = Constant(1.0)
Diff_coef = Constant(100)

R = FunctionSpace(M, 'R', 0)
JetIn = Function(R).assign(500)

F = (
    1.0 / Re * inner(grad(u), grad(v)) * dx +
    inner(dot(grad(u), u), v) * dx -
    p * div(v) * dx +
    div(u) * q * dx
)

Pol = (Diff_coef*inner(grad(t),grad(l))*dx + dot(u, grad(t))*l*dx
      )

bcs = [DirichletBC(Z.sub(0), as_vector([0, JetIn*(4*(x-4)*(x-5))]), (19,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (21, 22))]

bcp = [DirichletBC(T, Constant(100), (22))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"Re": Re, "velocity_space": 0}

parameters = {
        # should specify options from fieldsplit 1
        "snes_type": "newtonls",
        #"snes_monitor": None,
        "mat_type": "matfree",
        "ksp_type": "fgmres", #"fgmres",
        "snes_monitor": "",  # Monitor iterations during the solve
        "ksp_monitor_true_residual": None,
        #"ksp_view": None,
        "pc_type": "fieldsplit",
        #"pc_fieldsplit_type": "multiplicative",
        "pc_factor_mat_mumps_icntl_14": 200,
        "pc_fieldsplit_type": "schur",  # additive or multiplicative is more robust, look also into the pure Stokes example
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_off_diag_use_amat": True,
        #"fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_Mp_mat_type": "aij",
        "fieldsplit_0_assembled_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        #"fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
        "fieldsplit_1_Mp_mat_type": "aij",
        "fieldsplit_1_Mp_pc_type": "lu",
        "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
        "ksp_rtol": "1.0e-5",
        "ksp_atol": "1.0e-5",
        "snes_rtol": "1.0e-10",
        #"ksp_divtol": "1e5",
        #"ksp_max_it": "5000",
        "ksp_gmres_modifiedgramschmidt": True,
        }

bounds = [(0.0, 1000.0)]
J = assemble(conditional(le(x, 4), exp(10*t), Constant(0)) * dx) + 0.1*assemble(JetIn * ds(19))
Jhat = ReducedFunctional(J, [Control(JetIn)])

u_, p_ = up.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
t.rename("Polutant Concentration")

up.assign(0.5)
ConstRe = 1
Re = Constant(ConstRe)

File = VTKFile("navier_stokes_opt.pvd")

for ii in range(100):
    print("Re = ", ConstRe)
    solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
    print("Begin solve con diff")
    solve(Pol == 0, t, bcs=bcp)

    if ConstRe == 10000:
        stop_annotating()
        get_working_tape().progress_bar = ProgressBar
        optval = minimize(Jhat, bounds=bounds)
        JetIn.assign(optval[0])
        print(f"Inlet vel = {float(JetIn):1.2f}")

        print("final solve opt stokes")
        solve(F == 0, up, bcs=bcs, solver_parameters = parameters, nullspace=nullspace, appctx=appctx)
        print("final solve opt con diff")
        solve(Pol == 0, t, bcs=bcp)

        File.write(u, p, t)
        break

    ConstRe = min(ConstRe * 2, 10000)
    Re = Constant(ConstRe)


