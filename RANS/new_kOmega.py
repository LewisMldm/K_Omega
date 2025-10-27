from firedrake import *

mesh = Mesh('backward-facing-step.msh')

# Taylor hood elements
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
M = FunctionSpace(mesh, "CG", 1)
N = FunctionSpace(mesh, "CG", 1)
Z = V*Q#*M*N

# Functions and test functions
z = Function(Z)
z_prev = Function(Z)
u, p = split(z)
k = Function(M)
k_prev = Function(M)
w = Function(N)
w_prev = Function(N)
v, q = TestFunctions(Z)
r = TestFunction(M)
s = TestFunction(N)

# omega wall constant
#w_wall = Constant(0.06262)
w_wall = Constant(10)

# closure coefficients
alpha = Constant(5/9)
Beta = Constant(3/40)
BetaS = Constant(9/100)
SigS = Constant(0.5)
Sig = Constant(0.5)

# fluid constants
de = Constant(1) # density
Re = Constant(1)
FlInt = 0.05 # Fluid Intensity
TurLS = 0.22 # Turbulence length scale


def StrT(u):
    "Symmetric stress tensor"
    return 0.5*(grad(u) + grad(u).T)

Id = Identity(mesh.geometric_dimension())

def MuT(k, w):
    "Eddy viscosity."
    if norm(u) == 0 or norm(k) == 0:
        import ipdb; ipdb.set_trace()
        return Constant(0)
    else:
        return de*k/w

def Tau(k, w, u):
    """Auxiliary tensor to help with dissipation rate equation"""
    return 2*(de/w)*StrT(u) - (2/3)*de*Id

def RsT(k, w, u):
    """Reynolds Stress Tensor"""
    return k*Tau(k, w, u)

z.assign(0.8)
k.assign(0.8)
w.assign(0.8)

# weak form rans
F1 = (de*inner(dot(grad(u), u), v)*dx - p*div(v)*dx + q*div(u)*dx
      + 2*(1/Re + MuT(k, w))*inner(StrT(u), StrT(v))*dx
      + (2/3)*de*dot(grad(k), v)*dx
      )

F2 = (de*dot(u, grad(k))*r*dx - inner(RsT(k, w, u), StrT(u))*r*dx 
      + BetaS*de*k*w*r*dx
      + ((1/Re) + SigS*MuT(k, w))*dot(grad(k), grad(r))*dx
      )

F3 = (de*dot(u, grad(w))*s*dx - alpha*w*inner(Tau(k, w, u), StrT(u))*s*dx
        + Beta*de*(w**2)*s*dx + ((1/Re) + Sig*MuT(k, w))*dot(grad(w), grad(s))*dx
      )

x, y = SpatialCoordinate(mesh)
bcu = [DirichletBC(Z.sub(0), Constant((1, 0)), 16),
       DirichletBC(Z.sub(0), Constant((0, 0)), 18)]
bck = [DirichletBC(M, Constant(0), 18),
       DirichletBC(M, Constant(0.015), 16)] # 0.015 true bc for k
bcw = [DirichletBC(N, w_wall, 18),
       DirichletBC(N, Constant(46.385), 16)] # 46.385 true bc for w

# plotting tools
u_, p_ = z.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")
w.rename("Specific Dissipation rate")
k.rename("Specific Kinetic Energy")

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
#appctx = {"Re": Re, "velocity_space": 0}
parameters = {
        # should specify options from fieldsplit 1
        "snes_type": "newtonls",
        #"snes_monitor": None,
        #"snes_rtol": "1e-10",
        #"snes_atol": "1e-10",
        #"snes_divtol": "1e5",
        #"snes_max_it": 5000,
        "mat_type": "matfree",
        "ksp_type": "gmres", #"fgmres",
        "snes_monitor": "",  # Monitor iterations during the solve
        #"ksp_monitor_true_residual": None,
        #"ksp_view": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
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
        "fieldsplit_1_Mp_pc_type": "lu"
        #"ksp_rtol": "1e-8",
        #"ksp_atol": "1e-8",
        #"ksp_divtol": "1e5",
        #"ksp_max_it": "5000",
        #"ksp_gmres_modifiedgramschmidt": True,
        }
NVP1 = NonlinearVariationalProblem(F1, z, bcs=bcu)
NVS1 = NonlinearVariationalSolver(NVP1, nullspace=nullspace, solver_parameters=parameters)
#NVS1 = NonlinearVariationalSolver(NVP1, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
# snes composite solver is possible
params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "newtonls",  # Use Newton's method with line search   switch to basic line search
        #"ksp_view": None,
        "ksp_type": "gmres", #"preonly",  # Direct solver for the linear system,, maybe switch gmres so it checks
        "pc_type": "lu",  # Use ILU decomposition for preconditioning  # can do snes_lag_preconditioner to specify how often to reform the matrix decomposition
        "pc_factor_mat_solver_type" : "mumps",
        #"snes_converged_reason": "",  # Print convergence reason
        "snes_monitor": "",  # Monitor iterations during the solve
        #"ksp_rtol": 1.0e-5,
        #"snes_rtol": 1.0e-5,  # Set your desired relative tolerance for SNES here
        #"snes_atol": 1.0e-5, # set the absolute tolerance for SNES
        #"snes_max_it": 50, #  maximum number of iterations
        #"ksp_max_it": 10,  # should be 2 or 3
        #"snes_divergence_tolerance": 1e20,
        #"snes_stol": 1e-5,
        #"ksp_converged_reason":"",
        #"ksp_monitor":""
        }
NVP2 = NonlinearVariationalProblem(F2, k, bcs=bck)
NVS2 = NonlinearVariationalSolver(NVP2,solver_parameters=params)# add prefix argument to pass various solver parameters from script
NVP3 = NonlinearVariationalProblem(F3, w, bcs=bcw)
NVS3 = NonlinearVariationalSolver(NVP3,solver_parameters=params)

File = VTKFile("test.pvd")

ConstRe = 25
ConstW = 100 # 10
w_wall.assign(ConstW)
Re.assign(ConstRe)

effective_diffusion = Function(M, name="effective_diffusion")
File_d = VTKFile("diffusion.pvd")
for foo in range(100):
    print("Re = ", ConstRe)
    print("w wall = ", ConstW)
    for ii in range(10):
        effective_diffusion.interpolate(2*(1/Re + MuT(k, w)))
        File_d.write(effective_diffusion, k, w)
        print("Solve Navier_Stokes")
        NVS1.solve()
        print("Solve NVS2")
        NVS2.solve()
        print("Solve NVS3")
        NVS3.solve()
        #for jj in range(10):
        #    print("i is ", ii,", j is ", jj, ", foo is ", foo)
    File.write(u_, p_, k, w)
    ConstRe = min(ConstRe * 5, 5100)
    Re.assign(ConstRe)
    #ConstW *= 2
    #w_wall.assign(ConstW)

print("w wall is ", ConstW)
print("Re is ", ConstRe)
