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
        "ksp_monitor_true_residual": None,
        "ksp_gmres_modifiedgramschmidt": True,
        "snes_monitor": None
        }
NVP1 = NonlinearVariationalProblem(F1, z, bcs=bcu)
NVS1 = NonlinearVariationalSolver(NVP1, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)

params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "newtonls",  # Use Newton's method with line search 
        "ksp_type": "preonly",  # Direct solver for the linear system
        "pc_type": "ilu",  # Use ILU decomposition for preconditioning
        "snes_converged_reason": "",  # Print convergence reason
        "snes_monitor": "",  # Monitor iterations during the solve
        "ksp_rtol": 1.0e-5,
        "snes_rtol": 1.0e-5,  # Set your desired relative tolerance for SNES here
        "snes_atol": 1.0e-5, # set the absolute tolerance for SNES
        "snes_max_it": 10000, #  maximum number of iterations
        "ksp_max_it": 10000,
        "snes_divergence_tolerance": 1e20,
        "snes_stol": 1e-5,
        "ksp_converged_reason":"",
        "ksp_monitor":""
        }
NVP2 = NonlinearVariationalProblem(F2, k, bcs=bck)
NVS2 = NonlinearVariationalSolver(NVP2,solver_parameters=params)
NVP3 = NonlinearVariationalProblem(F3, w, bcs=bcw)
NVS3 = NonlinearVariationalSolver(NVP3,solver_parameters=params)

File = VTKFile("test.pvd")

ConstRe = 1
ConstW = 10
for foo in range(100):
    print("Re = ", ConstRe)
    print("w wall = ", ConstW)
    ConstRe = min(ConstRe * 5, 5100)
    Re.assign(ConstRe)
    for ii in range(10):
        print(ii)
        NVS1.solve()
        for jj in range(15):
            print("i is ", ii,", j is ", jj, ", foo is ", foo)
            NVS2.solve()
            NVS3.solve()
    File.write(u_, p_, k, w)
    ConstW *= 2
    w_wall.assign(ConstW)

print("w wall is ", ConstW)
print("Re is ", ConstRe)
