from firedrake import *

mesh = Mesh('BFS_ComparisonFile.msh')

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
w_wall = Constant(0.1)

# closure coefficients
alpha = Constant(5/9)
Beta = Constant(3/40)
BetaS = Constant(9/100)
SigS = Constant(0.5)
Sig = Constant(0.5)

# fluid constants
de = Constant(1) # density
mu = Constant(1)
Re = 1/mu # want 5100
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
      + (2/3)*de*inner(grad(k), v)*dx
      )

F2 = (de*inner(u, grad(k))*r*dx - inner(RsT(k, w, u), StrT(u))*r*dx 
      + BetaS*de*k*w*r*dx
      + ((1/Re) + SigS*MuT(k, w))*inner(grad(k), grad(r))*dx
      )

F3 = (de*inner(u, grad(w))*s*dx - alpha*w*inner(Tau(k, w, u), StrT(u))*s*dx
        + Beta*de*(w**2)*s*dx + ((1/Re) + Sig*MuT(k, w))*inner(grad(w), grad(s))*dx
      )

F = F1 + F2 + F3
x, y = SpatialCoordinate(mesh)
#[DirichletBC(Z.sub(0), as_vector([-(10*x - 1)*(10*x - 5) / 4, 0]), (1,)),
bcu = [DirichletBC(Z.sub(0), Constant((1, 0)), (1)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (2, 4, 5, 6)),
       DirichletBC(Z.sub(1), Constant(0), (3))]
bck = [DirichletBC(M, Constant(0), (2, 4, 5, 6)),
       DirichletBC(M, Constant(0.015), (1,))] # 0.015 true bc for k
       #DirichletBC(M, Constant(0.01), (1,))] 
bcw = [DirichletBC(N, w_wall, (2, 4, 5, 6)),
       DirichletBC(N, Constant(0.06262), (1,))] # 0.06262 true bc for w
       #DirichletBC(N, Constant(0.01), (1,))]

# plotting tools
u_, p_ = z.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")
w.rename("Specific Dissipation rate")
k.rename("Specific Kinetic Energy")

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
appctx = {"Re": Re, "velocity_space": 0}
"""parameters = {"mat_type": "matfree",
              "snes_monitor": None,
              "ksp_type": "gmres",
              "snes_type": "newtonls",
              "ksp_gmres_modifiedgramschmidt": True,
              "ksp_monitor_true_residual": None,
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "multiplicative",
              "fieldsplit_0_fields": "0,1",
              "fiedldsplit_1_fields": "2,3",
              "fieldsplit_0": {
                  "ksp_type": "gmres",
                  "ksp_gmres_modifiedgramschmidt": True,
                  "snes_rtol": 1e-10,
                  "ksp_rtol": 1e-10,
                  "snes_atol": 1e-10,
                  "ksp_atol": 1e-10,
                  "snes_max_it": 5000,
                  "snes_divtol": 1e5,
                  "ksp_divtol": 1e5,
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "schur",
                  "pc_fieldsplit_schur_fact_type": "lower"
                  },
                  "fieldsplit_0": {
                      "ksp_type": "preonly",
                      "snes_rtol": 1e-10,
                      "snes_atol": 1e-10,
                      "pc_type": "python",
                      "pcd_Mp_ksp_type": "ilu",
                      "pc_python_type": "firedrake.AssembledPC",
                      "assembled_pc_type": "hypre"
                      },
              "fieldsplit_1": {
                  "ksp_type": "preonly",
                  "snes_rtol": 1e-10,
                  "snes_atol": 1e-10,
                  "pc_type": "python",
                  "pc_python_type": "firedrake.PCDPC",
                  "pcd_Mp_ksp_type": "preonly",
                  "pcd_Mp_pc_type": "ilu",
                  "pcd_Kp_ksp_type": "preonly",
                  "pcd_Kp_pc_type": "hypre",
                  "pcd_Fp_mat_type": "aij"
                  }
              }"""

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

"""parameters = {
        "mat_type": "matfree",
        "snes_monitor": None,
        "ksp_type": "fgmres",
        "ksp_gmres_modifiedgramschmidt": None,
        "ksp_monitor_true_residual": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "ilu",
        "fieldsplit_1_ksp_type": "gmres",
        "fieldsplit_1_ksp_rtol": 1e-5,
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
        "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
        "fieldsplit_1_pcd_Mp_pc_type": "ilu",
        "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
        "fieldsplit_1_pcd_Kp_pc_type": "ilu",
        "fieldsplit_1_pcd_Fp_mat_type": "matfree"
        }"""
params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "newtonls",  # Use Newton's method with line search 
        "ksp_type": "preonly",  # Direct solver for the linear system
        "pc_type": "ilu",  # Use ILU decomposition for preconditioning
        #"snes_converged_reason": "",  # Print convergence reason
        #"snes_monitor": "",  # Monitor iterations during the solve
        "ksp_rtol": 1.0e-10,
        "snes_rtol": 1.0e-10,  # Set your desired relative tolerance for SNES here
        "snes_atol": 1.0e-10, # You can also set the absolute tolerance for SNES
        "snes_max_it": 10000,    # And the maximum number of iterations
        # "ksp_converged_reason":"",
        #"ksp_monitor":""
        }

File = VTKFile("NewkOmegaBFSCompare.pvd")

ConstMu = 1
ConstW = 1
Alternate = False

while (ConstMu >= 1/5100 and ConstMu <= 1 and ConstW >= 0.06262 and ConstW <= 1):
    try:
        print("Re = ", 1/ConstMu)
        print("w wall = ", ConstW)
        for ii in range(5):
            print("i is ", ii)
            solve(F1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
            for jj in range(10):
                print("j is ", jj)
                solve(F2 == 0, k, bcs=bck, solver_parameters = params)
                solve(F3 == 0, w, bcs=bcw, solver_parameters = params)
                File.write(u_, p_, k, w)
                w_prev.assign(w)
                k_prev.assign(k)
            z_prev.assign(z)


        if (ConstW == 0.06262 and ConstMu == 1/5100):
            break

        if (ConstW == 0.06262):
            Alternate = True

        if (Alternate == False):
            ConstW = max(0.5*ConstW, 0.06262)
            w_wall.assign(ConstW)
        else:
            ConstMu = max(0.5*ConstMu, 1/5100)
            mu.assign(ConstMu)

    except:
        if (Alternate == False):
            ConstW *= 1.1
            w_wall.assign(ConstW)
        else:
            ConstMu *= 1.1
            mu.assign(ConstMu)
        
        z.assign(z_prev)
        k.assign(k_prev)
        w.assign(w_prev)
"""
for foo in range(100):
    ConstMu = ConstMu * 0.5
    mu.assign(ConstMu)
    for ii in range(10):
        solve(F1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
        for jj in range(15):
            print("i is ", ii,", j is ", jj, ", foo is ", foo)
            print("Re = ", 1/ConstMu)
            solve(F2 == 0, k, bcs=bck, solver_parameters = params)
            solve(F3 == 0, w, bcs=bcw, solver_parameters = params)
    File.write(u_, p_, k, w)
"""
print("w wall is ", ConstW)
print("Mu is ", ConstMu)
