from firedrake import *

mesh_ = Mesh('BFS_ComparisonFile.msh')
mh = MeshHierarchy(mesh_, 1, refinements_per_level=3)
mesh = mh[-1]

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

F = F1 + F2 + F3
x, y = SpatialCoordinate(mesh)
#[DirichletBC(Z.sub(0), as_vector([-(10*x - 1)*(10*x - 5) / 4, 0]), (1,)),
bcu = [DirichletBC(Z.sub(0), Constant((1, 0)), (1)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (2, 4, 5, 6))]
bck = [DirichletBC(M, Constant(0), (2, 4, 5, 6)),
       DirichletBC(M, Constant(0.015), (1,))] # 0.015 true bc for k
       #DirichletBC(M, Constant(0.01), (1,))] 
bcw = [DirichletBC(N, w_wall, (2, 4, 5, 6)),
       DirichletBC(N, Constant(46.385), (1,))] # 46.385 true bc for w

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

File = VTKFile("NewkOmegaBFSComp2.pvd")

ConstRe = 1
ConstW = 10
Alternate = True
"""
while (ConstMu >= 1/5100 and ConstMu <= 1 and ConstW >= 0.06262 and ConstW <= 1):
    try:
        print("Re = ", 1/ConstMu)
        print("w wall = ", ConstW)
        for ii in range(10):
            print("i is ", ii)
            solve(F1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
            for jj in range(10):
                print("j is ", jj)
                solve(F2 == 0, k, bcs=bck, solver_parameters = params)
                solve(F3 == 0, w, bcs=bcw, solver_parameters = params)
                w_prev.assign(w)
                k_prev.assign(k)
            z_prev.assign(z)

        File.write(u_, p_, k, w, time=1/ConstMu)


        if (ConstW == 0.06262 and ConstMu == 1/5100):
            break

        if (ConstMu == 1/5100):
            Alternate = False

        if (Alternate == False):
            ConstW = min(2*ConstW, 10**10)
            w_wall.assign(ConstW)
        else:
            ConstMu = max(0.5*ConstMu, 1/5100)
            mu.assign(ConstMu)

    except:
        if (Alternate == False):
            ConstW *= 0.9
            w_wall.assign(ConstW)
        else:
            ConstMu *= 1.1
            mu.assign(ConstMu)
        
        z.assign(z_prev)
        k.assign(k_prev)
        w.assign(w_prev)
"""
for foo in range(100):
    print("Re = ", ConstRe)
    print("w wall = ", ConstW)
    ConstRe = min(ConstRe * 5, 5100)
    Re.assign(ConstRe)
    for ii in range(10):
        solve(F1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
        for jj in range(15):
            print("i is ", ii,", j is ", jj, ", foo is ", foo)
            solve(F2 == 0, k, bcs=bck, solver_parameters = params)
            solve(F3 == 0, w, bcs=bcw, solver_parameters = params)
    #File.write(u_, p_, k, w)
    ConstW *= 2
    w_wall.assign(ConstW)

print("w wall is ", ConstW)
print("Re is ", ConstRe)
