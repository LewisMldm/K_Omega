from firedrake import *

mesh = Mesh('backward-facing-step1.msh')

# Taylor hood elements
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "DG", 1)
M = FunctionSpace(mesh, "DG", 1)
N = FunctionSpace(mesh, "DG", 1)
Z = V*Q#*M*N

# Functions and test functions
z = Function(Z)
prev_z = Function(Z)
prev_z_try = Function(Z)
u, p = split(z)
k = Function(M)
prev_k = Function(M)
prev_k_try = Function(M)
w = Function(N)
prev_w = Function(N)
prev_w_try = Function(N)
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
bcu = [DirichletBC(Z.sub(0), as_vector([-y*(10*y - 5)*(8/5), 0]), 16),
       DirichletBC(Z.sub(0), Constant((0, 0)), 18)]
bck = [DirichletBC(M, Constant(0), 18),
       DirichletBC(M, Constant(0.015)*(-y*(10*y - 5)*(8/5)), 16)] # 0.015 true bc for k
bcw = [DirichletBC(N, w_wall, 18),
       DirichletBC(N, 46.385, 16)] # 46.385 true bc for w

# plotting tools
u_, p_ = z.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")
w.rename("Specific Dissipation rate")
k.rename("Specific Kinetic Energy")

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
appctx = {"Re": Re, "velocity_space": 0}
appctx2 = {"Re": Re}

parameters = {
        # should specify options from fieldsplit 1
        "snes_type": "newtonls",
        #"snes_monitor": None,
        "mat_type": "matfree",
        "ksp_type": "fgmres", #"fgmres",
        #"snes_monitor": "",  # Monitor iterations during the solve
        #"ksp_monitor_true_residual": None,
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
"""
parameters = {
    'mat_type': 'nest',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-5,
    'snes_rtol': 1e-5,
    'snes_stol': 1e-5,
    'ksp_type': 'fgmres',
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 1000,
    'ksp_atol': 1e-5,
    'ksp_rtol': 1e-5,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'fieldsplit_0': {'ksp_convergence_test': 'skip',
                     'ksp_max_it': 1,
                     'ksp_norm_type': 'unpreconditioned',
                     'ksp_richardson_self_scale': False,
                     'ksp_type': 'richardson',
                     'pc_type': 'mg',
                     'pc_mg_type': 'full',
                     'mg_coarse_assembled_pc_type': 'lu',
                     'mg_coarse_assembled_pc_factor_mat_solver_type': 'superlu_dist',
                     'mg_coarse_assembled_mat_mumps_icntl_14': 1000,
                     'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
                     'mg_coarse_pc_type': 'python',
                     'mg_levels': {'ksp_convergence_test': 'skip',
                                   'ksp_max_it': 5,
                                   'ksp_type': 'fgmres',
                                   'pc_python_type': 'firedrake.ASMStarPC',
                                   'pc_type': 'python'},
                    },
    'fieldsplit_1': {'ksp_type': 'richardson',
                     'ksp_max_it': 1,
                     'ksp_convergence_test': 'skip',
                     'ksp_richardson_scale': 1,
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.MassInvPC',
                     'Mp_pc_type': 'jacobi'},
}
"""
# pc lu, mat solv type mumps, no fieldsplit
NVP1 = NonlinearVariationalProblem(F1, z, bcs=bcu)
#NVS1 = NonlinearVariationalSolver(NVP1, nullspace=nullspace, solver_parameters=parameters)
NVS1 = NonlinearVariationalSolver(NVP1, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
# snes composite solver is possible
params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "newtonls",  # Use Newton's method with line search   switch to basic line search
        #"ksp_view": None,
        "ksp_type": "gmres", #"preonly",  # Direct solver for the linear system,, maybe switch gmres so it checks
        "pc_type": "lu",  # Use ILU decomposition for preconditioning  # can do snes_lag_preconditioner to specify how often to reform the matrix decomposition
        #"pc_factor_mat_solver_type" : "mumps",
	#"pc_factor_mat_mumps_icntl_14": 400,
        #"snes_converged_reason": "",  # Print convergence reason
        #"snes_monitor": "",  # Monitor iterations during the solve
        "ksp_rtol": 1.0e-5,
	"ksp_atol": 1.0e-5,
	"snes_rtol": 1.0e-5,  # Set your desired relative tolerance for SNES here
        #"snes_atol": 1.0e-5, # set the absolute tolerance for SNES
        #"snes_max_it": 50, #  maximum number of iterations
        #"ksp_max_it": 10,  # should be 2 or 3
        #"snes_divergence_tolerance": 1e20,
        #"snes_stol": 1e-5,
        #"ksp_converged_reason":"",
        #"ksp_monitor":""
        }
NVP2 = NonlinearVariationalProblem(F2, k, bcs=bck)
NVS2 = NonlinearVariationalSolver(NVP2,solver_parameters=params, appctx=appctx2) #add prefix argument to pass various solver parameters from script
NVP3 = NonlinearVariationalProblem(F3, w, bcs=bcw)
NVS3 = NonlinearVariationalSolver(NVP3,solver_parameters=params, appctx=appctx2)

File = VTKFile("kOmBFS_DG.pvd")

ConstRe = 1000
ConstW = 1000 # 10
w_wall.assign(ConstW)
Re.assign(ConstRe)

prev_z.assign(z)
prev_z_try.assign(z)
prev_k.assign(k)

prev_w.assign(w)

relax_z = 1
relax_k = 0.5
relax_w = 0.5


u_solution, p_solution = z.subfunctions

effective_diffusion = Function(M, name="effective_diffusion")
File_d = VTKFile("diffusion.pvd")
Re_increment = 1000
w_increment = 1000

switch = False

"""
while(w_increment > 1e-10 and ConstRe >= 500):
    #print("Re = ", ConstRe)
    #print("w wall = ", ConstW)
    # add try statement for addition of Re
    try:
        for ii in range(10):
            print("i is ", ii)
            print("Re is ", ConstRe)
            print("w wall is ", ConstW)
            effective_diffusion.interpolate(2*(1/Re + MuT(k, w)))
            #File_d.write(effective_diffusion, k, w)
            print("Solve Navier_Stokes")
            NVS1.solve()
            z.assign(prev_z * (1 - relax_z) + z * relax_z)
            prev_z.assign(z)
            #print(f"DEBUG (After Func1 solve): u_min = {u_solution.dat.data.min()}, u_max = {u_solution.dat.data.max()}")
            #print(f"DEBUG (After Func1 solve): p_min = {p_solution.dat.data.min()}, p_max = {p_solution.dat.data.max()}")
            #print("k min before NVS2 is ", k.dat.data.min())
            #print("k max before NVS2 is ", k.dat.data.max())
            print("Solve NVS2")
            NVS2.solve()
            k.assign(prev_k * (1 - relax_k) + k * relax_k)
            prev_k.assign(k)
            #print("k min after NVS2 is ", k.dat.data.min())
            #print("k max after NVS2 is ", k.dat.data.max())
            #print("w min before NVS3 is ", w.dat.data.min())
            #print("w max before NVS3 is ", w.dat.data.max())
            print("Solve NVS3")
            NVS3.solve()
            w.assign(prev_w * (1 - relax_w) + w * relax_w)
            prev_w.assign(w)
            #print("w min after NVS3 is ", w.dat.data.min())
            #print("w max after NVS3 is ", w.dat.data.max())
            #for jj in range(10):
            # print("i is ", ii,", j is ", jj, ", foo is ", foo)
        File.write(u_, p_, k, w, time = ConstRe)

        if (ConstRe == 5100):
            switch = True
        if (switch == False):
            ConstRe = min(ConstRe + Re_increment, 5100)
            Re.assign(ConstRe)
        else:
            ConstW = min(ConstW + w_increment, 10**10)
            w_wall.assign(ConstW)

        prev_z_try.assign(z)
        prev_k_try.assign(k)
        prev_w_try.assign(w)
    except:
        print("")
        print("error")
        print("")

        if (switch == False):
            ConstRe = ConstRe - Re_increment/2
            Re_increment = Re_increment / 2
            Re.assign(ConstRe)
        else:
            ConstW = ConstW - w_increment / 2
            w_wall.assign(ConstW)
            w_increment = w_increment * 0.5

        #assign previous values of z, k, w
        z.assign(prev_z_try)
        k.assign(prev_k_try)
        w.assign(prev_w_try)

"""

for jj in range(100):
    for ii in range(10):
            print("i is ", ii)
            print("Re is ", ConstRe)
            print("w wall is ", ConstW)
            effective_diffusion.interpolate(2*(1/Re + MuT(k, w)))
            #File_d.write(effective_diffusion, k, w)
            print("Solve Navier_Stokes")
            NVS1.solve()
            z.assign(prev_z * (1 - relax_z) + z * relax_z)
            prev_z.assign(z)
            #print(f"DEBUG (After Func1 solve): u_min = {u_solution.dat.data.min()}, u_max = {u_solution.dat.data.max()}")
            #print(f"DEBUG (After Func1 solve): p_min = {p_solution.dat.data.min()}, p_max = {p_solution.dat.data.max()}")
            #print("k min before NVS2 is ", k.dat.data.min())
            #print("k max before NVS2 is ", k.dat.data.max())
            print("Solve NVS2")
            NVS2.solve()
            k.assign(prev_k * (1 - relax_k) + k * relax_k)
            prev_k.assign(k)
            #print("k min after NVS2 is ", k.dat.data.min())
            #print("k max after NVS2 is ", k.dat.data.max())
            #print("w min before NVS3 is ", w.dat.data.min())
            #print("w max before NVS3 is ", w.dat.data.max())
            print("Solve NVS3")
            NVS3.solve()
            w.assign(prev_w * (1 - relax_w) + w * relax_w)
            prev_w.assign(w)
            #print("w min after NVS3 is ", w.dat.data.min())
            #print("w max after NVS3 is ", w.dat.data.max())
            #for jj in range(10):
            # print("i is ", ii,", j is ", jj, ", foo is ", foo)
    File.write(u_, p_, k, w, time = ConstRe)

# write
print("k min is ", k.dat.data.min())
print("w min is ", w.dat.data.min())
print("k max is ", k.dat.data.max())
print("w max is ", w.dat.data.max())

print("w wall is ", ConstW)
print("Re is ", ConstRe)
