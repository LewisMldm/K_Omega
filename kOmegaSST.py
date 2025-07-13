from firedrake import *
import math
#mesh = Mesh('BFS_ComparisonFile.msh')
mesh = UnitSquareMesh(64, 64)

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

epsilon = 10**-10

# omega wall constant
w_wall = Constant(100)
w_inflow = Constant(40)

# closure coefficients
alpha1 = Constant(5/9)
alpha2 = Constant(0.44)
Beta1 = Constant(3/40)
Beta2 = Constant(0.0828)
BetaS = Constant(9/100)
SigOm1 = Constant(0.5)
SigOm2 = Constant(0.856)
SigK1 = Constant(0.85)
SigK2 = Constant(1)
a1 = Constant(0.2)

# fluid constants
de = Constant(1)
#mu = Constant(1) # Viscosity
Re = Constant(10)
FlInt = Constant(0.05) # Fluid Intensity
TurLS = Constant(0.22) # Turbulence length scale
x, y = SpatialCoordinate(mesh)
y = y + epsilon
x = x + epsilon

def Dist(x, y):
    "Distance from wall boundary"

    d_bottom = y
    d_top = 1 - y
    d_left = x
    d_right = 1 - x
    raw_dist = min_value(d_bottom, min_value(d_top, min_value(d_left, d_right)))
    return max_value(raw_dist, epsilon)

    """
    v1 = ((0.5 - y)**2)**0.5 # top wall
    v2 = ((0 - y)**2)**0.5 # bottom wall
    v3 = ((y - conditional(le(0.1, y), 0.1, y))**2 + (x - 1)**2)**0.5 # vertical step
    v4 = ((y - 0.1)**2 +(x - conditional(le(1, x), 1, x))**2)**0.5 # horizontal step
    v5 = ((y - conditional(ge(y, 0.1), y, 0.1))**2 + x**2)**0.5 # inlet
    v6 = ((x - 0.4)**2)**0.5
    con1 = conditional(le(v1, v2), v1, v2)
    con2 = conditional(le(v3, v4), v3, v4)
    con3 = conditional(le(v5, v6), v5, v6)
    con4 = conditional(le(con1, con2), con1, con2)
    con5 = conditional(le(con3, con4), con3, con4)
    return conditional(le(con4, con5), con4, con5) # this needs changing for each mesh"""

def F1(k, w, y):
    "F1 auxillary relation"
    con1 = conditional(ge(k**0.5/(BetaS*w*y), 500*(1/Re)/(y*y*w)), k**0.5/(BetaS*w*y), 500*(1/Re)/(y*y*w))
    con2 = conditional(le(con1, 4*SigOm2*k/(CDkw(w, k)*y*y)), con1, 4*SigOm2*k/(CDkw(w, k)*y*y))
    return ufl.tanh(con2)

def F2(k, w, y):
    "F2 auxillary relation"
    return ufl.tanh(conditional(ge(2*k**0.5/(BetaS*w*y), 500*(1/Re)/(y*y*w)), 2*k**0.5/(BetaS*w*y), 500*(1/Re)/(y*y*w)))

def CDkw(w, k):
    "CD_{k omega} auxillary relation"
    return conditional(ge(2*de*SigOm2*(1/w)*dot(grad(k), grad(w)), 10**-10), 2*de*SigOm2*(1/w)*dot(grad(k), grad(w)), Constant(10**-10))

def Pk(u, k, w):
    "Pk auxillary relation"
    return conditional(ge(inner(RsT(k, w, u), StrT(u)), 10*BetaS*k*w), inner(RsT(k, w, u), StrT(u)), Constant(10)*BetaS*k*w)

def StrT(u):
    "Symmetric stress tensor"
    return 0.5*(grad(u) + grad(u).T)

Id = Identity(mesh.geometric_dimension())

def MuT(k, w, x, y):
    "Eddy viscosity."
    return conditional(ge(a1*w, Dist(x, y)*F2(k, w, y)), de*k/w, a1*de*k/(Dist(x, y)*F2(k, w, y)))

def Tau(k, w, u):
    """Auxiliary tensor to help with dissipation rate equation"""
    return 2*(de/w)*StrT(u) - (2/3)*de*Id

def RsT(k, w, u):
    """Reynolds Stress Tensor"""
    return k*Tau(k, w, u)


z.assign(1.5)
k.assign(1.5)
w.assign(1.5)

# weak form rans
Func1 = (de*inner(dot(grad(u), u), v)*dx - p*div(v)*dx + q*div(u)*dx
      + 2*((1/Re) + MuT(k, w, x, y))*inner(StrT(u), StrT(v))*dx 
      + (2/3)*de*dot(grad(k), v)*dx
      )

Func2 = (de*dot(u, grad(k))*r*dx - inner(RsT(k, w, u), StrT(u))*r*dx 
      + BetaS*de*k*w*r*dx
      + ((1/Re) 
      + (SigK1*F1(k, w, y) + SigK2*(1-F2(k, w, y)))*MuT(k, w, x, y))*dot(grad(k), grad(r))*dx
      )

Func3 = (de*inner(u, grad(w))*s*dx - (alpha1*F1(k, w, y) + alpha2*(1 - F2(k, w, y)))*w*inner(Tau(k, w, u), StrT(u))*s*dx
        + (Beta1*F1(k, w, y) + Beta2*(1 - F2(k, w, y)))*de*(w**2)*s*dx + ((1/Re) 
        + (SigOm1*F1(k, w, y) + SigOm2*(1 - F2(k, w, y)))*MuT(k, w, x, y))*dot(grad(w), grad(s))*dx
         - Constant(2)*(1 - F1(k, w, y))*(de*SigOm2/w)*dot(grad(k), grad(w))*s*dx
      )

F = Func1 + Func2 + Func3

"""
bcu = [DirichletBC(Z.sub(0), Constant((1, 0)), (1,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (2, 4, 5, 6))]
bck = [DirichletBC(M, Constant(0), (2, 4, 5, 6)),
       DirichletBC(M, Constant(1), (1,))] # 0.015 true bc for k
bcw = [DirichletBC(N, w_wall, (2, 4, 5, 6)),
       DirichletBC(N, Constant(w_inflow), (1,))] # 0.06262 true bc for w
"""
bcu = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]
bck = [DirichletBC(M, Constant(0), (1, 2, 3)),
       DirichletBC(M, Constant(0.004), (4,))]
bcw = [DirichletBC(N, w_wall, (1, 2, 3)),
       DirichletBC(N, w_inflow, (4,))] 
bca = bcu +  bck + bcw
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
        "ksp_divtol": "1e10",
        "ksp_max_it": "5000",
        "ksp_monitor_true_residual": None,
        "ksp_gmres_modifiedgramschmidt": True,
        "snes_monitor": None
        }

k_lower = Function(M).assign(1e-10) # Small positive lower bound for k
k_upper = Function(M).assign(1e5) # Reasonable upper bound for k (adjust as needed)

w_lower = Function(N).assign(1e-10) # Small positive lower bound for w
w_upper = Function(N).assign(1e10) # Upper bound for w (if w_wall can be large)

k_params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "newtonls",  # Use Newton's method with line search
        "ksp_type": "fgmres",  # Direct solver for the linear system
        "pc_type": "ilu",  # Use ILU decomposition for preconditioning
        "snes_converged_reason": "",  # Print convergence reason
        "snes_monitor": "",  # Monitor iterations during the solve
        "ksp_rtol": 1.0e-3,
        "snes_rtol": 1.0e-3,  # Set your desired relative tolerance for SNES here
        "snes_atol": 1.0e-3, # You can also set the absolute tolerance for SNES
        "snes_max_it": 10000,    # And the maximum number of iterations
        "ksp_converged_reason":"",
        "ksp_monitor":"",
        "ksp_max_it": 10000,
        "snes_stol": 1.0e-3,
        "snes_linesearch_monitor": None,
        "ksp_monitor_true_residual": None,
        "snes_bounds_lower": k_lower, # For the k solve
        "snes_bounds_upper": k_upper, # For the k solve
        }

w_params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "newtonls",  # Use Newton's method with line search
        "ksp_type": "fgmres",  # Direct solver for the linear system
        "pc_type": "ilu",  # Use ILU decomposition for preconditioning
        "snes_converged_reason": "",  # Print convergence reason
        "snes_monitor": "",  # Monitor iterations during the solve
        "ksp_rtol": 1.0e-3,
        "snes_rtol": 1.0e-3,  # Set your desired relative tolerance for SNES here
        "snes_atol": 1.0e-3, # You can also set the absolute tolerance for SNES
        "snes_max_it": 10000,    # And the maximum number of iterations
        "ksp_converged_reason":"",
        "ksp_monitor":"",
        "ksp_max_it": 10000,
        "snes_stol": 1.0e-3,
        "snes_linesearch_monitor": None,
        "ksp_monitor_true_residual": None,
        "snes_bounds_lower": w_lower, # For the w solve
        "snes_bounds_upper": w_upper, # For the w solve
        }

File = VTKFile("kOmegaSSTSoln.pvd")

ConstW = 2
ConstW_inflow = 1
ConstMu = 1

u_solution, p_solution = z.subfunctions
d_wall = Dist(x, y)
P1_scalar = FunctionSpace(mesh, "CG", 1)

z_prev.assign(z)
k_prev.assign(k)
w_prev.assign(w)

alpha_relax_u = 0.1
alpha_relax_k = 0.5
alpha_relax_w = 0.5

for foo in range(100):
    #ConstW = ConstW * 0.5
    #w_wall.assign(ConstW)
    for ii in range(10):
        solve(Func1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
        z.assign(z_prev * (1 - alpha_relax_u) + z * alpha_relax_u)
        z_prev.assign(z)

        print(f"DEBUG (After Func1 solve): u_min = {u_solution.dat.data.min()}, u_max = {u_solution.dat.data.max()}")
        print(f"DEBUG (After Func1 solve): p_min = {p_solution.dat.data.min()}, p_max = {p_solution.dat.data.max()}")

        for jj in range(15):
            print("i is ", ii,", j is ", jj, ", foo is ", foo)
            #print("W wall = ", ConstW)
            # Project and print d_wall values (though this shouldn't change, good to verify)
            d_wall_func_val = Function(P1_scalar).interpolate(d_wall)
            print(f"DEBUG: Min wall distance: {d_wall_func_val.dat.data.min()}")
            print(f"DEBUG: Max wall distance: {d_wall_func_val.dat.data.max()}")

            # Project and print F1, F2, MuT at *current* k, w, y, d_wall values
            f1_val = Function(P1_scalar).interpolate(F1(k, w, y))
            print(f"DEBUG: Min F1: {f1_val.dat.data.min()}, Max F1: {f1_val.dat.data.max()}")

            f2_val = Function(P1_scalar).interpolate(F2(k, w, y))
            print(f"DEBUG: Min F2: {f2_val.dat.data.min()}, Max F2: {f2_val.dat.data.max()}")

            mut_val = Function(P1_scalar).interpolate(MuT(k, w, x, y))
            print(f"DEBUG: Min MuT: {mut_val.dat.data.min()}, Max MuT: {mut_val.dat.data.max()}")

            cdkw_val = Function(P1_scalar).interpolate(CDkw(w, k))
            print(f"DEBUG: Min CDkw: {cdkw_val.dat.data.min()}, Max CDkw: {cdkw_val.dat.data.max()}")

            pk_val = Function(P1_scalar).interpolate(Pk(u, k, w))
            print(f"DEBUG: Min Pk before k solve: {pk_val.dat.data.min()}, Max Pk before k solve: {pk_val.dat.data.max()}")

            print("k min is ", k.dat.data.min())
            print("w min is ", w.dat.data.min())
            print("k max is ", k.dat.data.max())
            print("w max is ", w.dat.data.max())

            k_over_w = Function(M).interpolate(k/w)
            print(f"DEBUG: Min k/w = {k_over_w.dat.data.min()}, Max k/w = {k_over_w.dat.data.max()}") 
            one_over_w = Function(N).interpolate(1/w)
            print(f"DEBUG: Min 1/w = {one_over_w.dat.data.min()}, Max 1/w = {one_over_w.dat.data.max()}")

            solve(Func2 == 0, k, bcs=bck, solver_parameters = k_params)
            print("k min is ", k.dat.data.min())
            print("w min is ", w.dat.data.min())
            print("k max is ", k.dat.data.max())
            print("w max is ", w.dat.data.max())

            k_over_w = Function(M).interpolate(k/w)
            print(f"DEBUG: Min k/w = {k_over_w.dat.data.min()}, Max k/w = {k_over_w.dat.data.max()}")
            one_over_w = Function(N).interpolate(1/w)
            print(f"DEBUG: Min 1/w = {one_over_w.dat.data.min()}, Max 1/w = {one_over_w.dat.data.max()}")

            # Project and print F1, F2, MuT at *current* k, w, y, d_wall values
            f1_val = Function(P1_scalar).interpolate(F1(k, w, y))
            print(f"DEBUG: Min F1: {f1_val.dat.data.min()}, Max F1: {f1_val.dat.data.max()}")

            f2_val = Function(P1_scalar).interpolate(F2(k, w, y))
            print(f"DEBUG: Min F2: {f2_val.dat.data.min()}, Max F2: {f2_val.dat.data.max()}")

            mut_val = Function(P1_scalar).interpolate(MuT(k, w, x, y))
            print(f"DEBUG: Min MuT: {mut_val.dat.data.min()}, Max MuT: {mut_val.dat.data.max()}")

            cdkw_val = Function(P1_scalar).interpolate(CDkw(w, k))
            print(f"DEBUG: Min CDkw: {cdkw_val.dat.data.min()}, Max CDkw: {cdkw_val.dat.data.max()}")

            pk_val = Function(P1_scalar).interpolate(Pk(u, k, w))
            print(f"DEBUG: Min Pk after k solve: {pk_val.dat.data.min()}, Max Pk after k solve: {pk_val.dat.data.max()}")
            solve(Func3 == 0, w, bcs=bcw, solver_parameters = w_params)
    #File.write(u_, p_, k, w)
"""
while (ConstMu >= 1/5100 and ConstMu <= 1 and ConstW <= 10**10 and ConstW >= 1):
    try:
        print("Re = ", 1/ConstMu)
        print("w wall = ", ConstW)
        for ii in range(5):
            print("i is ", ii)
            solve(Func1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
            for jj in range(10):
                print("j is ", jj)
                solve(Func2 == 0, k, bcs=bck, solver_parameters = params)
                solve(Func3 == 0, w, bcs=bcw, solver_parameters = params)
                File.write(u_, p_, k, w)
                w_prev.assign(w)
                k_prev.assign(k)
            z_prev.assign(z)


        if (ConstW == 10**10 and ConstMu == 1/5100):
            break

        if (ConstW == 10**10):
            Alternate = True

        if (Alternate == False):
            ConstW = max(2*ConstW, 10**10)
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
"""
