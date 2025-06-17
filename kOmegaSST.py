from firedrake import *
import math
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
w_wall = Constant(0.1)

# closure coefficients
alpha1 = 5/9
alpha2 = 0.44
Beta1 = 3/40
Beta2 = 0.0828
BetaS = 9/100
SigOm1 = 0.5
SigOm2 = 0.856
SigK1 = 0.85
SigK2 = 1
a1 = 0.2

# fluid constants
de = 1 # density
mu = 1 # Viscosity
Re = 1 / mu
FlInt = 0.05 # Fluid Intensity
TurLS = 0.22 # Turbulence length scale
x, y = SpatialCoordinate(mesh)

def Dist(x, y):
    "Distance from wall boundary"
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
    return conditional(le(con4, con5), con4, con5) # this needs changing for each mesh

def F1(k, w, y):
    "F1 auxillary relation"
    con1 = conditional(ge(k**0.5/(BetaS*w*y), 500*mu/(y*y*w)), k**0.5/(BetaS*w*y), 500*mu/(y*y*w))
    con2 = conditional(le(con1, 4*SigOm2*k/(CDkw(w, k)*y*y)), con1, 4*SigOm2*k/(CDkw(w, k)*y*y))
    return tanh(con2)

def F2(k, w, y):
    "F2 auxillary relation"
    return tanh(conditional(ge(2*k**0.5/(BetaS*w*y), 500*mu/(y*y*w)), 2*k**0.5/(BetaS*w*y), 500*mu/(y*y*w)))

def CDkw(w, k):
    "CD_{k omega} auxillary relation"
    return conditional(ge(2*de*SigOm2*(1/w)*dot(grad(k), grad(w)), 10**-10), 2*de*SigOm2*(1/w)*dot(grad(k), grad(w)), 10**-10)

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

z.assign(0.5)
k.assign(0.5)
w.assign(0.5)

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

bcu = [DirichletBC(Z.sub(0), Constant((1, 0)), (1,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (2, 4, 5, 6)),
       DirichletBC(Z.sub(1), Constant(0), (3,))]
bck = [DirichletBC(M, Constant(0), (2, 4, 5, 6)),
       DirichletBC(M, Constant(0.015), (1,))] # 0.015 true bc for k
bcw = [DirichletBC(N, w_wall, (2, 4, 5, 6)),
       DirichletBC(N, Constant(0.06262), (1,))] # 0.06262 true bc for w

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

File = VTKFile("kOmegaSSTSoln.pvd")

ConstW = 0.5

for foo in range(100):
    ConstW = ConstW * 2
    w_wall.assign(ConstW)
    for ii in range(10):
        solve(F1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
        for jj in range(15):
            print("i is ", ii,", j is ", jj, ", foo is ", foo)
            print("Re = ", 1/ConstMu)
            solve(F2 == 0, k, bcs=bck, solver_parameters = params)
            solve(F3 == 0, w, bcs=bcw, solver_parameters = params)
    #File.write(u_, p_, k, w)
"""
while (ConstMu >= 1/5100 and ConstMu <= 1 and ConstW <= 10**10 and ConstW >= 1):
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
