from firedrake import *
from firedrake.adjoint import *
import numpy as npi

mesh = Mesh('backward-facing-step.msh')

#N = 64

#mesh = UnitSquareMesh(N, N)

# Taylor hood elements
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
M = FunctionSpace(mesh, "CG", 1)
N = FunctionSpace(mesh, "CG", 1)
Z = V*Q*M*N
T = FunctionSpace(mesh, "CG", 1)

# Functions and Test functions
z = Function(Z)
prev_z = Function(Z)
u, p, k, w = split(z)
v, q, r, s = TestFunctions(Z)
t = Function(T, name="polutant concentration")
l = TestFunction(T)

# variables
w_wall = Constant(10e10)
w_inflow = Constant(46.385)
Diff_coef = Constant(1)
R = FunctionSpace(mesh, 'R', 0)
JetIn = Function(R).interpolate(1)
Vent = Function(R).interpolate(1)
JetCost = 1
VentCost = 1

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

z.assign(0.5)

# weak form rans
F1 = (de*inner(dot(grad(u), u), v)*dx - p*div(v)*dx + q*div(u)*dx
      + 2*(Re**-1)*inner(StrT(u), StrT(v))*dx
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

F = F1 + F2 + F3

Ubdry1 = Function(V).interpolate(as_vector([0, JetIn*((x-7)*(x-5))]))
Ubdry2 = Function(M).interpolate((JetIn**2)*((x-7)*(x-5))*0.04)

#bc = [DirichletBC(Z.sub(0), as_vector([-y*(10*y - 5)*(8/5), 0]), 16),
#       DirichletBC(Z.sub(0), Constant((0, 0)), 18),
#       DirichletBC(Z.sub(2), Constant(0), 18),
#       DirichletBC(Z.sub(2), Constant(0.015)*(-y*(10*y - 5)*(8/5)), 16), # 0.015 true bc for k
#       DirichletBC(Z.sub(3), w_wall, 18),
#       DirichletBC(Z.sub(3), (w_wall - w_inflow)*(4*y-(1-(w_inflow/(w_inflow-w_wall))**0.5))*(4*y-(1+(w_inflow/(w_inflow-w_wall))**0.5)), 16)]

bc = [DirichletBC(Z.sub(0), Constant((1, 0)), 16),
       DirichletBC(Z.sub(0), Constant((0, 0)), 18),
       DirichletBC(Z.sub(2), Constant(0), 18),
       DirichletBC(Z.sub(2), Constant(0.015), 16), # 0.015 true bc for k
       DirichletBC(Z.sub(3), w_wall, 18),
       DirichletBC(Z.sub(3), w_inflow, 16)]


#bc = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
#	DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3)),
#	DirichletBC(Z.sub(2), Constant(0), (1, 2, 3)),
#	DirichletBC(Z.sub(2), Constant(0.04), (4)),
#	DirichletBC(Z.sub(3), Constant(2), (4)),
#	DirichletBC(Z.sub(3), Constant(w_wall), (1, 2, 3))]

appctx = {"Re": Re, "velocity_space": 0}

parameters = {
    "snes_type": "newtonls",
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-5,
    "ksp_atol": 1.0e-5,
    "snes_rtol": 1.0e-5,
    "ksp_gmres_modifiedgramschmidt": True,
    "snes_monitor": None,
    "ksp_monitor": None,
    
    # --- Top-Level Split: Navier-Stokes (0) vs Turbulence (1) ---
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",  # Multiplicative or block Gauss-Seidel coupling
    
    # If your MixedFunctionSpace is flat (e.g., V * P * K * W), uncomment the lines below 
    # to explicitly group the velocity/pressure and k/omega fields:
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2,3",

    # --- BLOCK 0: Navier-Stokes (Schur Complement) ---
    "fieldsplit_0_ksp_type": "fgmres",
    "fieldsplit_0_pc_type": "fieldsplit",
    "fieldsplit_0_pc_fieldsplit_type": "schur",
    "fieldsplit_0_pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_pc_fieldsplit_off_diag_use_amat": True,
    
    # Block 0 -> Sub-field 0 (Velocity)
    "fieldsplit_0_fieldsplit_0_pc_type": "python",
    "fieldsplit_0_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_fieldsplit_0_Mp_mat_type": "aij",
    "fieldsplit_0_fieldsplit_0_assembled_pc_type": "lu",
    "fieldsplit_0_fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    #"fieldsplit_0_fieldsplit_0_pc_factor_mat_mumps_icntl_14": 200,
    
    # Block 0 -> Sub-field 1 (Pressure)
    "fieldsplit_0_fieldsplit_1_pc_type": "python",
    "fieldsplit_0_fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
    "fieldsplit_0_fieldsplit_1_Mp_mat_type": "aij",
    "fieldsplit_0_fieldsplit_1_Mp_pc_type": "lu",
    "fieldsplit_0_fieldsplit_1_pc_factor_mat_solver_type": "mumps",

    # --- BLOCK 1: Turbulence Model (k and omega) ---
    "fieldsplit_1_ksp_type": "fgmres",  # Uses direct solver for the transport block
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_1_assembled_pc_type": "lu",
    "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    #"fieldsplit_1_pc_factor_mat_mumps_icntl_14": 400,
}

NVP = NonlinearVariationalProblem(F, z, bcs=bc)
NVS = NonlinearVariationalSolver(NVP, solver_parameters=parameters, appctx=appctx)

ConstRe = 1

prev_z.assign(z)

#Pol = (Diff_coef*inner(grad(t),grad(l))*dx + dot(u, grad(t))*l*dx - 100*exp(-10 * ((x - 8.5)**2 + (y - 0)**2))*l*dx
#      )

#bcp = [DirichletBC(T, Constant(0), (19))]

# plotting tools
u_, p_, k_, w_ = z.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")
w_.rename("Specific Dissipation rate")
k_.rename("Specific Kinetic Energy")
#t.rename("Polutant concentration")

File = VTKFile("kOm_monoBFS_vague.pvd")

for jj in range(100):
    print("Re is ", float(Re))
    print("w wall is ", float(w_wall))
    NVS.solve()
    File.write(u_, p_, k_, w_, time=ConstRe)

    if (ConstRe == 5100):
        break
    else:
        ConstRe = min(ConstRe*2, 5100)
        Re.assign(ConstRe)
