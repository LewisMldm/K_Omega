from firedrake import *
from firedrake.adjoint import *
import signal
import time
#import numpy as npi

continue_annotation()

mesh = Mesh('DomainThinCoarse.msh')
#mesh_hierarchy = MeshHierarchy(base_mesh, refinement_levels=2)
#mesh = mesh_hierarchy[-1]

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
w_w = 10
w_i = 1 #Constant(46.385)
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
Re = Function(R).assign(1)
FlInt = 0.05 # Fluid Intensity
TurLS = 0.22 # Turbulence length scale


def StrT(u):
    "Symmetric stress tensor"
    return 0.5*(grad(u) + grad(u).T)

Id = Identity(mesh.geometric_dimension())

def MuT(k, w):
    "Eddy viscosity."
    if norm(u) == 0 or norm(k) == 0:
        #import ipdb; ipdb.set_trace()
        return Constant(0)
    else:
       	return de*k/w

def Tau(k, w, u):
    """Auxiliary tensor to help with dissipation rate equation"""
    return 2*(de/w)*StrT(u) - (2/3)*de*Id

def RsT(k, w, u):
    """Reynolds Stress Tensor"""
    return k*Tau(k, w, u)

u_init, p_init, k_init, w_init = z.subfunctions
u_init.assign(0.0)
p_init.assign(0.0)
k_init.assign(0.1)
w_init.assign(10.0)

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

F = F1 + F2 + F3

Ubdry1 =Function(V).interpolate(as_vector([0, JetIn*((x-7)*(x-5))]))
Ubdry2 =Function(M).interpolate((JetIn**2)*(-(x-7)*(x-5))*0.04)
C_mu_quarter = 0.5477  # (BetaS)**0.25
TurLS = 0.22

# w = sqrt(k) / (C_mu^1/4 * TurLS)
W_inlet_expr = sqrt(Ubdry2 + 1e-6) / (C_mu_quarter * TurLS)
Ubdry3 = Function(N).interpolate(W_inlet_expr)

bc = [DirichletBC(Z.sub(0), Ubdry1, (19,)),
       DirichletBC(Z.sub(0), Constant((0,0)), (21, 22)),
       DirichletBC(Z.sub(2), Ubdry2, (19,)),
       DirichletBC(Z.sub(2), Constant(0), (21, 22)),
       DirichletBC(Z.sub(3), Ubdry3, (19,)),
       DirichletBC(Z.sub(3), Constant(w_w), (21, 22))]

appctx = {"Re": Re, "velocity_space": 0}

parameters = {
    "snes_type": "newtonls",
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-3,
    "ksp_atol": 1.0e-3,
    "snes_rtol": 1.0e-3,
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

Pol = (Diff_coef*inner(grad(t),grad(l))*dx + dot(u, grad(t))*l*dx - 100*exp(-10 * ((x - 8.5)**2 + (y - 0)**2))*l*dx
      )

bcp = [DirichletBC(T, Constant(0), (19))]

NVS.solve()
solve(Pol == 0, t, bcs=bcp)


J = assemble(conditional(le(x, 4), exp(10*t), Constant(0)) * dx) + 0.1*assemble(JetIn * ds(19))
Jhat = ReducedFunctional(J, [Control(JetIn)])

def Recur(CurrRe, WantedRe, parameters, NVP):
    stop_annotating()
    try:
        Re.assign(float(WantedRe))
        solve_with_timeout(NVP, z, 60.0, parameters)
        #NVS.solve()
        solve(Pol == 0, t, bcs=bcp)
        return float(WantedRe)
    except:
        ConAnn(CurrRe, WantedRe, parameters, NVP)

def ConAnn(CurrRe, WantedRe, parameters, NVP):
    values = list(np.linspace(CurrRe, WantedRe, 10)) #list(np.arange(1.0, WantedRe, 5))
    values.append(WantedRe)
    Cur_Succ_Re = CurrRe
    if CurrRe == WantedRe:
        input("something wrong")
    for value in values[1:]:
        print("Re = ", value)
        Recur(CurrRe, value, parameters, NVP)
        Curr_Succ_Re = recur(Curr_Succ_Re, value)

    return u

def solve_with_timeout(problem, solution, timeout_seconds, parameters):
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def alarm_handler(signum, frame):
        raise SolverTimeoutError(
            f"Solver timed out after {elapsed():.2f} seconds (limit: {timeout_seconds}s)."
        )

    def snes_monitor(snes, iteration, rnorm):
        if elapsed() > timeout_seconds:
            snes.setConvergedReason(PETSc.SNES.ConvergedReason.DIVERGED_USER)
            raise SolverTimeoutError(
                f"SNES timed out after {elapsed():.2f} seconds (limit: {timeout_seconds}s)."
            )

    def ksp_monitor(ksp, iteration, rnorm):
        if elapsed() > timeout_seconds:
            ksp.setConvergedReason(PETSc.KSP.ConvergedReason.DIVERGED_USER)

    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)
    snes = solver.snes
    ksp = snes.getKSP()

    snes.setMonitor(snes_monitor)
    ksp.setMonitor(ksp_monitor)

    # Set an OS-level alarm so even a slow single KSP/MUMPS step is interrupted
    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))

    try:
        solver.solve()
        if elapsed() > timeout_seconds:
            raise SolverTimeoutError(
                f"Solver exceeded {timeout_seconds}s limit."
            )
    finally:
        # Always cancel the timer and restore the previous signal handler
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        ksp.monitorCancel()
        snes.monitorCancel()

# plotting tools
u_, p_, k_, w_ = z.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")
w_.rename("Specific Dissipation rate")
k_.rename("Specific Kinetic Energy")
t.rename("Polutant concentration")

File = VTKFile("kOm_mono_optimized_jet.pvd")
Re.assign(1e8)
#solve_with_timeout(NVP, z, 10.0, parameters)
Recur(float(Re), 1e6, parameters, NVP)
File.write(u_, p_, k_, w_, t)
get_working_tape().progress_bar = ProgressBar
optval = minimize(Jhat)
print(f"Vent vel = {float(JetIn):1.2f}, J = {float(Jhat(JetIn)):1.2e}")
get_working_tape().progress_bar = ProgressBar
optval = minimize(Jhat)
JetIn.interpolate(optval)
print(f"Vent vel = {float(JetIn):1.2f}, J = {float(Jhat(JetIn)):1.2e}")
Ubdry1.interpolate(as_vector([0, JetIn*((x-7)*(x-5))]))
Ubdry2.interpolate(JetIn*((x-7)*(x-5))*JetIn*((x-7)*(x-5))*0.2*0.2)
print("final solve opt stokes")
NVS.solve()
print("final solve opt con diff")
solve(Pol == 0, t, bcs=bcp)
File.write(u_, p_, k_, w_, t)

"""
for jj in range(100):
    print("Re is ", float(Re))
    print("w wall is ", float(w_wall))
    NVS.solve()
    print("Begin solve con diff")
    solve(Pol == 0, t, bcs=bcp)
    print("solves done")

    if (ConstRe >= 1e8):
        File.write(u_, p_, k_, w_, t)
        get_working_tape().progress_bar = ProgressBar
        optval = minimize(Jhat)
        print(f"Vent vel = {float(JetIn):1.2f}, J = {float(Jhat(JetIn)):1.2e}")
        get_working_tape().progress_bar = ProgressBar
        optval = minimize(Jhat)
        JetIn.interpolate(optval)
        print(f"Vent vel = {float(JetIn):1.2f}, J = {float(Jhat(JetIn)):1.2e}")
        Ubdry1.interpolate(as_vector([0, JetIn*((x-7)*(x-5))]))
        Ubdry2.interpolate(JetIn*((x-7)*(x-5))*JetIn*((x-7)*(x-5))*0.2*0.2)

        print("final solve opt stokes")
        NVS.solve()

        print("final solve opt con diff")
        solve(Pol == 0, t, bcs=bcp)

        File.write(u_, p_, k_, w_, t)
        break
    else:
        ConstRe = ConstRe*1.5
        Re.assign(ConstRe)
"""
