from firedrake import *

import gmsh

def MeshGen(Hmax, elementOrder, elementType):
  # Given Hmax, construct a mesh to be read by Firedrake
  gmsh.initialize()
  gmsh.model.add('airCurtain')
  meshObject = gmsh.model

  # Add points
  # Points for the outer boundary
  point1  = meshObject.geo.addPoint(0,   0,   0, Hmax,1)
  point2  = meshObject.geo.addPoint(0,   3,   0, Hmax,2) # ceiling with air curtain
  point3  = meshObject.geo.addPoint(5,   3,   0, Hmax,3) # air curtain starts
  point4  = meshObject.geo.addPoint(6,   3,   0, Hmax,4) # air curtain ends
  point5  = meshObject.geo.addPoint(10,  3,   0, Hmax,5) # end of begining room
  point6  = meshObject.geo.addPoint(10,  5,   0, Hmax,6) # ceiling of room with contolled temp
  point7  = meshObject.geo.addPoint(15,  5,   0, Hmax,7) # end of ceiling
  point8  = meshObject.geo.addPoint(15,  0,   0, Hmax,8)
  point9  = meshObject.geo.addPoint(4,   0,   0, Hmax,9) # end of step for dual jet
  point10 = meshObject.geo.addPoint(4,   0.2, 0, Hmax,10) # top of step
  point11 = meshObject.geo.addPoint(3.8, 0.2, 0, Hmax,11) # top left of step 
  point12 = meshObject.geo.addPoint(3.8, 0.15,0, Hmax,12) # below thickness of step
  point13 = meshObject.geo.addPoint(3.95,0.15,0, Hmax,13)
  point14 = meshObject.geo.addPoint(3.95,0   ,0, Hmax,14) # bottom of step

  # Construct lines from points
  line1 = meshObject.geo.addLine(1, 2) # inflow/outflow boundary
  line2 = meshObject.geo.addLine(2, 3) # ceiling before air curtain
  line3 = meshObject.geo.addLine(3, 4) # air curtain jet
  line4 = meshObject.geo.addLine(4, 5) # air curtain to end of non controlled room
  line5 = meshObject.geo.addLine(5, 6) 
  line6 = meshObject.geo.addLine(6, 7) # ceiling of controlled room
  line7 = meshObject.geo.addLine(7, 8) # far wall of controlled room
  line8 = meshObject.geo.addLine(8, 9) 
  line9 = meshObject.geo.addLine(9, 10) # beginning of step |
  line10 = meshObject.geo.addLine(10, 11) # top of step -
  line11 = meshObject.geo.addLine(11, 12) # edge of step
  line12 = meshObject.geo.addLine(12, 13) # underside of top of step -
  line13 = meshObject.geo.addLine(13, 14) # end of step |
  line14 = meshObject.geo.addLine(14, 1) # step to inflow/outflow

  # Construct closed curve loops
  outerBoundary = meshObject.geo.addCurveLoop([line1, line2, line3, line4, line5, line6,
                                               line7, line8, line9, line10, line11,
                                               line12, line13, line14])
  
  # Define the domain as a 2D plane surface with holes
  domain2D = meshObject.geo.addPlaneSurface([outerBoundary])

  # Synchronize gmsh
  meshObject.geo.synchronize()

  # Add physical groups for firedrake
  meshObject.addPhysicalGroup(2, [domain2D], name='domain')

  meshObject.addPhysicalGroup(1, [line1], 1)
  meshObject.addPhysicalGroup(1, [line2], 2)
  meshObject.addPhysicalGroup(1, [line3], 3)
  meshObject.addPhysicalGroup(1, [line4], 4)
  meshObject.addPhysicalGroup(1, [line5], 5)
  meshObject.addPhysicalGroup(1, [line6], 6)
  meshObject.addPhysicalGroup(1, [line7], 7)
  meshObject.addPhysicalGroup(1, [line8], 8)
  meshObject.addPhysicalGroup(1, [line9], 9)
  meshObject.addPhysicalGroup(1, [line10], 10)
  meshObject.addPhysicalGroup(1, [line11], 11)
  meshObject.addPhysicalGroup(1, [line12], 12)
  meshObject.addPhysicalGroup(1, [line13], 13)
  meshObject.addPhysicalGroup(1, [line14], 14)

  # Set element order
  meshObject.mesh.setOrder(elementOrder)

  if elementType == 2:
    # Generate quad mesh from triangles by recombination
    meshObject.mesh.setRecombine(2, domain2D)

  # Generate the mesh
  gmsh.model.mesh.generate(2)

  gmsh.write('airCurtain.msh')

  gmsh.finalize()

  return

import math
import numpy as np
from firedrake.adjoint import *
continue_annotation()
#MeshGen(0.0001, 1, 1)

mesh = Mesh('dumbbell.msh')
x, y = SpatialCoordinate(mesh)

# Taylor hood elements
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
M = FunctionSpace(mesh, "CG", 1)
N = FunctionSpace(mesh, "CG", 1)
T = FunctionSpace(mesh, "CG", 2)
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
t = Function(T)
t_prev = Function(T)
l = TestFunction(T)

epsilon = 1e-10

# omega wall constant
w_wall = Constant(1e4)
w_inflow = Constant(4.175)

# closure coefficients
alpha = Constant(5/9)
Beta = Constant(3/40)
BetaS = Constant(9/100)
SigS = Constant(0.5)
Sig = Constant(0.5)

# fluid constants
de = Constant(1) # density
Re = Constant(4)
FlInt = 0.05 # Fluid Intensity
TurLS = 0.22 # Turbulence length scale
K = Constant(0.6) # thermal conductivity
C = Constant(4.184) # specific heat capacity kJ/kg.K


R = FunctionSpace(mesh, 'R', 0)
tin = Function(R).assign(-1.)
twall = Function(R).assign(-5.)
angle = Function(R).assign(3*np.pi/2)


g_ = - (x - 0.3) * (x + 0.3) / (0.3 ** 2)
g = as_vector([g_*cos(angle), g_*sin(angle)])

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

n = K/(de*C)

Ft = (inner(grad(t),grad(l))*dx + dot(u, grad(t))*l*dx
      )

F = F1 + F2 + F3
x, y = SpatialCoordinate(mesh)

bcu = [DirichletBC(Z.sub(0), g, 23), 
       DirichletBC(Z.sub(0), 0, [20, 22])]
bck = [DirichletBC(M, (g**2) * (0.2**2), 23),
       DirichletBC(M, 0, (20, 22))] 
bcw = [DirichletBC(N, w_wall, [20, 22])]
bct = [DirichletBC(T, 20, 21),
        DirichletBC(T, twall, 22),
        DirichletBC(T, tin, 23)]
# plotting tools
u_, p_ = z.subfunctions
u_.rename("Mean Velocity")
p_.rename("Pressure")
w.rename("Specific Dissipation rate")
k.rename("Specific Kinetic Energy")
t.rename("Temperature")

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
appctx = {"Re": Re, "velocity_space": 0}

k_lower_limit = 1e-12
full_lower_bounds = Function(M).assign(k_lower_limit)

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
        "snes_monitor": None,
        }

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

k_params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "ngs",  # Use Newton's method with line search
        "ksp_type": "fgmres",  # Direct solver for the linear system
        "pc_type": "ilu",  # Use ILU decomposition for preconditioning
        "snes_converged_reason": "",  # Print convergence reason
        "snes_monitor": "",  # Monitor iterations during the solve
        "ksp_rtol": 1.0e-3,
        "snes_rtol": 1.0e-3,  # Set your desired relative tolerance for SNES here
        "snes_atol": 1.0e-3, # You can also set the absolute tolerance for SNES
        "snes_max_it": 100000,    # And the maximum number of iterations
        "ksp_converged_reason":"",
        "ksp_monitor":"",
        "ksp_max_it": 100000,
        "snes_stol": 1.0e-5,
        "snes_linesearch_monitor": None,
        "ksp_monitor_true_residual": None,
        "snes_line_search_type": "bt", # Backtracking line search (often good)
        "snes_linesearch_alpha": 1e-12, # Factor by which to reduce step if func increases (default 1e-4)
        "snes_linesearch_maxstep": 1.0,
        }

w_params = {
        "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
        "snes_type": "qn",  # Use Newton's method with line search
        "ksp_type": "gmres",  # Direct solver for the linear system
        "pc_type": "ilu",  # Use ILU decomposition for preconditioning
        "snes_converged_reason": "",  # Print convergence reason
        "snes_monitor": "",  # Monitor iterations during the solve
        "ksp_rtol": 1.0e-5,
        "snes_rtol": 1.0e-5,  # Set your desired relative tolerance for SNES here
        "snes_atol": 1.0e-5, # You can also set the absolute tolerance for SNES
        "snes_max_it": 100000,    # And the maximum number of iterations
        "ksp_converged_reason":"",
        "ksp_monitor":"",
        "ksp_max_it": 100000,
        "snes_stol": 1.0e-5,
        "snes_linesearch_monitor": None,
        "ksp_monitor_true_residual": None,
        }

File = VTKFile("AirCurtain_initial_kOm.pvd")

ConstW = 1e5
ConstW_inflow = 4.175
ConstRe = 4

u_solution, p_solution = z.subfunctions
P1_scalar = FunctionSpace(mesh, "CG", 1)

z_prev.assign(z)
k_prev.assign(k)
w_prev.assign(w)

alpha_relax_u = 0.5
alpha_relax_k = 0.1
alpha_relax_w = 0.1


for foo in range(100):
    print("Re = ", ConstRe)
    for ii in range(5):
        solve(F1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
            #z.assign(z_prev * (1 - alpha_relax_u) + z * alpha_relax_u)
            #z_prev.assign(z)

        print(f"DEBUG (After Func1 solve): u_min = {u_solution.dat.data.min()}, u_max = {u_solution.dat.data.max()}")
        print(f"DEBUG (After Func1 solve): p_min = {p_solution.dat.data.min()}, p_max = {p_solution.dat.data.max()}")

        for jj in range(10):
            print("i is ", ii,", j is ", jj, ", foo is ", foo)

            print("k min is ", k.dat.data.min())
            print("w min is ", w.dat.data.min())
            print("k max is ", k.dat.data.max())
            print("w max is ", w.dat.data.max())

            k_over_w = Function(M).interpolate(k/w)
            print(f"DEBUG: Min k/w = {k_over_w.dat.data.min()}, Max k/w = {k_over_w.dat.data.max()}")
            one_over_w = Function(N).interpolate(1/w)
            print(f"DEBUG: Min 1/w = {one_over_w.dat.data.min()}, Max 1/w = {one_over_w.dat.data.max()}")
                
            solve(F2 == 0, k, bcs=bck,  solver_parameters = params)
            print("k min is ", k.dat.data.min())
            print("w min is ", w.dat.data.min())
            print("k max is ", k.dat.data.max())
            print("w max is ", w.dat.data.max())
            
            k_over_w = Function(M).interpolate(k/w)
            print(f"DEBUG: Min k/w = {k_over_w.dat.data.min()}, Max k/w = {k_over_w.dat.data.max()}")
            one_over_w = Function(N).interpolate(1/w)
            print(f"DEBUG: Min 1/w = {one_over_w.dat.data.min()}, Max 1/w = {one_over_w.dat.data.max()}")

            solve(F3 == 0, w, bcs=bcw, solver_parameters = params)

    print("solve temperature")
    solve(Ft == 0, t, bcs=bct)
    print("finished solving temperature")

    if (ConstRe == 4):
        File.write(u_, p_, k, w, t)
        break
        #ConstW *= 2
        #w_wall = Constant(ConstW)

    ConstRe = min(2 * ConstRe, 4)
    print("ConstRe was doubled")
    Re = Constant(ConstRe)
    print("Re assigned")

J = assemble(conditional(ge(x,1), exp(10*(t + 4)), Constant(0)) * dx) + 0.1*assemble(10*tin**2 * ds(23) + twall**2 * ds(22))
Jhat = ReducedFunctional(J, [Control(tin), Control(twall)])
#Jhat = ReducedFunctional(J, [Control(tin), Control(twall), Control(angle)])

stop_annotating()
print(f"Jet temperature = {float(tin):1.2f}, wall temperature = {float(twall):1.2f}")
optval = minimize(Jhat)
tin.assign(optval[0])
twall.assign(optval[1])
print(f"Jet temperature = {float(tin):1.2f}, wall temperature = {float(twall):1.2f}")

for ii in range(5):
    solve(F1 == 0, z, bcs=bcu, nullspace=nullspace, solver_parameters=parameters, appctx=appctx)
    #z.assign(z_prev * (1 - alpha_relax_u) + z * alpha_relax_u)
    #z_prev.assign(z)

    print(f"DEBUG (After Func1 solve): u_min = {u_solution.dat.data.min()}, u_max = {u_solution.dat.data.max()}")
    print(f"DEBUG (After Func1 solve): p_min = {p_solution.dat.data.min()}, p_max = {p_solution.dat.data.max()}")

    for jj in range(10):
        print("i is ", ii,", j is ", jj, ", foo is ", foo)

        print("k min is ", k.dat.data.min())
        print("w min is ", w.dat.data.min())
        print("k max is ", k.dat.data.max())
        print("w max is ", w.dat.data.max())

        k_over_w = Function(M).interpolate(k/w)
        print(f"DEBUG: Min k/w = {k_over_w.dat.data.min()}, Max k/w = {k_over_w.dat.data.max()}")
        one_over_w = Function(N).interpolate(1/w)
        print(f"DEBUG: Min 1/w = {one_over_w.dat.data.min()}, Max 1/w = {one_over_w.dat.data.max()}")

        solve(F2 == 0, k, bcs=bck,  solver_parameters = params)
        print("k min is ", k.dat.data.min())
        print("w min is ", w.dat.data.min())
        print("k max is ", k.dat.data.max())
        print("w max is ", w.dat.data.max())

        k_over_w = Function(M).interpolate(k/w)
        print(f"DEBUG: Min k/w = {k_over_w.dat.data.min()}, Max k/w = {k_over_w.dat.data.max()}")
        one_over_w = Function(N).interpolate(1/w)
        print(f"DEBUG: Min 1/w = {one_over_w.dat.data.min()}, Max 1/w = {one_over_w.dat.data.max()}")

        solve(F3 == 0, w, bcs=bcw, solver_parameters = params)

solve(Ft == 0, t, bcs=bct)

print("Re is ", ConstRe)
print("w wall is ", ConstW)

VTKFile("optimized_curtain_kOm.pvd").write(u_, p_, k, w, t)

