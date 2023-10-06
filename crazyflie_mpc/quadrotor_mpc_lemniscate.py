from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt,  cos, sin, norm_2, tanh
from scipy.linalg import block_diag
from quadrotor2 import Quadrotor
from threading import Thread
from time import sleep, time
from pathlib import Path
import importlib
import sys
import os

class QuadrotorMPC(Thread):
    def __init__(self, name: str, generate_c_code, quadrotor: Quadrotor, horizon: float, num_steps: int,  solution_callback, x0, xr):
        super(QuadrotorMPC, self).__init__(target=self.solve_mpc_control, args=(x0, xr))
        self.model_name = name
        self.quad = quadrotor
        self.horizon = horizon
        self.num_steps = num_steps
        self.solution_callback = solution_callback
        self.ocp_solver = None
        self.acados_generated_files_path = Path(__file__).parent.parent.resolve() / 'acados_generated_files'
        if generate_c_code:
            self.generate_mpc(x0)
        else:
            try:
                sys.path.append(str(self.acados_generated_files_path / (self.model_name + '_c_generated_code')))
                acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx')
                # from acados_ocp_solver_pyx import AcadosOcpSolverCython
                self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)
            except ImportError:
                self.generate_mpc(x0)
    def generate_mpc(self, x0):
        f_expl, x, u = self.quad.dynamics()
        # Define the Acados model 
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = self.model_name

        # Define the optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        ocp.code_export_directory = str(self.acados_generated_files_path / (self.model_name + '_c_generated_code'))
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of controls
        ny = nx + nu - 1 # size of intermediate cost reference vector in least squares objective
        ny_e = nx - 1 # size of terminal reference vector

        N = self.num_steps
        Tf = self.horizon
        ocp.dims.N = N
        ocp.solver_options.tf = Tf

        # setup state and control reference vectors
        
        uref = self.quad.mass*self.quad.gravity*vertcat(0.0,0.0,0.0,1.0)
        # Intermediate cost coditions\
  
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        Q = diag(horzcat(10., 10., 10., 10., 10., 10., 1., 1., 1.))
        R = diag(horzcat(5., 5., 5., 5.))
        W = block_diag(Q, R)
        t = x[-1]
        a = 1.0
        b = 1.5*tanh(0.1*t)
        initial_position = [0., 0., 1.0]
        pxr = initial_position[0] + a*sin(b*t)
        pyr = initial_position[1] + a*sin(b*t)*cos(b*t)
        pzr = initial_position[2]
        vxr = a*b*cos(b*t)
        vyr = a*b*cos(2*b*t)
        vzr = 0.0
        y = vertcat(x[:-1], u)
        xref = vertcat(pxr, pyr, pzr, vxr, vyr, vzr, 0., 0., 0.)
        yref = vertcat(xref,uref)
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)
        ocp.model.cost_expr_ext_cost_e = 0.
  
        # bounds on control
        max_angle = np.radians(50)
        max_thrust = 0.477627618
        ocp.constraints.lbu = np.array([-max_angle, -max_angle, -np.radians(180), 0.])
        ocp.constraints.ubu = np.array([max_angle, max_angle, np.radians(180), max_thrust])
        ocp.constraints.idxbu = np.array([0,1,2,3])

        # initial state
        ocp.constraints.x0 = x0

        json_file = str(self.acados_generated_files_path / (self.model_name + '_acados_ocp.json'))
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP'
        AcadosOcpSolver.generate(ocp, json_file=json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        # sys.path.append(str(self.acados_generated_files_path / 'c_generated_code'))
        sys.path.append(str(self.acados_generated_files_path / (self.model_name + '_c_generated_code')))
        acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx')
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)
        # self.ocp_solver = AcadosOcpSolver.create_cython_solver(json_file=json_file)

    def solve_mpc_trajectory(self, x0):
        N = self.num_steps
        nx = len(x0)
        nu = 4

        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        start_time = time()
        status = self.ocp_solver.solve()
        # print(time() - start_time)

        # extract state and control solution from solver
        for i in range(N):
            x_mpc[i,:] = self.ocp_solver.get(i, "x")
            u_mpc[i,:] = self.ocp_solver.get(i, "u")
        x_mpc[N,:] = self.ocp_solver.get(N, "x")

        if self.solution_callback is not None:
            self.solution_callback(status, x_mpc, u_mpc)
        else:    
            return status, x_mpc, u_mpc
        
    def solve_mpc_control(self, x0):
        N = self.num_steps
        nx = len(x0)
        nu = 4

        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        start_time = time()
        status = self.ocp_solver.solve()
        # print(time() - start_time)

        x_mpc = self.ocp_solver.get(0, "x")
        u_mpc = self.ocp_solver.get(0, "u")

        return status, x_mpc, u_mpc



