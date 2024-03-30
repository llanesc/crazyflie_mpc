from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt,  cos, sin, norm_2, tanh, GenMX_zeros
from scipy.linalg import block_diag
from .quadrotor2 import Quadrotor
from threading import Thread
from time import sleep, time
from pathlib import Path
import importlib
import sys

class QuadrotorMpcTrajectory:
    def __init__(self, name: str, quadrotor: Quadrotor, horizon: float, num_steps: int, code_export_directory : Path=Path('acados_generated_files'), solution_callback=None):
        self.model_name = name
        self.quad = quadrotor
        self.horizon = horizon
        self.num_steps = num_steps
        self.solution_callback = solution_callback
        self.ocp_solver = None
        # self.acados_generated_files_path = Path(__file__).parent.resolve() / 'acados_generated_files'
        self.acados_generated_files_path = code_export_directory
        try:
            if self.acados_generated_files_path.is_dir():
                sys.path.append(str(self.acados_generated_files_path))
            acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')
            self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)
        except ImportError:
            print('Acados cython code not generated. Generating cython code now...')
            self.generate_mpc()
    def generate_mpc(self):
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

        ocp.code_export_directory = self.acados_generated_files_path / ('c_generated_code')
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of controls
        ny = nx + nu  # size of intermediate cost reference vector in least squares objective
        ny_e = nx # size of terminal reference vector

        N = self.num_steps
        Tf = self.horizon
        ocp.dims.N = N
        ocp.solver_options.tf = Tf

        # setup state and control reference vectors
        
        # uref = self.quad.mass*self.quad.gravity*vertcat(0.0,0.0,0.0,1.0)
        # Intermediate cost coditions\
  
        Q = np.diag([50., 50., 50., 1., 1., 1., 5., 5., 5.])
        R = diag(horzcat(5., 5., 5., 1.))
        W = block_diag(Q,R)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.cost.Vx = np.vstack([np.identity(9), np.zeros((4,9))])
        ocp.cost.Vu = np.vstack([np.zeros((9, 4)),np.identity(4)])
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(13)

        # y = vertcat(x[:-1], u)
        # xref = vertcat(pxr, pyr, pzr, vxr, vyr, vzr)
        # yref = vertcat(xref,uref)

        # ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)
        ocp.model.cost_expr_ext_cost_e = 0.
  
        # bounds on control
        max_angle = np.radians(40)
        max_thrust = 0.477627618
        ocp.constraints.lbu = np.array([-max_angle, -max_angle, -np.radians(180), 0.])
        ocp.constraints.ubu = np.array([max_angle, max_angle, np.radians(180), max_thrust])
        ocp.constraints.idxbu = np.array([0,1,2,3])

        # initial state
        ocp.constraints.x0 = np.zeros(9)

        json_file = str(self.acados_generated_files_path / ('acados_ocp.json'))
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.print_level = 0
        ocp.solver_options.integrator_type = 'ERK'

        AcadosOcpSolver.generate(ocp, json_file=json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)

        if self.acados_generated_files_path.is_dir():
            sys.path.append(str(self.acados_generated_files_path))
        acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)


    def solve_mpc_trajectory(self, x0, yref):
        N = self.num_steps
        nx = len(x0)
        nu = 4
        
        if yref.shape[1] != self.num_steps:
            raise Exception('incorrect size of yref')

        for i in range(yref.shape[1]):
            self.ocp_solver.set(i, 'yref', yref[:,i])

        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        # start_time = time()
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
        
    def solve_mpc_control(self, x0, yref):
        N = self.num_steps
        nx = len(x0)
        nu = 4

        if yref.shape[1] != self.num_steps:
            raise Exception('incorrect size of yref')

        for i in range(yref.shape[1]):
            self.ocp_solver.set(i, 'yref', yref[:,i])

        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        # start_time = time()
        status = self.ocp_solver.solve()
        # print(time() - start_time)

        x_mpc = self.ocp_solver.get(0, "x")
        u_mpc = self.ocp_solver.get(0, "u")

        return status, x_mpc, u_mpc