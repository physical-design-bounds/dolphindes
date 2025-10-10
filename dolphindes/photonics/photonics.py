"""
Classes for calculating QCQP design bounds for photonics problems.

Bridges the QCQP Dual Problem Interface in cvxopt 
and the Maxwell Solvers in maxwell
"""

__all__ = []

import warnings
from typing import Tuple
import copy  # added

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dolphindes.cvxopt import DenseSharedProjQCQP, SparseSharedProjQCQP
from dolphindes.maxwell import TM_FDFD
from dolphindes.util import check_attributes


class Photonics_FDFD():
    """
    Mother class for frequency domain problems with finite difference discretization.

    To allow for lazy initialization, only demands omega upon init
    
    Specification of the photonics design objective:
    if sparseQCQP is False, the objective is specified as a quadratic function of the 
    polarization p: max_p -p^dagger A0 p + 2 Re (p^dagger s0) + c0
    
    if sparseQCQP is True, the objective is specified as a quadratic function of (Gp):
    max_{Gp} -(Gp)^dagger A0 (Gp) + 2 Re((Gp)^dagger s0) + c0
    
    Attributes
    ----------
    omega : complex
        Circular frequency, can be complex to allow for finite bandwidth effects.
    chi : complex
        Bulk susceptibility of material used. 
    Nx : int
        Number of pixels along the x direction.
    Ny : int
        Number of pixels along the y direction.
    Npmlx : int
        Size of the x direction PML in pixels.
    Npmly : int
        Size of the y direction PML in pixels.
    dx : float
        Finite difference grid pixel size in x direction, in units of 1.
    dy : float
        Finite difference grid pixel size in y direction, in units of 1.
    des_mask : boolean ndarray
        Boolean mask over computation domain that is TRUE for pixels in design region.
    ji : complex ndarray
        Incident current source that produces an incident field.
    ei : complex ndarray
        Incident field. 
    chi_background : complex ndarray
        The background structure. 
        The default is None, in which case it is set to vacuum.
    bloch_x : float
        x-direction phase shift associated with the periodic boundary condtions. 
        Default: 0.0
    bloch_y : float
        y-direction phase shift associated with the periodic boundary condtions. 
        Default: 0.0
    sparseQCQP : boolean
        Boolean flag indicating whether the sparse QCQP convention is used.
    A0 : complex np.ndarray or scipy.sparse.csc_array
        A0 array in the QCQP field design objective. 
    s0 : complex np.ndarray
        The vector s0 in the QCQP field design objective. 
    c0 : float
        The constant c0 in the QCQP field design objective. 
    """

    def __init__(self, omega, chi=None, Nx=None, Ny=None, Npmlx=None, Npmly=None, dx=None, dy=None, # FDFD solver attr
                 des_mask=None, ji=None, ei=None, chi_background=None, # design problem attr
                 bloch_x=0.0, bloch_y=0.0, # FDFD solver attr
                 sparseQCQP=None, A0=None, s0=None, c0=0.0, Pdiags=None): # design problem attr
        """
        only omega is absoluted needed for initialization, other attributes can be added later
        """
        self.omega = omega
        self.chi = chi
        
        self.Nx = Nx
        self.Ny = Ny
        self.Npmlx = Npmlx
        self.Npmly = Npmly
        self.dx = dx
        self.dy = dy
        self.bloch_x = bloch_x
        self.bloch_y = bloch_y
        
        self.des_mask = des_mask
        self.ji = ji
        self.ei = ei
        self.chi_background = chi_background
        
        self.sparseQCQP = sparseQCQP
        self.A0 = A0
        self.s0 = s0
        self.c0 = c0
        self.Pdiags = Pdiags

    def __deepcopy__(self, memo):
        """Deep copy this instance."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            try:
                setattr(new, k, copy.deepcopy(v, memo))
            except Exception:
                setattr(new, k, v)  # fallback to reference
        return new

class Photonics_TM_FDFD(Photonics_FDFD):
    def __init__(self, omega, chi=None, 
                 grid_size : Tuple[int, int] = (None, None), 
                 pml_size : Tuple[int, int] = (None, None), 
                 dl=None, # FDFD solver attr
                 des_mask : np.ndarray = None, ji : np.ndarray = None, 
                 ei : np.ndarray = None, chi_background : np.ndarray = None, # design problem attr
                 bloch_x=0.0, bloch_y=0.0, # FDFD solver attr
                 sparseQCQP=True, A0=None, s0=None, c0=0.0): # design problem attr
        
        Nx, Ny = grid_size
        Npmlx, Npmly = pml_size
        self.dl = dl
        self.des_mask = des_mask
        self.Ginv = None
        self.G = None
        self.M = None

        super().__init__(omega, chi, Nx, Ny, Npmlx, Npmly, dl, dl,
                         des_mask, ji, ei, chi_background,
                         bloch_x, bloch_y,
                         sparseQCQP, A0, s0, c0)
        
        try:
            check_attributes(self, 'omega', 'chi', 'Nx', 'Ny', 'Npmlx', 'Npmly', 'des_mask', 'bloch_x', 'bloch_y', 'dl', 'sparseQCQP')
            self.setup_EM_solver()
            self.setup_EM_operators()
        except AttributeError as e:
            warnings.warn("Photonics_TM_FDFD initialized with missing attributes (lazy initialization). We strongly recommend passing all arguments for expected behavior.")

        ## structure adjoint
        if sparseQCQP:
            self.structure_objective = self.structure_objective_sparse
        else:
            self.structure_objective = self.structure_objective_dense


    def __repr__(self):
        return (f"Photonics_TM_FDFD(omega={self.omega}, chi={self.chi}, Nx={self.Nx}, "
                f"Ny={self.Ny}, Npmlx={self.Npmlx}, Npmly={self.Npmly}, dl={self.dl}, "
                f"des_mask={self.des_mask is not None}, ji={self.ji is not None}, "
                f"ei={self.ei is not None}, chi_background={self.chi_background is not None}, "
                f"bloch_x={self.bloch_x}, bloch_y={self.bloch_y}, sparseQCQP={self.sparseQCQP})")

    def set_objective(self, A0=None, s0=None, c0=None, denseToSparse=False):
        """
        Set the QCQP objective function parameters. Not specifying a particular
        parameter leaves it unchanged.
        
        Parameters
        ----------
        A0 : complex np.ndarray or scipy.sparse.csc_array, optional
            The matrix A0 in the QCQP objective function.
        s0 : complex np.ndarray, optional
            The vector s0 in the QCQP objective function.
        c0 : float, optional
            The constant c0 in the QCQP objective function. Default: 0.0
        denseToSparse: bool, optional
            If True, treat input A0 and s0 as describing forms
            of the polarization p, and convert them to the equivalent forms
            of (Gp) before assigning to the class attributes. Default: False
        """
        if denseToSparse:
            if not self.sparseQCQP:
                raise ValueError('sparseQCQP needs to be True to use dense-to-sparse conversion.')
            if A0 is not None:
                self.A0 = self.Ginv.T.conj() @ sp.csc_array(A0) @ self.Ginv
            if s0 is not None:
                self.dense_s0 = s0
                self.s0 = self.Ginv.T.conj() @ s0
        else:
            if A0 is not None: 
                self.A0 = A0
            if s0 is not None: 
                self.s0 = s0
                self.dense_s0 = None
        
        if c0 is not None: self.c0 = c0

            
    def setup_EM_solver(self, omega=None, Nx=None, Ny=None, Npmlx=None, Npmly=None, dl=None, bloch_x=None, bloch_y=None):
        """
        Setup the FDFD electromagnetic solver with given parameters.
        
        Parameters
        ----------
        omega : complex, optional
            Circular frequency, can be complex to allow for finite bandwidth effects.
        Nx : int, optional
            Number of pixels along the x direction.
        Ny : int, optional
            Number of pixels along the y direction.
        Npmlx : int, optional
            Size of the x direction PML in pixels.
        Npmly : int, optional
            Size of the y direction PML in pixels.
        dl : float, optional
            Finite difference grid pixel size, assumed same for x and y directions.
        bloch_x : float, optional
            x-direction phase shift for periodic boundary conditions. Default: 0.0
        bloch_y : float, optional
            y-direction phase shift for periodic boundary conditions. Default: 0.0
            
        Notes
        -----
        Non-None arguments will define or modify corresponding attributes.
        Creates a TM_FDFD solver instance and stores it in self.EM_solver.
        """
        params = locals()
        params.pop('self')
        for param_name, param_value in params.items():
            if param_value is not None:
                setattr(self, param_name, param_value)

        check_attributes(self, 'Nx', 'Ny', 'Npmlx', 'Npmly', 'dl', 'bloch_x', 'bloch_y')
        self.EM_solver = TM_FDFD(self.omega, self.Nx, self.Ny, self.Npmlx, self.Npmly, self.dl, self.bloch_x, self.bloch_y)
    
    
    def setup_EM_operators(self):
        """
        Setup electromagnetic operators for the design region and background.
        
        Notes
        -----
        This method creates the appropriate operators based on whether sparse or dense
        QCQP formulation is used:
        - For sparse QCQP: Creates Ginv (inverse Green's function) and M operators
        - For dense QCQP: Creates G (Green's function) operator
        
        Requires self.des_mask to be defined.
        
        Raises
        ------
        AttributeError
            If des_mask is not defined.
        """
        check_attributes(self, 'des_mask')
        if self.sparseQCQP:
            self.Ginv, self.M = self.EM_solver.get_GaaInv(self.des_mask, self.chi_background)
        else:
            if self.chi_background is None:
                self.M = self.EM_solver.M0
                self.G = self.EM_solver.get_TM_Gba(self.des_mask, self.des_mask)
            else:
                self.M = self.EM_solver.M0 + self.EM_solver._get_diagM_from_chigrid(self.chi_background)
                Id = np.diag(self.des_mask.astype(complex).flatten())[:,self.des_mask.flatten()]
                self.G = self.omega**2 * np.linalg.solve(self.M.toarray(), Id)[self.des_mask.flatten(),:]
    
    def get_ei(self, ji : np.ndarray = None, update=False):
        """
        Get or compute the incident electromagnetic field.
        
        Parameters
        ----------
        ji : np.ndarray, optional
            Current source for computing incident field. If None, uses self.ji.
        update : bool, optional
            Whether to update self.ei with the computed field. Default: False
            
        Returns
        -------
        ei : np.ndarray
            The incident electromagnetic field. If self.ei exists, returns it directly.
            Otherwise computes it using the EM solver.
        """
        if self.ei is None:
            ei = self.EM_solver.get_TM_field(ji, self.chi_background) if self.ji is None else self.EM_solver.get_TM_field(self.ji, self.chi_background)
        else:
            ei = self.ei
        if update: self.ei = ei
        return ei
    
    def set_ei(self, ei : np.ndarray):
        """
        Set the incident electromagnetic field.
        
        Parameters
        ----------
        ei : np.ndarray
            The incident electromagnetic field to store.
        """
        self.ei = ei 
        
    def setup_QCQP(self, Pdiags: str = "global", verbose: float = 0) -> None:
        """
        Set up the quadratically constrained quadratic programming (QCQP) problem.
        
        Parameters
        ----------
        Pdiags : str or np.ndarray, optional
            Specification for projection matrix diagonals. If "global", creates
            global projectors with ones and -1j entries. Default: "global"
        verbose : float, optional
            Verbosity level for debugging output. Default: 0
            
        Notes
        -----        
        For sparse QCQP, creates SparseSharedProjQCQP with transformed matrices.
        For dense QCQP, creates DenseSharedProjQCQP with original matrices.
        
        Raises
        ------
        AttributeError
            If required attributes (des_mask, A0, s0, c0) are not defined.
        ValueError
            If neither ji nor ei is specified, or if Pdiags specification is invalid.
        """
        check_attributes(self, 'des_mask', 'A0', 's0', 'c0')
        
        # number of field degrees of freedom / pixels in design region
        self.Ndes = int(np.sum(self.des_mask)) 
        
        # generate initial field
        if (self.ji is None) and (self.ei is None):
            raise AttributeError("an initial current ji or field ei must be specified.")
        if not (self.ji is None) and not (self.ei is None):
            warnings.warn("If both ji and ei are specified then ji is ignored.")
        
        self.get_ei(self.ji, update=True)

        if Pdiags=="global":
            # Build projectors as a list of matrices (new API)
            # old behavior: two columns with [1, -1j] on the diagonal
            I = sp.eye_array(self.Ndes, dtype=complex, format="csc")
            self.Plist = [I, (-1j) * I]
        else:
            raise ValueError("Not a valid Pdiags specification / needs implementation")
        
        if self.sparseQCQP: # rewrite later when sparse and dense QCQP classes are unified
            if (self.Ginv is None) or (self.M is None):
                self.setup_EM_operators()
            
            A1_sparse = sp.csc_array(np.conj(1.0/self.chi) * self.Ginv.conj().T - sp.eye(self.Ndes))
            A2_sparse = sp.csc_array(self.Ginv)

            self.QCQP = SparseSharedProjQCQP(self.A0, self.s0, self.c0, 
                                            A1_sparse, A2_sparse, self.ei[self.des_mask]/2, 
                                            self.Plist, verbose=verbose
                                            )
        else:
            if self.G is None: 
                self.setup_EM_operators()
            
            A1_dense = np.conj(1.0/self.chi)*np.eye(self.G.shape[0]) - self.G.conj().T
            print(self.A0.shape, self.s0.shape, self.c0, A1_dense.shape, self.ei[self.des_mask].shape)
            self.QCQP = DenseSharedProjQCQP(self.A0, self.s0, self.c0,
                                            A1_dense, self.ei[self.des_mask]/2,
                                            self.Plist, verbose=verbose
                                            ) # for dense QCQP formulation A2 is not needed 

    def bound_QCQP(self, method : str = 'bfgs', init_lags : np.ndarray = None, opt_params : dict = None):
        """
        Calculate a bound on the QCQP dual problem.
        
        Parameters
        ----------
        method : str, optional
            Optimization method to use. Options: 'bfgs', 'newton'. Default: 'bfgs'
        init_lags : np.ndarray, optional
            Initial Lagrange multipliers for optimization. If None, finds feasible point.
        opt_params : dict, optional
            Additional parameters for the optimization algorithm.
            
        Returns
        -------
        result : tuple
            Result from QCQP dual problem solver containing:
            (dual_value, lagrange_multipliers, gradient, hessian, primal_variable)
        """
        return self.QCQP.solve_current_dual_problem(method = method, init_lags = init_lags, opt_params = opt_params)

    def get_chi_inf(self):
        """
        Get the inferred susceptibility from the QCQP dual solution.
        
        Returns
        -------
        chi_inf : np.ndarray
            The inferred susceptibility χ_inf = P / E_total, where P is the 
            polarization current and E_total is the total electric field.
            
        Notes
        -----
        This represents the material susceptibility that would be required to 
        achieve the optimal field distribution found by the QCQP solver.
        The inferred chi may not be physically feasible for nonzero duality gap.
        
        Raises
        ------
        AssertionError
            If QCQP has not been initialized or solved yet.
        """
        assert hasattr(self, 'QCQP'), "QCQP not initialized. Initialize and solve QCQP first."
        assert self.QCQP.current_xstar is not None, "Inferred chi not available before solving QCQP dual"
        
        if self.sparseQCQP:
            P = self.QCQP.A2 @ self.QCQP.current_xstar  # Calculate polarization current
            Es = self.QCQP.current_xstar
        else:
            P = self.QCQP.current_xstar
            Es = self.G @ P
        
        Etotal = self.get_ei()[self.des_mask] + Es
        return P / Etotal
    
    def _get_dof_chigrid_M_es(self, dof : np.ndarray):
        """
        setup method for structure_objective_sparse and structure_objective_dense
        """
        chigrid_dof = np.zeros((self.Nx,self.Ny), dtype=complex)
        chigrid_dof[self.des_mask] = dof * self.chi
        M_dof = self.M + self.EM_solver._get_diagM_from_chigrid(chigrid_dof)
        es = spla.spsolve(M_dof, self.omega**2 * (chigrid_dof*self.ei).flatten())[self.des_mask.flatten()]
        
        return chigrid_dof, M_dof, es
    
    def structure_objective_sparse(self, dof : np.ndarray, grad : np.ndarray):
        """
        Structural optimization objective and gradient for the specified problem when sparseQCQP=True.
        Follows convention of the optimization package NLOPT: returns objective value
        and stores gradient with respect to objective in the input argument grad.
        
        Parameters
        ----------
        dof : np.ndarray
        Pixel-wise structure degrees of freedom over the design region as specified by self.des_mask. 
        dof[j] is a linear interpolation between dof[j] = 0 (self.chi_background) and dof[j] = 1 (self.chi_background + self.chi)
        
        grad : np.ndarray
        Adjoint gradient of the design objective with respect to dof. 
        Specify grad = [] if only the objective is needed. 
        Otherwise, grad should be an array of the same size as dof; upon method exit grad will store the gradient.
        
        Returns
        -------
        obj : float
        The design objective for the structure specified by dof.
        """
        
        chigrid_dof, M_dof, es = self._get_dof_chigrid_M_es(dof)
        obj = np.real(-np.vdot(es, self.A0@es) + 2*np.vdot(self.s0,es) + self.c0)
        
        if len(grad)>0:
            adj_src = np.zeros((self.Nx,self.Ny), dtype=complex)
            adj_src[self.des_mask] = np.conj(self.s0 - self.A0 @ es)
            adj_v = spla.spsolve(M_dof, adj_src.flatten())[self.des_mask.flatten()]
            grad[:] = 2*np.real(self.omega**2 * self.chi * (adj_v * (self.ei[self.des_mask]+es)))
        
        return obj
    
    def structure_objective_dense(self, dof, grad):
        """
        Structural optimization objective and gradient for the specified problem when sparseQCQP=False.
        Specifications exactly the same as structure_objective_sparse.
        """
        
        chigrid_dof, M_dof, es = self._get_dof_chigrid_M_es(dof)
        
        et = self.ei[self.des_mask] + es
        p = chigrid_dof[self.des_mask] * et
        
        obj = np.real(-np.vdot(p, self.A0@p) + 2*np.vdot(self.s0,p) + self.c0)
        
        if len(grad)>0:
            adj_src = np.zeros((self.Nx,self.Ny), dtype=complex)
            adj_src[self.des_mask] = chigrid_dof[self.des_mask] * np.conj(self.s0 - self.A0 @ p)
            adj_v = spla.spsolve(M_dof, adj_src.flatten())[self.des_mask.flatten()]
            grad[:] = 2*np.real((self.chi*np.conj(self.s0 - self.A0 @ p) + self.omega**2*self.chi*adj_v) * et)
        
        return obj


class Photonics_TE_Yee_FDFD(Photonics_FDFD):
    def __init__(self):
        pass

## Utility functions for photonics problems

def chi_to_feasible_rho(chi_inf, chi_design):
    """
    Project the inferred chi to the feasible set defined by chi_design.
    Resulting chi is chi_design * rho, where rho is in [0, 1].
    
    Parameters
    ----------
    chi_inf : np.ndarray
        Inferred chi from Verlan optimization.
    chi_design : complex
        The design susceptibility of the problem.
    """
    rho = np.real(chi_inf.conj() * chi_design) / np.abs(chi_design)**2
    rho = np.clip(rho, 0, 1)
    return rho
