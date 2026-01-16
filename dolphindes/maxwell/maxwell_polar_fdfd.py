"""
2D scalar Helmholtz equation solver in polar coordinates.

Following the finite difference discretization scheme given in
A Note on Finite Difference Discretizations for Poisson Equation on a Disk
by Ming-Chih Lai

Use stretched-coordinate PML as detailed in Choice of the perfectly matched layer
boundary condition for frequency-domain Maxwell's equations solvers
Shin and Fan 2012
"""

__all__ = ["Maxwell_Polar_FDFD", "TM_Polar_FDFD"]

from abc import ABC
from typing import cast

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dolphindes.geometry import PolarFDFDGeometry
from dolphindes.types import (
    BoolGrid,
    ComplexArray,
    ComplexGrid,
    FloatNDArray,
    IntNDArray,
)


class Maxwell_Polar_FDFD(ABC):
    """
    Base class for finite-difference frequency-domain solver in polar coordinates.

    Attributes
    ----------
    omega : complex
        Circular frequency, can be complex to allow for finite bandwidth effects.
    Nr : int
        Number of radial grid points.
    Nphi : int
        Number of azimuthal grid points (in one sector if using symmetry).
    Npml : int
        Number of radial PML layers.
    dr : float
        Radial grid spacing.
    n_sectors : int
        Number of rotational symmetry sectors (1 for full circle).
    bloch_phase : float
        Phase shift for Bloch-periodic boundary conditions in azimuthal direction.
    m : int
        PML polynomial grading order.
    lnR : float
        Logarithm of desired PML reflection coefficient.
    """

    def __init__(
        self,
        omega: complex,
        geometry: PolarFDFDGeometry,
    ) -> None:
        self.omega = omega
        self.geometry = geometry
        self.Nr = geometry.Nr
        self.Nphi = geometry.Nphi
        self.Npml = geometry.Npml
        self.dr = geometry.dr
        self.n_sectors = geometry.n_sectors
        self.bloch_phase = geometry.bloch_phase
        self.m = geometry.m
        self.lnR = geometry.lnR

        self.EPSILON_0 = 1.0
        self.MU_0 = 1.0
        self.C_0 = 1.0
        self.ETA_0 = 1.0
        self.k = self.omega / self.C_0

        self.r_grid: FloatNDArray = (np.arange(self.Nr, dtype=float) + 0.5) * self.dr
        self.dphi = 2 * np.pi / self.n_sectors / self.Nphi
        self.phi_grid: FloatNDArray = cast(
            FloatNDArray,
            np.linspace(0, 2 * np.pi / self.n_sectors, self.Nphi, endpoint=False),
        )

        self.nonpmlNr = self.Nr - self.Npml
        assert self.m > 0, "PML polynomial order m must be positive."
        assert self.nonpmlNr > 0, "Non-PML radial grid size must be positive."


class TM_Polar_FDFD(Maxwell_Polar_FDFD):
    """
    TM polarization FDFD solver in polar coordinates.

    Solves the scalar Helmholtz equation for Ez field.

    Attributes
    ----------
    All attributes from Maxwell_Polar_FDFD, plus:
    M0 : sp.csc_array
        Vacuum Maxwell operator (Laplacian - omega^2).
    """

    def __init__(
        self,
        omega: complex,
        geometry: PolarFDFDGeometry,
    ) -> None:
        super().__init__(omega, geometry)
        self.M0 = self._make_TM_Maxwell_Operator()

    def _make_TM_Maxwell_Operator(self) -> sp.csc_array:
        """
        Assemble the vacuum Maxwell operator in polar coordinates.

        Returns
        -------
        M0 : sp.csc_array
            Maxwell operator: -Laplacian - omega^2 * I
        """
        L = self._get_polar_Laplacian()
        M0 = -L - self.omega**2 * sp.eye_array(self.Nphi * self.Nr, format="csr")
        return sp.csc_array(M0)

    def _get_polar_Laplacian(self) -> sp.csr_array:
        """
        Construct the polar coordinate Laplacian with PML.

        Returns
        -------
        L : sp.csr_array
            Laplacian operator with stretched-coordinate PML.
        """
        r_grid = self.r_grid
        dphi = self.dphi
        dr = self.dr
        omega = self.omega
        m = self.m
        lnR = self.lnR
        Nphi = self.Nphi
        Npml = self.Npml
        bloch_phase = self.bloch_phase

        # Helper grids for PML
        pml_width = dr * Npml
        sigma_max = -(m + 1) * lnR / 2 / pml_width
        l_grid = r_grid - r_grid[-Npml]
        l_grid[l_grid < 0.0] = 0.0  # depth into PML region
        s_grid = 1.0 + 1j * sigma_max * (l_grid / pml_width) ** m / omega
        rcplx_grid = (
            r_grid + 1j * sigma_max * l_grid ** (m + 1) / omega / (m + 1) / pml_width**m
        )
        sprime_grid = 1j * sigma_max * m * l_grid ** (m - 1) / omega / pml_width**m
        sinvsqr_grid = 1.0 / s_grid**2
        g_grid = 1.0 / (rcplx_grid * s_grid) - sprime_grid / s_grid**3

        # Build Laplacian term by term
        # Radial second derivative
        L = (
            sp.kron(
                sp.eye_array(Nphi),
                sp.diags_array(
                    [sinvsqr_grid[1:], -2 * sinvsqr_grid, sinvsqr_grid[:-1]],
                    offsets=[-1, 0, 1],
                ),
                format="csr",
            )
            / dr**2
        )

        # Radial first derivative
        L += (
            sp.kron(
                sp.eye_array(Nphi),
                sp.diags_array([-g_grid[1:], g_grid[:-1]], offsets=[-1, 1]),
                format="csr",
            )
            / 2
            / dr
        )

        # Azimuthal part with symmetry/Bloch boundary conditions
        phase_factor = np.exp(1j * bloch_phase)

        if Nphi == 1:
            az_diags = [np.array([phase_factor + np.conj(phase_factor) - 2.0])]
            az_offsets = [0]
        else:
            az_diags = [
                -2 * np.ones(Nphi, dtype=complex),
                np.ones(Nphi - 1, dtype=complex),
                np.ones(Nphi - 1, dtype=complex),
            ]
            az_offsets = [0, 1, -1]

            if Nphi == 2:
                az_diags[1][0] += np.conj(phase_factor)
                az_diags[2][0] += phase_factor
            else:
                az_diags.append(np.array([np.conj(phase_factor)]))
                az_offsets.append(Nphi - 1)
                az_diags.append(np.array([phase_factor]))
                az_offsets.append(-(Nphi - 1))

        L += (
            sp.kron(
                sp.diags_array(az_diags, offsets=az_offsets, shape=(Nphi, Nphi)),
                sp.diags_array(1.0 / rcplx_grid**2),
                format="csr",
            )
            / dphi**2
        )

        return L

    def _get_diagM_from_chigrid(self, chigrid: ComplexGrid) -> sp.dia_array:
        """
        Get the diagonal contribution to Maxwell operator from susceptibility.

        Parameters
        ----------
        chigrid : ComplexGrid
            Material susceptibility grid.

        Returns
        -------
        diagM : sp.dia_array
            Diagonal matrix -omega^2 * chi.
        """
        return -sp.diags_array(chigrid.flatten() * self.omega**2, format="dia")

    def get_TM_field(
        self, sourcegrid: ComplexGrid, chigrid: ComplexGrid | None = None
    ) -> ComplexArray:
        """
        Solve for the TM field given a source distribution.

        Parameters
        ----------
        sourcegrid : ComplexGrid
            Current source distribution (shape: Nphi * Nr or (Nphi, Nr)).
        chigrid : ComplexGrid, optional
            Material susceptibility distribution. Default is vacuum.

        Returns
        -------
        Ez : ComplexArray
            Electric field solution (flattened).
        """
        M = (
            self.M0 + self._get_diagM_from_chigrid(chigrid)
            if chigrid is not None
            else self.M0
        )
        RHS = 1j * self.omega * np.asarray(sourcegrid).flatten()
        Ez: ComplexArray = spla.spsolve(M, RHS)
        return Ez

    def get_TM_dipole_field(
        self, ir: int, iphi: int, chigrid: ComplexGrid | None = None
    ) -> ComplexArray:
        """
        Get the field from a point dipole source at grid location (ir, iphi).

        Parameters
        ----------
        ir : int
            Radial grid index.
        iphi : int
            Azimuthal grid index.
        chigrid : ComplexGrid, optional
            Material susceptibility. Default is vacuum.

        Returns
        -------
        Ez : ComplexArray
            Electric field solution.
        """
        area = self.geometry.get_pixel_areas()
        idx = iphi * self.Nr + ir
        sourcegrid: ComplexArray = np.zeros(self.Nphi * self.Nr, dtype=complex)
        sourcegrid[idx] = 1.0 / area[idx]
        return self.get_TM_field(sourcegrid, chigrid)

    def get_symmetric_grids(
        self,
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
        """
        Get coordinate grid arrays.

        Returns
        -------
        phi_grid : FloatNDArray
            Azimuthal coordinates for the sector.
        r_grid : FloatNDArray
            Radial coordinates.
        phi_grid_full : FloatNDArray
            Azimuthal coordinates for full circle (for plotting).
        """
        Nphi_full = self.Nphi * self.n_sectors
        phi_grid_full: FloatNDArray = cast(
            FloatNDArray, np.linspace(0, 2 * np.pi, Nphi_full, endpoint=False)
        )
        return self.phi_grid, self.r_grid, phi_grid_full

    def get_TM_Gba(
        self,
        design_mask: BoolGrid,
        observe_mask: BoolGrid,
    ) -> ComplexArray:
        """
        Compute vacuum Green's function from design to observation region.

        Convention: E_obs = (i/ω) * G_ba @ J
        This means it is a propagator (i.e., it already contains its integral).
        Thus, G is scaled so we do not need to modify the field by areas before
        applying it.

        Parameters
        ----------
        design_mask : BoolGrid
            Boolean mask for source/design region (shape Nr x Nphi).
        observe_mask : BoolGrid
            Boolean mask for observation region (shape Nr x Nphi).

        Returns
        -------
        G_ba : ComplexArray
            Green's function matrix (N_obs x N_design).
        """

        def to_sector_lin_idx(mask: BoolGrid) -> IntNDArray:
            if mask.shape == (self.Nr, self.Nphi):
                m = mask
            elif mask.shape == (self.Nphi, self.Nr):
                m = mask.T
            else:
                assert mask.size == self.Nr * self.Nphi, "mask has incompatible size"
                m = mask.reshape((self.Nr, self.Nphi), order="F")
            return np.nonzero(m.flatten(order="F"))[0]

        design_lin = to_sector_lin_idx(design_mask)
        observe_lin = to_sector_lin_idx(observe_mask)

        # Factorize once
        solve = spla.factorized(self.M0.tocsc())

        N_obs = observe_lin.size
        N_des = design_lin.size
        G: ComplexArray = np.zeros((N_obs, N_des), dtype=complex)

        # Build each column by a single RHS solve
        for j, p in enumerate(design_lin):
            b: ComplexArray = np.zeros(self.Nphi * self.Nr, dtype=complex)
            b[p] = 1j * self.omega
            E = solve(b)
            G[:, j] = (-1j * self.omega) * E[observe_lin]

        return G

    def get_GaaInv(
        self, A_mask: BoolGrid, chigrid: ComplexGrid | None = None
    ) -> tuple[sp.csc_array, sp.csc_array]:
        """
        Compute the inverse Green's function on region A, G_{AA}^{-1}.

        Uses the Woodbury identity for block inversion.

        Convention: J_A = GaaInv @ E_A

        Parameters
        ----------
        A_mask : BoolGrid
            Boolean mask for design region A.
        chigrid : ComplexGrid, optional
            Material susceptibility. Default is vacuum.

        Returns
        -------
        GaaInv : sp.csc_array
            Inverse Green's function on region A.
        M : sp.csc_array
            Full Maxwell operator used.
        """
        M = (
            self.M0
            if chigrid is None
            else self.M0 + self._get_diagM_from_chigrid(chigrid)
        )

        flat_A_mask = A_mask.flatten(order="F")
        designInd = np.nonzero(flat_A_mask)[0]
        backgroundInd = np.nonzero(~flat_A_mask)[0]

        A = (M[:, backgroundInd])[backgroundInd, :]
        B = (M[:, designInd])[backgroundInd, :]
        C = (M[:, backgroundInd])[designInd, :]
        D = (M[designInd, :])[:, designInd]

        AinvB = spla.spsolve(A, B)

        Gfac = self.MU_0 / self.k**2
        GaaInv = (D - (C @ AinvB)) * Gfac

        return sp.csc_array(GaaInv), M


# ============================================================================
# Plotting utilities
# ============================================================================


def _as_polar_mesh(
    field: ComplexGrid, phi_grid: FloatNDArray, r_grid: FloatNDArray
) -> ComplexArray:
    """Convert field into a (Nr, Nphi) mesh consistent with solver ordering."""
    Nphi = len(phi_grid)
    Nr = len(r_grid)
    f = np.asarray(field)

    if f.ndim == 1:
        if f.size != Nphi * Nr:
            raise ValueError(f"field has size {f.size}, expected {Nphi * Nr}")
        return f.reshape((Nphi, Nr), order="C").T

    if f.ndim == 2:
        if f.shape == (Nphi, Nr):
            return f.T
        if f.shape == (Nr, Nphi):
            return f
        raise ValueError(
            f"field has shape {f.shape}, expected {(Nphi, Nr)} or {(Nr, Nphi)}"
        )

    raise ValueError(f"field must be 1D or 2D, got ndim={f.ndim}")


def plot_real_polar_field(
    field: ComplexGrid,
    phi_grid: FloatNDArray,
    r_grid: FloatNDArray,
    cmap: str = "Oranges",
    figsize: tuple[float, float] = (5, 5),
    savename: str | None = None,
    colorbar: bool = True,
    axes: bool = True,
    dpi: int = 200,
    show_sector_lines: bool = False,
    n_sectors: int = 1,
    use_log: bool = False,
    log_eps: float = 1e-12,
) -> None:
    """Plot a real-valued polar field."""
    if len(phi_grid) == 1:
        # Handle 1D radial case by expanding to full circle for visualization
        field_mesh = _as_polar_mesh(field, phi_grid, r_grid)
        phi_grid = np.linspace(0, 2 * np.pi, 361, endpoint=True, dtype=float)
        field_mesh = np.tile(field_mesh, (1, len(phi_grid)))
    else:
        field_mesh = _as_polar_mesh(field, phi_grid, r_grid)

    r_mesh, phi_mesh = np.meshgrid(r_grid, phi_grid, indexing="ij")

    norm = None
    if use_log:
        norm = colors.LogNorm(vmin=max(log_eps, np.finfo(float).tiny))

    plt.figure(figsize=figsize)
    ax = plt.subplot(projection="polar")
    p = plt.pcolormesh(phi_mesh, r_mesh, field_mesh, cmap=cmap, norm=norm)
    if not np.allclose(2 * np.pi - phi_grid[-1], phi_grid[1] - phi_grid[0]):
        plt.xlim([phi_grid[0], phi_grid[-1]])

    if colorbar:
        plt.colorbar(p)

    if show_sector_lines and n_sectors > 1:
        sector_angle = 2 * np.pi / n_sectors
        r_max = np.max(r_grid)
        for i in range(n_sectors):
            angle = i * sector_angle
            ax.plot([angle, angle], [0, r_max], "r-", linewidth=0.4, alpha=0.8)

    if not axes:
        plt.axis("off")
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=dpi)
        plt.close()
    else:
        plt.show()


def plot_cplx_polar_field(
    field: ComplexGrid,
    phi_grid: FloatNDArray,
    r_grid: FloatNDArray,
    figsize: tuple[float, float] = (10, 4),
    show_grid: bool = True,
    savename: str | None = None,
    dpi: int = 200,
    show_sector_lines: bool = False,
    n_sectors: int = 1,
    use_log: bool = False,
    linthresh: float = 1e-3,
    linscale: float = 1.0,
) -> None:
    """Plot real and imaginary parts of a complex field side by side."""
    if len(phi_grid) == 1:
        # Handle 1D radial case by expanding to full circle for visualization
        field_mesh = _as_polar_mesh(field, phi_grid, r_grid)
        phi_grid = np.linspace(0, 2 * np.pi, 361, endpoint=True, dtype=float)
        field_mesh = np.tile(field_mesh, (1, len(phi_grid)))
    else:
        field_mesh = _as_polar_mesh(field, phi_grid, r_grid)

    r_mesh, phi_mesh = np.meshgrid(r_grid, phi_grid, indexing="ij")

    norm: colors.Normalize
    if use_log:
        norm = colors.SymLogNorm(linthresh=linthresh, linscale=linscale)
    else:
        norm = colors.CenteredNorm()

    plt.figure(figsize=figsize)
    ax1 = plt.subplot(121, projection="polar")
    ax1.grid(show_grid)
    ax2 = plt.subplot(122, projection="polar")
    ax2.grid(show_grid)

    if not np.allclose(2 * np.pi - phi_grid[-1], phi_grid[1] - phi_grid[0]):
        ax1.set_xlim((phi_grid[0], phi_grid[-1]))
        ax2.set_xlim((phi_grid[0], phi_grid[-1]))

    p1 = ax1.pcolormesh(phi_mesh, r_mesh, np.real(field_mesh), cmap="bwr", norm=norm)
    ax1.set_title("real")

    p2 = ax2.pcolormesh(phi_mesh, r_mesh, np.imag(field_mesh), cmap="bwr", norm=norm)
    ax2.set_title("imag")

    if show_sector_lines and n_sectors > 1:
        sector_angle = 2 * np.pi / n_sectors
        r_max = np.max(r_grid)
        for i in range(n_sectors):
            angle = i * sector_angle
            ax1.plot([angle, angle], [0, r_max], "r-", linewidth=2, alpha=0.8)
            ax2.plot([angle, angle], [0, r_max], "r-", linewidth=2, alpha=0.8)

    plt.colorbar(p1, ax=ax1)
    plt.colorbar(p2, ax=ax2)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=dpi)
        plt.close()
    else:
        plt.show()


def expand_symmetric_field(
    field_symmetric: ComplexGrid, n_sectors: int, Nr: int, m: int = 0
) -> ComplexArray:
    """
    Expand a symmetric field solution to the full circle, applying Bloch phase.

    Parameters
    ----------
    field_symmetric : ComplexGrid
        The field solution for a single sector (flattened).
    n_sectors : int
        Number of rotational symmetry sectors.
    Nr : int
        Number of radial grid points.
    m : int, optional
        Angular momentum number (azimuthal mode index). Default is 0.

    Returns
    -------
    ComplexArray
        The expanded field over the full circle (flattened).
    """
    Nphi_sector = len(field_symmetric) // Nr

    # Reshape to 2D (Nphi_sector, Nr) using the solver's C-order convention
    field_2d = np.asarray(field_symmetric).reshape((Nphi_sector, Nr), order="C")

    # Create list to hold the field for each sector
    sectors_list = []

    # Calculate the phase shift per sector based on angular momentum m
    # Shift = m * (angle of one sector)
    sector_phase_shift = m * (2 * np.pi / n_sectors)

    for k in range(n_sectors):
        # Calculate phase for the k-th sector
        # Sector 0: phase 0
        # Sector 1: phase shift
        # Sector 2: 2 * phase shift...
        phase_factor = np.exp(1j * k * sector_phase_shift)

        # Apply phase and append
        sectors_list.append(field_2d * phase_factor)

    # Stack all sectors vertically (along phi axis)
    field_full_2d = np.vstack(sectors_list)

    return field_full_2d.flatten()
