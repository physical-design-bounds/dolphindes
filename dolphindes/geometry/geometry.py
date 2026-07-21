"""Geometry type for Dolphindes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, cast

import numpy as np

from dolphindes.types import FloatNDArray


@dataclass
class GeometryHyperparameters(ABC):
    """Base class for geometry specifications."""

    @abstractmethod
    def get_grid_size(self) -> Tuple[int, int]:
        """
        Return (Nx, Ny) or equivalent grid dimensions.

        Returns
        -------
        tuple of int
            Grid dimensions (Nx, Ny).
        """
        pass

    @abstractmethod
    def get_pixel_areas(self) -> FloatNDArray:
        """
        Return the area of each pixel in the grid.

        Returns
        -------
        ndarray of float
            Flattened array of pixel areas.
        """
        pass


@dataclass
class CartesianFDFDGeometry(GeometryHyperparameters):
    """
    Cartesian FDFD geometry specification.

    Attributes
    ----------
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
    bloch_x : float
        x-direction phase shift associated with the periodic boundary conditions.
        Default: 0.0
    bloch_y : float
        y-direction phase shift associated with the periodic boundary conditions.
        Default: 0.0
    """

    Nx: int
    Ny: int
    Npmlx: int
    Npmly: int
    dx: float
    dy: float
    bloch_x: float = 0.0
    bloch_y: float = 0.0

    def __post_init__(self) -> None:
        """Validate grid dimensions and spacings."""
        if self.Nx <= 0 or self.Ny <= 0:
            raise ValueError(
                f"Nx and Ny must be positive, got Nx={self.Nx}, Ny={self.Ny}."
            )
        if self.Npmlx < 0 or self.Npmly < 0:
            raise ValueError(
                f"Npmlx and Npmly must be non-negative, got "
                f"Npmlx={self.Npmlx}, Npmly={self.Npmly}."
            )
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError(
                f"dx and dy must be positive, got dx={self.dx}, dy={self.dy}."
            )
        if 2 * self.Npmlx >= self.Nx or 2 * self.Npmly >= self.Ny:
            raise ValueError(
                "PML regions must fit inside the grid: need 2*Npmlx < Nx and "
                f"2*Npmly < Ny, got Npmlx={self.Npmlx}, Nx={self.Nx}, "
                f"Npmly={self.Npmly}, Ny={self.Ny}."
            )

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Return grid dimensions.

        Returns
        -------
        tuple of int
            Grid dimensions (Nx, Ny).
        """
        return (self.Nx, self.Ny)

    def get_pixel_areas(self) -> FloatNDArray:
        """
        Return the area of each pixel in the grid.

        Returns
        -------
        ndarray of float
            Flattened array of pixel areas (all equal to dx * dy).
        """
        return np.full(self.Nx * self.Ny, self.dx * self.dy, dtype=float)


@dataclass
class PolarFDFDGeometry(GeometryHyperparameters):
    """
    Polar FDFD geometry specification.

    Attributes
    ----------
    Nphi : int
        Number of azimuthal grid points (in one irreducible region if using symmetry).
    Nr : int
        Number of radial grid points.
    Npml : int
        Width of outer radial PML layers in pixels.
    dr : float
        Radial grid spacing.
    Npml_inner : int
        Width of inner radial PML layers in pixels.
        Default: 0
    r_inner : float
        Inner radius of computational domain.
        Default: 0
    mirror : bool
        Whether reflection symmetry is assumed with Neumann boundary conditions.
        Default: False
    n_sectors : int
        Number of rotational symmetry sectors (1 for full circle).
        Default: 1
    bloch_phase : float
        Phase shift for Bloch-periodic boundary conditions in azimuthal direction.
        Default: 0.0
    m : int
        PML polynomial grading order.
        Default: 3
    lnR : float
        Logarithm of desired PML reflection coefficient.
        Default: -20.0
    """

    Nphi: int
    Nr: int
    Npml: int
    dr: float
    Npml_inner: int = 0
    r_inner: float = 0.0
    mirror: bool = False
    n_sectors: int = 1
    bloch_phase: float = 0.0
    m: int = 3
    lnR: float = -20.0

    def __post_init__(self) -> None:
        """Validate grid dimensions and spacings."""
        if self.Nphi <= 0 or self.Nr <= 0:
            raise ValueError(
                f"Nphi and Nr must be positive, got Nphi={self.Nphi}, Nr={self.Nr}."
            )
        if self.Npml < 0 or self.Npml_inner < 0:
            raise ValueError(
                f"Npml and Npml_inner must be non-negative, got "
                f"Npml={self.Npml}, Npml_inner={self.Npml_inner}."
            )
        if self.dr <= 0:
            raise ValueError(f"dr must be positive, got dr={self.dr}.")
        if self.r_inner < 0:
            raise ValueError(
                f"r_inner must be non-negative, got r_inner={self.r_inner}."
            )
        if self.n_sectors <= 0:
            raise ValueError(
                f"n_sectors must be positive, got n_sectors={self.n_sectors}."
            )
        if self.Npml + self.Npml_inner >= self.Nr:
            raise ValueError(
                "PML regions must fit inside the radial grid: need "
                f"Npml + Npml_inner < Nr, got Npml={self.Npml}, "
                f"Npml_inner={self.Npml_inner}, Nr={self.Nr}."
            )

    @property
    def r_grid(self) -> FloatNDArray:
        """Get the radial grid points."""
        return cast(FloatNDArray, self.r_inner + (np.arange(self.Nr) + 0.5) * self.dr)

    @property
    def nonpmlNr(self) -> int:
        """Get the number of radial grid points excluding PML regions."""
        return self.Nr - self.Npml - self.Npml_inner

    @property
    def dphi(self) -> float:
        """Get the azimuthal grid spacing."""
        val = 2 * np.pi / self.n_sectors / self.Nphi
        if self.mirror:
            val /= 2
        return val

    @property
    def phi_grid(self) -> FloatNDArray:
        """Get the azimuthal grid points."""
        return cast(FloatNDArray, np.arange(self.Nphi) * self.dphi)

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Return grid dimensions.

        Returns
        -------
        tuple of int
            Grid dimensions (Nr, Nphi).
        """
        return (self.Nr, self.Nphi)

    def get_pixel_areas(self) -> FloatNDArray:
        """
        Return the area of each pixel in the grid.

        Returns
        -------
        ndarray of float
            Flattened array of pixel areas.
        """
        area_r = self.r_grid * self.dr * self.dphi
        return cast(FloatNDArray, np.kron(np.ones(self.Nphi), area_r))
