"""Geometry type for Dolphindes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


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

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Return grid dimensions.

        Returns
        -------
        tuple of int
            Grid dimensions (Nx, Ny).
        """
        return (self.Nx, self.Ny)


@dataclass
class PolarFDFDGeometry(GeometryHyperparameters):
    """
    Polar FDFD geometry specification.

    Attributes
    ----------
    Nphi : int
        Number of azimuthal grid points (in one sector if using symmetry).
    Nr : int
        Number of radial grid points.
    Npml : int
        Number of radial PML layers.
    dr : float
        Radial grid spacing.
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
    n_sectors: int = 1
    bloch_phase: float = 0.0
    m: int = 3
    lnR: float = -20.0

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Return grid dimensions.

        Returns
        -------
        tuple of int
            Grid dimensions (Nr, Nphi).
        """
        return (self.Nr, self.Nphi)
