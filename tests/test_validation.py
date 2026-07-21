"""Tests for runtime input validation at public entry points (issue #28)."""

import numpy as np
import pytest

from dolphindes.geometry import CartesianFDFDGeometry, PolarFDFDGeometry
from dolphindes.maxwell import TM_FDFD
from dolphindes.photonics import Photonics_TM_FDFD
from dolphindes.util.validation import (
    validate_bool_mask,
    validate_numeric_array,
)


class TestArrayValidators:
    """Unit tests for the array validators."""

    def test_validate_bool_mask_accepts_bool(self):
        mask = np.zeros((3, 4), dtype=bool)
        out = validate_bool_mask(mask, "mask", shape=(3, 4))
        assert out is mask

    def test_validate_bool_mask_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            validate_bool_mask([[True, False]], "mask")

    def test_validate_bool_mask_rejects_float_dtype(self):
        mask = np.zeros((3, 4), dtype=float)
        with pytest.raises(ValueError, match="boolean dtype"):
            validate_bool_mask(mask, "mask")

    def test_validate_bool_mask_rejects_wrong_shape(self):
        mask = np.zeros((3, 4), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            validate_bool_mask(mask, "mask", shape=(4, 3))

    def test_validate_bool_mask_rejects_wrong_size(self):
        mask = np.zeros((3, 4), dtype=bool)
        with pytest.raises(ValueError, match="elements"):
            validate_bool_mask(mask, "mask", size=10)

    def test_validate_numeric_array_accepts_numeric(self):
        for dtype in (np.int64, np.float64, np.complex128):
            validate_numeric_array(np.ones(5, dtype=dtype), "arr", size=5)

    def test_validate_numeric_array_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            validate_numeric_array([1, 2, 3], "arr")

    def test_validate_numeric_array_rejects_bool_and_object(self):
        with pytest.raises(ValueError, match="numeric"):
            validate_numeric_array(np.ones(3, dtype=bool), "arr")
        with pytest.raises(ValueError, match="numeric"):
            validate_numeric_array(np.array([object()]), "arr")

    def test_validate_numeric_array_rejects_wrong_size(self):
        with pytest.raises(ValueError, match="elements"):
            validate_numeric_array(np.ones(3, dtype=complex), "arr", size=5)


class TestGeometryValidation:
    """Geometry dataclasses validate their invariants at construction."""

    def test_cartesian_valid(self):
        CartesianFDFDGeometry(Nx=20, Ny=20, Npmlx=4, Npmly=4, dx=0.1, dy=0.1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"Nx": 0},
            {"Ny": -5},
            {"dx": 0.0},
            {"dy": -0.1},
            {"Npmlx": -1},
        ],
    )
    def test_cartesian_invalid_values(self, kwargs):
        base = dict(Nx=20, Ny=20, Npmlx=4, Npmly=4, dx=0.1, dy=0.1)
        base.update(kwargs)
        with pytest.raises((ValueError, TypeError)):
            CartesianFDFDGeometry(**base)

    def test_cartesian_pml_too_large(self):
        with pytest.raises(ValueError, match="PML regions must fit"):
            CartesianFDFDGeometry(Nx=10, Ny=20, Npmlx=5, Npmly=4, dx=0.1, dy=0.1)

    def test_polar_valid(self):
        PolarFDFDGeometry(Nphi=40, Nr=30, Npml=5, dr=0.05)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"Nr": 0},
            {"Nphi": -1},
            {"dr": 0.0},
            {"n_sectors": 0},
            {"Npml": -1},
        ],
    )
    def test_polar_invalid_values(self, kwargs):
        base = dict(Nphi=40, Nr=30, Npml=5, dr=0.05)
        base.update(kwargs)
        with pytest.raises((ValueError, TypeError)):
            PolarFDFDGeometry(**base)

    def test_polar_pml_too_large(self):
        with pytest.raises(ValueError, match="PML regions must fit"):
            PolarFDFDGeometry(Nphi=40, Nr=10, Npml=8, dr=0.05, Npml_inner=3)


class TestMaxwellMaskValidation:
    """The mask-consuming Maxwell methods reject bad masks with clear errors."""

    @pytest.fixture
    def solver(self):
        geometry = CartesianFDFDGeometry(
            Nx=20, Ny=20, Npmlx=4, Npmly=4, dx=0.1, dy=0.1
        )
        return TM_FDFD(2 * np.pi, geometry)

    def test_get_TM_Gba_float_mask_raises_valueerror(self, solver):
        # Regression for issue #28: a float mask previously raised a cryptic
        # ``IndexError: arrays used as indices must be ...`` deep in indexing.
        float_mask = np.zeros((20, 20), dtype=float)
        with pytest.raises(ValueError, match="boolean dtype"):
            solver.get_TM_Gba(float_mask, float_mask)

    def test_get_TM_Gba_list_raises_typeerror(self, solver):
        mask = np.zeros((20, 20), dtype=bool)
        with pytest.raises(TypeError):
            solver.get_TM_Gba(mask.tolist(), mask)

    def test_get_TM_Gba_wrong_shape_raises(self, solver):
        mask = np.zeros((20, 20), dtype=bool)
        bad = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            solver.get_TM_Gba(bad, mask)

    def test_get_GaaInv_float_mask_raises(self, solver):
        float_mask = np.zeros((20, 20), dtype=float)
        with pytest.raises(ValueError, match="boolean dtype"):
            solver.get_GaaInv(float_mask)


class TestPhotonicsConstructorValidation:
    """Photonics_TM_FDFD validates array inputs at construction."""

    @pytest.fixture
    def geometry(self):
        return CartesianFDFDGeometry(
            Nx=20, Ny=20, Npmlx=4, Npmly=4, dx=0.1, dy=0.1
        )

    def test_float_des_mask_raises(self, geometry):
        bad_mask = np.zeros((20, 20), dtype=float)
        with pytest.raises(ValueError, match="boolean dtype"):
            Photonics_TM_FDFD(
                omega=2 * np.pi, geometry=geometry, chi=3 + 0.1j, des_mask=bad_mask
            )

    def test_wrong_shape_des_mask_raises(self, geometry):
        bad_mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            Photonics_TM_FDFD(
                omega=2 * np.pi, geometry=geometry, chi=3 + 0.1j, des_mask=bad_mask
            )

    def test_wrong_size_source_raises(self, geometry):
        mask = np.zeros((20, 20), dtype=bool)
        bad_ji = np.zeros(17, dtype=complex)
        with pytest.raises(ValueError, match="elements"):
            Photonics_TM_FDFD(
                omega=2 * np.pi,
                geometry=geometry,
                chi=3 + 0.1j,
                des_mask=mask,
                ji=bad_ji,
            )
