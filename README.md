# Dolphindes 🐬

[![CI: Quality](https://github.com/physical-design-bounds/dolphindes/actions/workflows/quality.yml/badge.svg)](https://github.com/physical-design-bounds/dolphindes/actions/workflows/quality.yml)
[![CI: Tests](https://github.com/physical-design-bounds/dolphindes/actions/workflows/tests.yml/badge.svg)](https://github.com/physical-design-bounds/dolphindes/actions/workflows/tests.yml)
[![Daily Full Tests](https://github.com/physical-design-bounds/dolphindes/actions/workflows/daily_checks.yml/badge.svg)](https://github.com/physical-design-bounds/dolphindes/actions/workflows/daily_checks.yml)
[![codecov](https://codecov.io/github/physical-design-bounds/dolphindes/graph/badge.svg?token=3C7DSQLXIS)](https://codecov.io/github/physical-design-bounds/dolphindes)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Dolphindes (very loosely, Dual Optimization Limits for PHotonic/PHysical INverse DESign) is a Python package for calculating limits on the performance of photonic devices using dual optimization methods. It can calculate structure-agnostic performance bounds for a wide range of photonic problems. The package works by relaxing the photonic inverse design problem into a field optimization problem, which can then be further relaxed into a convex problem using Lagrange duality. 

## 📦 Installation 

### 1. Install System Dependencies

Make sure you have `libsuitesparse-dev` installed. This is required by scikit-sparse 

For Debian/Ubuntu systems:

```bash
sudo apt-get update
sudo apt-get install libsuitesparse-dev
``` 

### 2. Clone this repo and activate the provided conda environment dolphindes.yml

### 3. If using your own environment, instead run

```bash
pip install .
```

## 🔧 Running Tests

To run the dolphindes tests, simply run 

```bash
pytest
```

Optionally, provide the -s flag to print the output of the tests. You will need to have `pytest` and `pytest-dependency` installed in your environment. 

## 📚 Documentation and Tutorials

Documentation may be found at [dolphindes.readthedocs.io](https://dolphindes.readthedocs.io)

## Citations

If you use dolphindes in your work, please cite the following paper:

```
[Review article coming soon]
```

Additionally, if you use dolphindes to do Verlan design, you should cite the initial Verlan papers:

```
@article{chao_amaolo_blueprints_2025,
      title={Bounds as blueprints: towards optimal and accelerated photonic inverse design}, 
      author={Pengning Chao and Alessio Amaolo and Sean Molesky and Alejandro W. Rodriguez},
      year={2025},
      eprint={2504.10469},
      archivePrefix={arXiv},
      primaryClass={physics.optics},
      url={https://arxiv.org/abs/2504.10469}, 
}

@article{molesky_verlan_2025,
    title = {Inferring {{Structure}} via {{Duality}} for {{Photonic Inverse Design}}},
    author = {Molesky, Sean and Chao, Pengning and Amaolo, Alessio and Rodriguez, Alejandro W.},
    year = {2025},
    month = apr,
    number = {arXiv:2504.14083},
    eprint = {2504.14083},
    primaryclass = {math},
    publisher = {arXiv},
    doi = {10.48550/arXiv.2504.14083},
    archiveprefix = {arXiv}
}
```


