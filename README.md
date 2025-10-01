# QNetwork
This repository contains some tools for the calculation of satellite trajectory, atmospheric attenuation and channel transmittance (from SatOptLink repository).
It is currenly developped in [LIP6](www.lip6.fr)’s [QI team](https://qi.lip6.fr) for applications in space quantum networks.
We also used various tools developed in the QuantumCity(https://github.com/RajaYehia/QuantumCity/blob/main) and the balloon_qnet(https://github.com/RajaYehia/balloon_qnet/blob/packaging/) repositories.

## Dependencies

SatOptLink depends on various other libraries:

* Standard scientific computing libraries, like [`numpy`](https://numpy.org/) and [`matplotlib`](https://matplotlib.org/)
* [`orekit`](https://gitlab.orekit.org/orekit-labs/python-wrapper/-/wikis/home) to compute the actual orbits, which was installed in a conda environment using the method found on [this](https://www.orekit.org/download.html) page

Note that we are currently using a python wrapper around `orekit` and its installation is not totally trivial, to say the least.

In addition to these, another crucial component is lowtran(-piccia). Installing lowtran(-piccia) involves several steps, primarily focused on setting up the necessary environment and dependencies:
- Install gfortran, an essential compiler for Fortran programs, with "sudo apt install gfortran".
- Install cmake, a tool for managing the build process of software, using "sudo apt install cmake".
- Ensure Python is installed on your system. If not, install it using sudo apt install python3 python3-pip. Additionally, install NumPy, as it contains f2py, which is crucial for the build process. This can be done with pip3 install numpy.
- Clone the lowtran(-piccia) repository from GitHub using the command git clone https://github.com/francescopiccia/lowtran-piccia.git.
- Install lowtran(-piccia) by navigating into the cloned directory and running "pip3 install ."

For the installation of NetSquid you need to create an account via their website https://netsquid.org/

## Development

### Git branches

Currently, the development is directly done on the `main` branch, until we have a usable tool.

The `orekitdata` branch contains extra orekit files which are sometimes useful. To update this branch, always use `git rebase main` (or equivalent).

The branch `lowtran` depends on the installation of the lowtran package and `gfortran`.


### Credits

SatOptLink is developped by:

* Matteo Schiavon
* Aeden Leal
* Hanafi Issahnane
* Frédéric Grosshans
* Raja Yehia
* Giorgos Papagrigoriou
* Ilektra Karakosta-Amarantidou
