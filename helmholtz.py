# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import dolfinx
import numpy
from mpi4py import MPI
from dolfinx.io import XDMFFile
from fem import assemble_matrix
from petsc4py import PETSc
from scipy.sparse.linalg import spsolve

# Create/read mesh
comm = MPI.COMM_WORLD
mesh = dolfinx.UnitSquareMesh(comm, 200, 200, ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)
mesh.topology.create_connectivity_all()

# find interace faces
tdim = mesh.topology.dim
cfc = mesh.topology.connectivity(tdim - 1, tdim)
boundary_facets = numpy.where(numpy.diff(cfc.offsets)==1)[0]

# dolfinx.mesh.locate_entities_boundary(tdim-1, )

n = ufl.FacetNormal(mesh)
k0 = 10

# Definition of test and trial function spaces
deg = 1  # polynomial degree
V = dolfinx.FunctionSpace(mesh, ("Lagrange", deg))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def plane_wave(x):
    '''Plane Wave Expression'''
    theta = numpy.pi/4
    return numpy.exp(1.0j * k0 * (numpy.cos(theta) * x[0] + numpy.sin(theta) * x[1]))

# Prepare Expression as FE function
ui = dolfinx.Function(V)
ui.interpolate(plane_wave)
g = ufl.dot(ufl.grad(ui), n) + 1j * k0 * ui


a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k0**2 * ufl.inner(u, v) * ufl.dx  \
    + 1j * k0 * ufl.inner(u, v) * ufl.ds
L = ufl.inner(g, v) * ufl.ds

T = 1j * k0 * ufl.inner(u, v) * ufl.ds

# Assemble petsc distribute Matrix
A = dolfinx.fem.assemble_matrix(a)
A.assemble()

# Assemble scipy matrix on csr format
active_entities = {"facets": boundary_facets}
Aij = assemble_matrix(T, active_entities)


indices = V.dofmap.index_map.indices(True).astype(numpy.int32)
is_A = PETSc.IS().createGeneral(indices)
lA = A.createSubMatrices(is_A)[0]

print(lA.local_size, Aij.shape)


