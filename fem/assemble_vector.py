# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from petsc4py import PETSc
import dolfinx
import numpy
import ufl

from odd.communication import IndexMap
from odd import DistArray

def assemble_vector(form: ufl.Form, dtype=numpy.complex128) -> numpy.ndarray:
    """
    Create and assemble vector given a rank 1 ufl form
    """
    _L = dolfinx.Form(form)._cpp_object
    if _L.rank != 1:
        raise ValueError

    dofmap = _L.function_spaces[0].dofmap

    b = dolfinx.fem.assemble_vector(form)
    vec_size = dofmap.index_map.size_local + dofmap.index_map.num_ghosts

    np_b = numpy.zeros(vec_size, dtype)
    with b.localForm() as b_local:
        if not b_local:
            np_b[:] = b.array
        else:
            np_b[:] = b_local.array

    comm = _L.mesh.mpi_comm()
    owned_size = dofmap.index_map.size_local
    ghosts = dofmap.index_map.ghosts
    ghosts = numpy.sort(dofmap.index_map.ghosts)
    # ghost_owners = dofmap.index_map.ghost_owner_rank()
    imap = IndexMap(comm, owned_size, ghosts)
    shape = dofmap.index_map.size_global
    b = DistArray(shape, dtype=np_b.dtype, buffer=np_b, index_map=imap, comm=comm)

    return b
