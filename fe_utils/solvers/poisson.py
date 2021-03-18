"""Solve a model poisson problem with Dirichlet boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from __future__ import division
from fe_utils import *
import numpy as np
from numpy import sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Poisson problem given
    the function space in which to solve and the right hand side
    function."""

    fe = fs.element
    # Create an appropriate (complete) quadrature rule.
    Q = gauss_quadrature(fe.cell, 2*fe.degree)

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi = fe.tabulate(Q.points)
    grad_phi = fe.tabulate(Q.points, grad=True)

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)
    # Now loop over all the cells and assemble A and l
    for c in range(fs.mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes = fs.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = fs.mesh.jacobian(c)
        J_inv_t = np.linalg.inv(J.T)
        det_J = np.abs(np.linalg.det(J))

        f_inner_sum = np.einsum('k, qk->q', f.values[nodes], phi)
        l[nodes] += np.einsum('q, qi, q->i', f_inner_sum, phi, Q.weights) * det_J

        # Sum phi * phi * wq
        phi_phi = np.einsum('qi, qj->ijq', phi, phi)
        phi_sum = np.einsum('ijq, q->ij', phi_phi, Q.weights)
        
        # J @ grad_phi term
        J_grad = np.einsum('ba, qia->biq', J_inv_t, grad_phi)
        # (J @ grad_phi ) (J @ grad_phi) term
        grad_grad = np.einsum('ajq, aiq->ijq', J_grad, J_grad)
        # Sum (J @ grad_phi ) (J @ grad_phi) * wq
        grad_sum = np.einsum('ijq, q->ij', grad_grad, Q.weights)

        A[np.ix_(nodes, nodes)] += (phi_sum + grad_sum) * det_J
    
    boundary = boundary_nodes(fs)
    l[boundary] = 0
    A[boundary] = np.zeros((len(boundary), fs.node_count))
    A[boundary, boundary] = np.ones(len(boundary))
    return A, l 


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_poisson(degree, resolution, analytic=False, return_error=False):
    """Solve a model Poisson problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: sin(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: (16*pi**2*(x[1] - 1)**2*x[1]**2 - 2*(x[1] - 1)**2 -
                             8*(x[1] - 1)*x[1] - 2*x[1]**2) * sin(4*pi*x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Poisson problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_poisson(degree, resolution, analytic, plot_error)

    u.plot()
