"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
from argparse import ArgumentParser
import numpy as np
from numpy import cos, sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser
import pdb
import matplotlib.pylab as plt


def assemble(fs1, fs2, f):
    """Assemble the finite element system for the Stokes problem given
    the mixed function space in which to solve and the right hand side
    function."""
    
    fe1 = fs1.element
    fe2 = fs2.element

    # Create an appropriate (complete) quadrature rule.
    Q = gauss_quadrature(fe1.cell, 2*max(fe1.degree, fe2.degree))

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi1 = fe1.tabulate(Q.points)
    grad_phi1 = fe1.tabulate(Q.points, grad=True)
    phi2 = fe2.tabulate(Q.points)
    grad_phi2 = fe2.tabulate(Q.points, grad=True)

    # Create the left hand side matrix and right hand side vector.
    A = sp.lil_matrix((fs1.node_count, fs1.node_count))
    B = sp.lil_matrix((fs2.node_count, fs1.node_count))
    l = np.zeros(fs1.node_count + fs2.node_count)


    # Now loop over all the cells and assemble A and l
    for c in range(fs1.mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes1 = fs1.cell_nodes[c, :]
        nodes2 = fs2.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = fs1.mesh.jacobian(c)
        J_inv_t = np.linalg.inv(J.T)
        det_J = np.abs(np.linalg.det(J))

        f_inner_sum = np.einsum('i, qdi->qd', f.values[nodes1], phi1)
        l[nodes1] += np.einsum('qd, qdi, q->i', f_inner_sum, phi1, Q.weights, optimize=True) * det_J
        
        grad_X = np.einsum('ba, qdia->qdib', J_inv_t, grad_phi1)
        grad_X_T = np.einsum('qdib->qbid', grad_X)
        eps = (grad_X + grad_X_T) / 2
        A[np.ix_(nodes1, nodes1)] +=  np.einsum('qlih, qljh, q->ij', eps, eps, Q.weights) * det_J


        div = np.einsum('qdid->qi', grad_X)
        B[np.ix_(nodes2, nodes1)] += np.einsum("qi, qj, q ->ij", phi2, div, Q.weights)* det_J

    return A, B, l

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

    def on_boundary_2d(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return [1., 1.]
        else:
            return [0., 0.]

    if isinstance(fs.element, VectorFiniteElement):
        f.interpolate(on_boundary_2d)
    else:
        f.interpolate(on_boundary)
    return np.flatnonzero(f.values)


def vector_errornorm(f1, f2):
    """Calculate the L^2 norm of the difference between f1 and f2 vector valued function spaces."""

    fs1 = f1.function_space
    fs2 = f2.function_space

    fe1 = fs1.element
    fe2 = fs2.element
    mesh = fs1.mesh

    # Create a quadrature rule which is exact for (f1-f2)**2.
    Q = gauss_quadrature(fe1.cell, 2*max(fe1.degree, fe2.degree))

    # Evaluate the local basis functions at the quadrature points.
    phi1 = fe1.tabulate(Q.points)
    phi2 = fe2.tabulate(Q.points)

    norm = 0.
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes1 = fs1.cell_nodes[c, :]
        nodes2 = fs2.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        det_J = np.abs(np.linalg.det(J))
        
        # Compute the actual cell quadrature.
        f1_onbasis = np.einsum('i,qdi->qd', f1.values[nodes1], phi1)
        f2_onbasis = np.einsum('i,qdi->qd', f2.values[nodes2], phi2)
        err = f1_onbasis - f2_onbasis
        norm += np.einsum('qd, qd, q->', err, err, Q.weights) * det_J

    return norm**0.5


def solve_mastery(resolution, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given resolution. It
    should return both the solution :class:`~fe_utils.function_spaces.Function` and
    the :math:`L^2` error in the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, 2)
    ve = VectorFiniteElement(fe)
    V = FunctionSpace(mesh, ve)
    qe = LagrangeElement(mesh.cell, 1)
    Q = FunctionSpace(mesh, qe)
    
    if analytic:
        return (u_analytic, p_analytic), 0.0


    u_analytic= Function(V)
    u_analytic.interpolate(lambda x: (2 * pi * (1 - cos(2 * pi * x[0])) * sin(2 * pi * x[1]),
                                            -2 * pi * (1 - cos(2 * pi * x[1])) * sin(2 * pi * x[0])))
    p_analytic = Function(Q)
    p_analytic.interpolate(lambda x: 0)

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(V)
    u1 = lambda x: -4* pi**3 * (cos(2 * pi * x[0]) * sin(2 * pi * x[1]) + sin(2 * pi * x[1]) * (cos(2 * pi * x[0]) - 1))
    p1 = lambda x: 4 * pi**3 * (cos(2 * pi * x[1]) * sin(2 * pi * x[0]) + sin(2 * pi * x[0]) * (cos(2 * pi * x[1]) - 1))
    f.interpolate(lambda x: (u1(x), p1(x)))

    # Assemble the finite element system.
    A,B,l = assemble(V,Q,f)
    
    # Create the function to hold the solution.
    u = Function(V)
    p = Function(Q)
    M = sp.bmat([[A, B.T],[B, None]], format='lil')
    # plt.spy(M)
    # plt.show()

    boundary = boundary_nodes(V)
    # arbitrary node not on boundary
    boundary = np.hstack([boundary, [V.node_count +  10]])
    M[boundary] = np.zeros((len(boundary), V.node_count + Q.node_count))
    M[boundary, boundary] = np.ones(len(boundary))
    l[boundary] = 0


    # Cast the matrix to a sparse format and use a superLU solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    M = sp.csc_matrix(M)
    lu = sp.linalg.splu(M)
    res = lu.solve(l)
    u.values[:] = res[:V.node_count]
    p.values[:] = res[V.node_count:]

    # Compute the L^2 error in the solution for testing purposes.
    u_error = vector_errornorm(u_analytic, u)
    p_error = errornorm(p_analytic, p)
    error = (u_error**2 + p_error**2)**0.5

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return (u,p), error


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve the mastery problem.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_mastery(resolution, analytic, plot_error)

    u.plot()
