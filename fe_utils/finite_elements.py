# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    if cell.dim == 1:
        res = np.array([[i/degree] for i in range(degree + 1)])

    elif cell.dim == 2:
        res = np.array([[i/degree, j/degree] for j in range(degree + 1) for i in range(degree + 1 - j)])

    else:
        raise NotImplementedError("dim>2 not implemented")
    return res 


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    m = len(points)
    
    
    if cell.dim == 1:
        if grad:
            V = np.array([[[0] for _ in range(degree+1)] for _ in range(m)])
            if degree >= 1:
                V[:, 1] = np.array([[1] for _ in range(m)])
                for k in range(2, degree + 1):
                    V[:, k] = np.array([[k * point[0]**k] for point in points])
        else:
            V = np.zeros((m, int(degree + 1)))
            V[:, 0] = np.ones((m,))
            if degree >= 1:
                V[:, 1] = np.array([point[0] for point in points]) 
                for k in range(2, degree + 1):
                    V[:, k] = V[:, 1] * V[:, k-1]
    
    elif cell.dim == 2:
        if grad:
            n = (degree + 1) * (degree + 2) // 2
            V = np.array([[[0, 0] for _ in range(n)] for _ in range (m)])
            if degree >= 1:
                V[:, 1] = np.array([[1, 0] for _ in range(m)])
                V[:, 2] = np.array([[0, 1] for _ in range(m)])
                for k in range(2, degree + 1):
                    # x^k column initialisation in V
                    startXk = k * (k + 1) // 2
                    V[:, startXk] = np.array([[k*point[0]**k, 0] for point in points])

                    # y^k column initialisation in V
                    endYk = (k + 1) * (k + 2) // 2 - 1 
                    V[:, endYk] = np.array([[0, k*point[1]**k] for point in points])

                    # we need to fill all columns between startXk and endYk 
                    # with [(k-j) * x^k-j-1 * y^j, j*x^k-j*y^j-1  for j=0 to k
                    for i in range(startXk + 1, endYk):
                        j = i - startXk
                        V[:, i] = np.array([[(k-j) * point[0]**(k-j-1) * point[1]**j, j * point[0]**(k-j) * point[1]**(j-1)] for point in points]) 
        else:
            n = (degree + 1) * (degree + 2) // 2
            V = np.zeros((m, n))
            V[:, 0] = np.ones((m,))
            if degree >= 1:
                V[:, 1] = np.array([point[0] for point in points]) 
                V[:, 2] = np.array([point[1] for point in points]) 
                for k in range(2, degree + 1):
                    # x^k column initialisation in V
                    startXk = k * (k + 1) // 2
                    V[:, startXk] = V[:, 1] * V[:, k*(k-1)//2]

                    # y^k column initialisation in V
                    endYk = (k + 1) * (k + 2) // 2 - 1 
                    V[:, endYk] = V[:, 2] * V[:, startXk-1]

                    # we need to fill all columns between startXk and endYk 
                    # with x^k-j * y^j for j=0 to k
                    for i in range(startXk + 1, endYk):
                        j = i - startXk
                        # column index of x^k-j vector in vandermonde_matrix V
                        xPower_j = (k - j) * (k - j + 1) // 2
                        # column jndex of y^j vector jn vandermonde_matrix V
                        yPower_j = (j + 1) * (j + 2) // 2 - 1

                        V[:, i] = V[:, xPower_j] * V[:, yPower_j]
    else:
        raise NotImplementedError("dim>2 not implemented")

    return V


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(self.cell, 
            self.degree, self.nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        """
        self.basis_coefs -> each colum coefs for monomial basis
        result -> column evaluation of a basis at p points 
        """
        if grad:
            gradVandermonde = vandermonde_matrix(self.cell, self.degree, points, grad=True)
            return  np.einsum("ijk,jl->ilk", gradVandermonde, self.basis_coefs)
        
        vandermondePoints = vandermonde_matrix(self.cell, self.degree, points)
        return vandermondePoints@self.basis_coefs

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        return [fn(x) for x in self.nodes]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """
        
        entity_nodes = self.get_entity_nodes(degree, cell) 
        nodes = lagrange_points(cell, degree)
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes=entity_nodes)

    def get_entity_nodes(self, degree, cell):
        dim = cell.dim
        N = int((degree + 1) * (degree + 2) / 2)
        all_indices = [i for i in range(N)]
        
        if dim == 1:
            entity_nodes = {0: {0: [0],
                                1: [degree]},
                            1: {0:[i for i in range(1,degree)]}}
            return entity_nodes
        
        elif dim == 2:
            v1, v2, v3 = 0, degree, N - 1
        
            entity_nodes = {0: {0: [v1],
                                1: [v2],
                                2: [v3]
                                },
                            1: {0: [],
                                1: [],
                                2: []},
                            2: {0: []}
                                }
            if degree == 1:
                return entity_nodes
            
            indexes_0 = lambda i: int((i + 2) * degree - i * (1 + i) / 2)
            edge_0 = [indexes_0(i) for i in range(degree - 1)]

            indexes_1 = lambda i: int((i + 1) * (1 + degree) - i * (1 + i) / 2)
            edge_1 = [indexes_1(i) for i in range(degree - 1)]

            edge_2 = [i for i in range(1,degree)]
            
            seen = [v1, v2, v3] + edge_0 + edge_1 + edge_2
            
            # what is not on edges or vertex is in interior
            interior = [i for i in all_indices if not i in seen]

            entity_nodes[1] = {0: edge_0,
                                1: edge_1,
                                2: edge_2}
            entity_nodes[2] = {0: interior}

        return entity_nodes
        
