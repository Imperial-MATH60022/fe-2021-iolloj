'''Test tabulation of basis functions.'''
import pytest
from fe_utils import ReferenceTriangle, ReferenceInterval, LagrangeElement, VectorFiniteElement
import numpy as np
import pdb


@pytest.mark.parametrize('cell, degree',
                         [(c, d)
                          for c in [ReferenceTriangle]
                          for d in range(1, 8)])
def test_tabulate_at_nodes(cell, degree):
    """Check that tabulating at the nodes produces the identity matrix."""
    fe = LagrangeElement(cell, degree)
    ve = VectorFiniteElement(fe)
    a = ve.tabulate(ve.nodes)
    assert (np.round(a[:, 0, :]-np.eye(len(fe.nodes)), 10) == 0).all()
    assert (np.round(a[:, 1, :]-np.eye(len(fe.nodes)), 10) == 0).all()

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
