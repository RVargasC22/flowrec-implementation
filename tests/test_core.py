import unittest
import numpy as np
import networkx as nx
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowrec_core import HierarchicalNetwork, FlowRec

class TestFlowRecCore(unittest.TestCase):
    def setUp(self):
        # Create a simple hierarchy: Total -> (A, B)
        # We'll use a known structure: 0->1, 0->2.
        # 0=Total, 1=A, 2=B.
        self.edges_idx = [(0, 1), (0, 2)]
        self.labels_idx = {0: 'Total', 1: 'A', 2: 'B'}
        
        self.network = HierarchicalNetwork.from_custom_graph(self.edges_idx, self.labels_idx)
        self.reconciler = FlowRec(self.network)

    def test_network_structure(self):
        """Test if network structure is correctly parsed"""
        self.assertEqual(self.network.n_total, 3)
        self.assertEqual(self.network.n_base, 2)
        
        node_names = self.network.get_node_names()
        # get_node_names returns a list of strings
        self.assertIn('Total', node_names)

    def test_reconciliation_coherence(self):
        """Test if reconciled forecasts are coherent (Parent = Sum(Children))"""
        # Base forecasts (Incoherent): Total=100, A=40, B=50 (Sum=90 != 100)
        # y_hat vector [Total, A, B] (assuming 0,1,2 order)
        y_hat = np.array([100.0, 40.0, 50.0])
        
        # Reconcile returns (y_tilde, elapsed)
        result = self.reconciler.reconcile(y_hat)
        
        if isinstance(result, tuple):
             y_tilde = result[0]
        else:
             y_tilde = result
             
        # Check coherence: Total (idx 0) should equal A (idx 1) + B (idx 2)
        total = y_tilde[0]
        a = y_tilde[1]
        b = y_tilde[2]
        
        # Allow small floating point error
        self.assertAlmostEqual(total, a + b, places=5)

    def test_matrix_projection_shape(self):
        """Test if Projection Matrix P has correct shape"""
        # P should be n x n
        # Property is P_proj based on code inspection
        self.assertEqual(self.reconciler.P_proj.shape, (3, 3))

if __name__ == '__main__':
    unittest.main()
