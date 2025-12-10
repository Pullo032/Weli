"""
Tests pour les couches de Weli
"""
import numpy as np
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weli.layers import Dense, ReLU, Sigmoid

def test_dense_layer():
    """Test de la couche Dense"""
    layer = Dense(units=10, input_dim=5)
    layer.initialize((5,))
    
    # Test forward
    x = np.random.randn(32, 5)
    output = layer.forward(x)
    
    assert output.shape == (32, 10), f"Expected shape (32, 10), got {output.shape}"
    print("✓ Test Dense layer passed")

def test_relu_activation():
    """Test de l'activation ReLU"""
    layer = ReLU()
    layer.initialize((10,))
    
    x = np.array([[-1, 0, 1, 2, -3]])
    output = layer.forward(x)
    
    expected = np.array([[0, 0, 1, 2, 0]])
    assert np.allclose(output, expected), f"ReLU failed: {output} != {expected}"
    print("✓ Test ReLU activation passed")

if __name__ == '__main__':
    print("Running layer tests...")
    test_dense_layer()
    test_relu_activation()
    print("\n✅ All tests passed!")

