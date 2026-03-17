import os
import torch
import numpy as np

from rover_navigation.perception.dataset import _normalize_points, _normalize_features, _random_sample_indices, _random_sample_indices, _knn_indices

def test_normalize_points_centers_and_scales():
     points = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ], dtype=np.float32)
     out = _normalize_points(points)
     assert out.shape == points.shape
     assert np.allclose(np.mean(out, axis=0), 0.0, atol=1e-6)
     
     norms = np.linalg.norm(out, axis=1)
     assert np.isclose(np.max(norms), 1.0, atol=1e-6)


def test_normalize_points_identical_points():
     points = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float32)
     
     out = _normalize_points(points)

     assert out.shape == points.shape
     assert np.allclose(out, 0.0)

def test_normalize_features_standardize_columns():
     features = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ], dtype=np.float32)
     
     out = _normalize_features(features)

     assert out.shape == features.shape
     assert np.allclose(np.mean(out, axis=0), 0.0, atol=1e-6)
     assert np.allclose(np.std(out, axis=0),1.0,atol=1e-6)

def test_normalize_features_constant_column():
    features = np.array([
    [7.0, 1.0],
    [7.0, 2.0],
    [7.0, 3.0],
], dtype=np.float32)
    
    out = _normalize_features(features)

    assert np.allclose(out[:,0],0.0)

def test_random_sample_indices_without_replacement():
    idx = _random_sample_indices(10,5)

    assert len(idx) == 5
    assert len(np.unique(idx)) == 5
    assert np.all(idx >= 0)
    assert np.all(idx < 10)        

def test_random_sample_indices_with_replacement():
    idx = _random_sample_indices(3,5)

    assert len(idx) ==5
    assert set([0,1,2]).issubset(set(idx))
    assert np.all(idx >= 0)
    assert np.all(idx < 3)

def test_knn_indices_shape_and_bounds():
     support = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
     
     query = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.9, 0.0],
    ], dtype=np.float32)
     
     idx = _knn_indices(query, support, 2)

     assert idx.shape == (2,2)
     assert np.all(idx >= 0)
     assert np.all(idx < len(support))




    
     
