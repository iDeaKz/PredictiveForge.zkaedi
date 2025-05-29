import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pytest
import tempfile
import asyncio
from optimized_system import (
    adaptive_parallel,
    polymorphic_compute,
    nanomorphic_add,
    transversal_async_read_write,
    memory_database,
    recursive_matrix_multiply,
    singular_value_decomposition,
    convex_hull,
    logistic_regression_predict,
    chi_square_test,
)

def test_adaptive_parallel_large_input():
    # Stress with a very large list
    data = list(range(10**6))
    result = adaptive_parallel(data[:10000])  # Limit to 10k for practical runtime
    assert len(result) == 10000

def test_polymorphic_compute_large_tuple():
    # Stress LRU cache with many unique tuples
    for i in range(1000):
        res = polymorphic_compute(tuple(range(i, i+100)))
        assert isinstance(res, list)

def test_nanomorphic_add_large_numbers():
    # Test with very large integers
    a, b = 10**12, 10**12
    result = nanomorphic_add(a, b)
    assert isinstance(result, int)

@pytest.mark.asyncio
async def test_transversal_async_read_write_large_file():
    # Write a large file asynchronously
    data = os.urandom(2 * 1024 * 1024)  # 2MB
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        await transversal_async_read_write(tmp.name, data)
        assert os.path.getsize(tmp.name) == len(data)
    os.remove(tmp.name)

def test_memory_database_many_rows():
    # Insert many rows into the in-memory DB
    db = memory_database()
    for i in range(10000):
        db.execute("INSERT INTO data (value) VALUES (?)", (str(i),))
    db.commit()
    count = db.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    assert count == 10000

def test_recursive_matrix_multiply_large_square():
    # Test with a large (power of 2) matrix
    size = 8
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    result = recursive_matrix_multiply(A, B)
    assert result.shape == (size, size)

def test_singular_value_decomposition_large_matrix():
    # SVD on a large matrix
    matrix = np.random.rand(100, 100)
    U, S, VT = singular_value_decomposition(matrix)
    assert U.shape == (100, 100)
    assert S.shape == (100, 100)
    assert VT.shape == (100, 100)

def test_convex_hull_many_points():
    # Convex hull with many points
    points = np.random.rand(1000, 2)
    hull = convex_hull(points)
    assert hasattr(hull, 'vertices')

def test_logistic_regression_predict_large():
    # Logistic regression with large dataset
    X_train = np.random.rand(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.rand(100, 10)
    preds = logistic_regression_predict(X_train, y_train, X_test)
    assert preds.shape == (100,)

def test_chi_square_test_large_table():
    # Chi-square test with a large contingency table
    observed = np.random.randint(1, 100, (20, 20))
    chi2, p, dof, expected = chi_square_test(observed)
    assert expected.shape == (20, 20)