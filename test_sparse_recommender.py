import pytest
import numpy as np

from sparse_recommender import SparseMatrix

def test_set_get():
    sparse_mx = SparseMatrix()
    with pytest.raises(ValueError):
        sparse_mx.get(0, 0)
        
    sparse_mx.set(0,0,6)
    assert sparse_mx.get(0,0) == 6
    with pytest.raises(KeyError):
        sparse_mx.get(20, 1) # raise KeyError if invalid row, column is given
    with pytest.raises(KeyError):
        sparse_mx.get(-23, 1)
    with pytest.raises(ValueError):
        sparse_mx.set(-23, 1, 5) # raise KeyError if trying to enter value at invalid row, column
    

def test_recommend():
    sparse_mx = SparseMatrix()
    sparse_mx.set(0,0,1)
    sparse_mx.set(0,1,2)
    sparse_mx.set(1,0,3)
    sparse_mx.set(1,1,4)
    my_list = [] 
    with pytest.raises(ValueError):
        result = sparse_mx.recommend(my_list)
        
    my_list1 = [1,2]
    vector1 = np.array(my_list1).reshape(2,1)
    result = sparse_mx.recommend(vector1)
    assert result[0][0] == 5
    assert result[1][0] == 11
    
    my_list2 = [1,2, 3] # if matrix multiplication fails, it should raise ValueError
    vector2 = np.array(my_list2).reshape(3,1)
    with pytest.raises(ValueError):
        result = sparse_mx.recommend(vector2)
    
    my_list3 = [[1,2], [3,4]] # if vector is not 1D, code should raise ValueError
    vector3 = np.array(my_list3)
    with pytest.raises(ValueError):
        result = sparse_mx.recommend(vector3)
    
def test_add_movie():
    sparse_mx1 = SparseMatrix()
    sparse_mx1.set(0,1,3)
   
    sparse_mx2 = SparseMatrix()
    sparse_mx2.set(0,1,4)
    sparse_mx2.set(0,0,6)
    
    result = sparse_mx1.add_movie(sparse_mx2.entries)
    assert result.get((0,0),0) == 0
    assert result[(0,1)] == 3
    assert result[(1,0)] == 6
    assert result[(1,1)] == 4

    sparse_mx2.set(0,5,9)
    with pytest.raises(IndexError):
        result = sparse_mx1.add_movie(sparse_mx2.entries)
        
    sparse_mx3 = SparseMatrix()
    sparse_mx3.set(0,1,5)
    sparse_mx3.set(1,1,7)
    result = sparse_mx1.add_movie(sparse_mx3.entries)
    assert result[(2,1)] == 5
    assert result[(3,1)] == 7
    
    sparse_mx4 = SparseMatrix() # if sparse matrix is empty, then valueerror will be raised
    with pytest.raises(ValueError):
        result = sparse_mx1.add_movie(sparse_mx4.entries)
        
    sparse_mx5 = SparseMatrix()
    sparse_mx5.set(0, 1, '')
    with pytest.raises(ValueError): # it should raise ValueError if invalid values are passed
        result = sparse_mx1.add_movie(sparse_mx5.entries)
    

def test_to_dense():
    sparse_mx = SparseMatrix()
    result = sparse_mx.to_dense()
    assert result == []
    sparse_mx.set(0,0,10)
    sparse_mx.set(1,1,2)
    sparse_mx.set(1,2,6)
    sparse_mx.set(2,2,5)
    result = sparse_mx.to_dense()
    assert result[0][0] == 10
    assert result[1][1] == 2
    assert result[2][2] == 5
    assert result[2][0] == 0
    
    
     
     
    
    
    