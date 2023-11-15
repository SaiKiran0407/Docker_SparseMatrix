import numpy as np


class SparseMatrix:

    def __init__(self):
        self.entries = {}
        
    def set(self, row, col, value):
        if row < 0 or col < 0:
            raise ValueError("Negative row or column!")
        if value !=0 and value: # dont enter if value is 0 or invalid
           self.entries[(row, col)] = value

    def get(self, row, col):
        if not self.entries:
            raise ValueError("No data available to access!")
        sparse_mx_max_row = self.find_max_row(self.entries) 
        sparse_mx_max_col = self.find_max_col(self.entries)
        if row > sparse_mx_max_row or col > sparse_mx_max_col:
            raise KeyError("Invalid row or column!")
        if row < 0 or col < 0:
            raise KeyError("Negative row or column!")
        return self.entries.get((row, col), 0)
    
    def recommend(self, vector):
        if len(vector) == 0:
            raise ValueError("Vector is empty!")
        sparse_mx_max_row = self.find_max_row(self.entries) +1
        sparse_mx_max_col = self.find_max_col(self.entries) +1
        vector_max_row = np.shape(vector)[0]
        vector_max_col = np.shape(vector)[1]
        if sparse_mx_max_col != vector_max_row:
            raise ValueError("matrix multiplication is not possible")
        if vector_max_col != 1:
            raise ValueError("expected single column vector")
        
        else: 
            result = np.zeros((sparse_mx_max_row, vector_max_col))
            for (row, col), val in self.entries.items():
                result[row][0] += val * vector[col][0]
                
            return result   
     
    def add_movie(self, new_matrix):
        if not new_matrix:
            raise ValueError("Empty matrix is not supported")
        sparse_mx_max_row = self.find_max_row(self.entries)
        sparse_mx_max_col = self.find_max_col(self.entries)
        new_mx_max_row = self.find_max_row(new_matrix)
        new_mx_max_col = self.find_max_col(new_matrix)
        if sparse_mx_max_col != new_mx_max_col:
            raise IndexError("cannot add new movie due to it's invalid column size")
        
        for (row, col), val in new_matrix.items():
            if not val:
                raise ValueError("Matrix data is not valid")
            self.entries[(sparse_mx_max_row + 1 + row, col)] = val
        
        return self.entries
            
    def to_dense(self):
        if not self.entries:
            return [] # no entries, so return empty list
        sparse_mx_max_row = self.find_max_row(self.entries)
        sparse_mx_max_col = self.find_max_col(self.entries)
        dense_mx = []
        for i in range(sparse_mx_max_row +1):
            dense_mx.append([0] * (sparse_mx_max_col +1))
            
        for (row, col), val in self.entries.items():
            dense_mx[row][col] = val
            
        return dense_mx
            
    '''
        finds the maximum row value for a sparse matrix
    '''
    def find_max_row(self, movie_dict):
        if not self.entries:
            return
        else:
            entry_keys = [row for (row, col) in movie_dict.keys()]
            return max(entry_keys)
    
    '''
        finds the maximum col value for a sparse matrix
    '''    
    def find_max_col(self, movie_dict):
        if not self.entries:
            return
        else:
            entry_keys = [col for (row, col) in movie_dict.keys()]
            return max(entry_keys)
        
        
    
    