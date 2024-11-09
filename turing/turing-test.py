'''
You are given an m x n integer matrix matrix with the following two properties:

Each row is sorted in non-decreasing order. The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.
'''

def matrix_search(matrix: list[list[int]], target: int) -> bool:
    # destructure the matrixs
    for row in matrix:
        if target <= row[-1] and target >= row[0]:
            print(f"Row {row} | Target {target}: meets condition")
            # then target is likely to be in this row.
            for col in row:
                if col == target:
                    return True
        else:
            continue
    return False



matrix =  [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
print(f"Example 1: {matrix_search(matrix=matrix, target=3)}")
print(f"Example 2: {matrix_search(matrix=matrix, target=13)}")
