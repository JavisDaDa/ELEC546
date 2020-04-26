class BinaryMatrix(object):
    def __init__(self, mat):
        self.mat = mat

    def get(self, x: int, y: int) -> int:
        return self.mat[x][y]

    def dimensions(self):
        n = len(self.mat)
        m = len(self.mat[0])
        return n, m


class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        n, m = binaryMatrix.dimensions()
        res = float('inf')
        for i in range(n):
            low = 0
            high = m - 1
            while low + 1 < high:
                mid = low + (high - low) // 2
                if binaryMatrix.get(i, mid) == 1:
                    high = mid
                else:
                    low = mid
            if binaryMatrix.get(i, low) == 1:
                res = min(res, low)
            elif binaryMatrix.get(i, high) == 1:
                res = min(res, high)
        return res


if __name__ == '__main__':
    s = Solution()
    mat = [[0,0,0,1],[0,0,1,1],[0,1,1,1]]
    matrix = BinaryMatrix(mat)
    res = s.leftMostColumnWithOne(matrix)
    print(res)