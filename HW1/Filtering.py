import utils
import numpy as np
if __name__ == '__main__':
    # Filtering
    barbara = utils.load_image('barbara.jpg')
    barbara_gray = utils.convert2gray(barbara)
    I1 = np.array([[120, 110, 90, 115, 40],
                   [145, 135, 135, 65, 35],
                   [125, 115, 55, 35, 25],
                   [80, 45, 45, 20, 15],
                   [40, 35, 25, 10, 10]])
    I2 = np.array([[125, 130, 135, 110, 125],
                   [145, 135, 135, 155, 125],
                   [65, 60, 55, 45, 40],
                   [40, 35, 40, 25, 15],
                   [15, 15, 20, 15, 10]])
    filter1 = 1/3 * np.array(np.ones(3)).reshape((1, 3))
    filter2 = 1/3 * np.array(np.ones(3)).reshape((3, 1))
    filter3 = np.array(1/9 * np.ones((3,3)))
    matrix = [I1, I2]
    filters = [filter1, filter2, filter3]
    for i, matrix in enumerate(matrix):
        for j, fil in enumerate(filters):
            res = utils.matrix_filter(matrix, fil, flag=j)
            print(f'matrix I{i+1} after filter {j+1} is \n {res}')

    # Apply filters
    utils.show_image(barbara_gray, gray=True)
    # cv2.filter2D()
    '''
    Todo
    '''