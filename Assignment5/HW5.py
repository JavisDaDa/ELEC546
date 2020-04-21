import scipy.io
import cv2
import numpy as np
from scipy.linalg import rq, inv, det
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def load_mat(path):
    data = scipy.io.loadmat(path)
    pts_2D = data['pts_2D']
    pts_3D = data['cam_pts_3D']
    return pts_2D, pts_3D


def claculateK(pts_2D, pts_3D):
    pts_2DT = pts_2D.T
    pts_3DT = pts_3D.T
    add = np.ones((28, 1))
    A = np.hstack((pts_2DT, add))
    AT = A.T

    add1 = np.empty((28, 1))
    add2 = np.empty((28, 1))
    for i in range(28):
        add1[i] = pts_3DT[i][0] / pts_3DT[i][2]
        add2[i] = pts_3DT[i][1] / pts_3DT[i][2]

    B = np.hstack((add1, add2, add))
    BT = B.T
    Binv = np.linalg.pinv(BT)
    K = AT @ Binv
    print(f'Intrinsic parameter is:\n {K}')
    return K


def calibrate(x, X):
    '''
    This function computes camera projection matrix from 3D scene points
    and corresponding 2D image points with the Direct Linear Transformation (DLT).

    Usage:
    P = calibrate(x, X)

    Input:
    x: 2xn image points
    X: 3xn scene points

    Output:
    P: 3x4 camera projection matrix

    '''
    xt = np.transpose(x)
    Xt = np.transpose(X)
    # print(Xt)
    # print(xt)

    Xt = np.hstack((Xt, np.ones((28, 1))))
    print(Xt.shape)
    zero4 = np.array((0, 0, 0, 0))

    M = np.array((56, 12))
    for i in range(0, 28):
        A = np.hstack((zero4, -Xt[i], xt[i][1] * Xt[i]))
        B = np.hstack((Xt[i], zero4, -xt[i][0] * Xt[i]))
        A = np.reshape(A, (1, 12))
        B = np.reshape(B, (1, 12))
        if i == 0:
            M = np.vstack((A, B))

        else:
            M = np.vstack((M, A, B))

    u, s, vtranspose = np.linalg.svd(M)
    v = np.transpose(vtranspose)
    p = v[:, 11]

    P = p.reshape((3, 4))
    return P


def P_to_KRt(P):
    '''

    This function computes the decomposition of the projection matrix into intrinsic parameters, K, and extrinsic parameters Q (the rotation matrix) and t (the translation vector)

    Usage:
    K, R, t = P_to_KRt(P)

    Input:
    P: 3x4 projection matrix

    Outputs:
    K: 3x3 camera intrinsics
    R: 3x3 rotation matrix (extrinsics)
    t: 3x1 translation vector(extrinsics)

    '''

    M = P[0:3, 0:3]

    Q, R = rq(M)

    K = Q / float(Q[2, 2])

    if K[0, 0] < 0:
        K[:, 0] = -1 * K[:, 0]
        R[0, :] = -1 * R[0, :]

    if K[1, 1] < 0:
        K[:, 1] = -1 * K[:, 1]
        R[1, :] = -1 * R[1, :]

    if det(R) < 0:
        print('Warning: Determinant of the supposed rotation matrix is -1')

    P_3_3 = np.dot(K, R)

    P_proper_scale = (P_3_3[0, 0] * P) / float(P[0, 0])

    t = np.dot(inv(K), P_proper_scale[:, 3])

    return K, R, t


def calculate_error(pts_2D, pts_3D):
    P = calibrate(pts_2D, pts_3D)
    print(f'P: \n{P}')
    K, R, t = P_to_KRt(P)
    print(f'K: \n{K}')
    print(f'R: \n{R}')
    print(f't: \n{t}')
    XT = pts_3D.T
    XT = np.hstack((XT, np.ones((28, 1))))
    X_ = XT.T
    xnew = (P @ X_).T
    add1 = np.empty((28, 1))
    add2 = np.empty((28, 1))
    for i in range(28):
        add1[i] = xnew[i][0] / xnew[i][2]
        add2[i] = xnew[i][1] / xnew[i][2]
    x_proj = np.hstack((add1, add2))
    xt = pts_2D.T
    error = 0
    for i in range(28):
        error += np.square(xt[i][0] - x_proj[i][0]) + np.square(xt[i][1] - x_proj[i][1])
    res = np.sqrt(error) / 28
    print(f'reprojection error: {res}')
    return xt, x_proj


def display(xt, x_proj, img_path):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    img = cv2.imread(img_path)
    xorg = xt[:, 0]
    yorg = xt[:, 1]
    xproj = x_proj[:, 0]
    yproj = x_proj[:, 1]
    for xx, yy in zip(xorg, yorg):
        circ = Circle((xx, yy), 20, color='r')
        ax.add_patch(circ)

    for xx, yy in zip(xproj, yproj):
        circ = Circle((xx, yy), 10, color='y')
        ax.add_patch(circ)
    ax.imshow(img)
    plt.savefig('corner.png')
    plt.show()


def main():
    mat_path = 'Matlab_funs_data/pt_corres.mat'
    pts_2D, pts_3D = load_mat(mat_path)
    # print(pts_2D.shape)
    # print(pts_3D.shape)

    # print(pts_2DT.shape)
    # print(pts_3DT.shape)


    K = claculateK(pts_2D, pts_3D)
    pts_3d = scipy.io.loadmat('rubik_3D_pts.mat')['pts_3d']
    pts_2d = scipy.io.loadmat('rubik_2D_pts.mat')['pts_2d']
    xt, x_proj = calculate_error(pts_2d, pts_3d)
    img_path = 'rubik_cube.jpg'
    display(xt, x_proj, img_path)


if __name__ == '__main__':
    main()