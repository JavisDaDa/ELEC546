function [ K, Q, t ] = P_to_KRt( P )

M = P(1:3, 1:3);
%Code to get RQ decomposition from QR decomposition (of M)

%The RQ decomposition transforms a matrix A into the product of an upper triangular matrix R (also known as right-triangular) and an orthogonal matrix Q. The only difference from QR decomposition is the order of these matrices.
%QR decomposition is Gram–Schmidt orthogonalization of columns of A, started from the first column.
%RQ decomposition is Gram–Schmidt orthogonalization of rows of A, started from the last row.

%extrating the rows of A
r1 = M(1, :);
r2 = M(2, :);
r3 = M(3, :);

%putting the rows of A as columns in another matrix and then doing QR of
%that matrix
[q r] = qr([r3' r2' r1']);

%extracting the columns of the rotation matrix q
q_c1 = q(:,1);
q_c2 = q(:,2);
q_c3 = q(:,3);

%putting the columns of q as rows (in a certain order, see the introductory comments) in a new matrix Q
Q = [q_c3'; q_c2'; q_c1'];

%rearranging (in a certain manner, see the introductory comments) the coeff in the upper triangular matrix 'r' to construct
%another upper triangular matrix R
R = zeros(3,3);
R(3,3) = r(1,1);
R(2,3) = r(1,2);
R(1,3) = r(1,3);
R(2,2) = r(2,2);
R(1,2) = r(2,3);
R(1,1) = r(3,3);

%normalizing so that the (3,3) coefficient = 1
K = R/R(3,3);

%K(1,1) and K(2,2) are supposed to be focal lengths and hence should be >0
if K(1,1) < 0
    K(:,1) = -1*K(:,1);
    Q(1,:) = -1*Q(1,:);
end
if K(2,2) < 0
    K(:,2) = -1*K(:,2)
    Q(2,:) = -1*Q(2,:);
end

%Sanity check; Q must be a rotation matrix
if det(Q) < 0
    disp('Warning: determinant of the supposed rotation matrix is -1');
end

%Computing t
P_3_3 = K*Q;
P_proper_scale = P_3_3(1,1)*P/P(1,1) ;

t = inv(K)*P_proper_scale(:,4);
end

