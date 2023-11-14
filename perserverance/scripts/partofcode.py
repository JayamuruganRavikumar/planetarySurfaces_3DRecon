Y=np.arange(0,2048)
X=np.arange(0,2048)
(XX_field,YY_field)=np.meshgrid(X,Y)

#I mount the X, Y and disparity in a same 3D array 
stock = np.concatenate((np.expand_dims(XX_field,2),np.expand_dims(YY_field,2)),axis=2)
XY_disp = np.concatenate((stock,np.expand_dims(disp,2)),axis=2)

XY_disp_reshape = XY_disp.reshape(XY_disp.shape[0]*XY_disp.shape[1],3)

Ts = np.hstack((np.zeros((3,3)),T_0)) #i use only the translations obtained with the rectified calibration...Is it correct?


# I establish the projective matrix with the homography matrix
P11 = np.dot(rectmat1,C1)
P1 = np.vstack((np.hstack((P11,np.zeros((3,1)))),np.zeros((1,4))))
P1[3,3] = 1

# P1 = np.dot(C1,np.hstack((np.identity(3),np.zeros((3,1)))))

P22 = np.dot(np.dot(rectmat2,C2),Ts)
P2 = np.vstack((P22,np.zeros((1,4))))
P2[3,3] = 1

lambda_t = cv2.norm(P1[0,:].T)/cv2.norm(P2[0,:].T)


#I define the reconstruction matrix
Q = np.zeros((4,4))

Q[0,:] = P1[0,:].T
Q[1,:] = P1[1,:].T
Q[2,:] = lambda_t*P2[1,:].T - P1[1,:].T
Q[3,:] = P1[2,:].T

#I do the calculation to get my 3D coordinates
test = []
for i in range(0,XY_disp_reshape.shape[0]):
    a = np.dot(inv(Q),np.expand_dims(np.concatenate((XY_disp_reshape[i,:],np.ones((1))),axis=0),axis=1))
    test.append(a)

test = np.asarray(test)

XYZ = test[:,:,0].reshape(XY_disp.shape[0],XY_disp.shape[1],4)

