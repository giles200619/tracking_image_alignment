import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import random
import math
from scipy.interpolate import interpn


def find_match(img1, img2):
    # To do
    #img1, img2 = template, target_list[0]
    #
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.015,edgeThreshold = 5,sigma = 1.4)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #find the 2 nearestN for each point in des1
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des2)
    distances,matches = neigh.kneighbors(des1,n_neighbors=2, return_distance=True)#output rows: des1[0]
    #apply ratio test:
    x1_des1 = []
    x2_des1 = []
    for i in range(0,matches.shape[0]): # loop through all the matches for des1
        if(distances[i][0]/distances[i][1]<0.7):
            x1_des1.append(kp1[i])
            x2_des1.append(kp2[matches[i][0]])
    #find nearestN for each point in des2?         
    
    
    x1 = np.zeros((len(x1_des1),2))
    x2 = np.zeros((len(x2_des1),2))
    
    for i in range(0,len(x1_des1)):
        (x1[i][0],x1[i][1]) = x1_des1[i].pt
        (x2[i][0],x2[i][1]) = x2_des1[i].pt
    return x1, x2

def get_Ab(x1,x2):# x1*H=x2
    #u1 = x1[0][0] v1 = x1[0][1]  u1'=x2[0][0] v1'=x2[0][1]
    #Homography matrix calculation
# =============================================================================
#     A =  np.array([[x1[0][0],x1[0][1],1,0,0,0,-x1[0][0]*x2[0][0],-x1[0][1]*x2[0][0]],
#                [0,0,0,x1[0][0],x1[0][1],1,-x1[0][0]*x2[0][1],-x1[0][1]*x2[0][1]],
#                [x1[1][0],x1[1][1],1,0,0,0,-x1[1][0]*x2[1][0],-x1[1][1]*x2[1][0]],
#                [0,0,0,x1[1][0],x1[1][1],1,-x1[1][0]*x2[1][1],-x1[1][1]*x2[1][1]],
#                [x1[2][0],x1[2][1],1,0,0,0,-x1[2][0]*x2[2][0],-x1[2][1]*x2[2][0]],
#                [0,0,0,x1[2][0],x1[2][1],1,-x1[2][0]*x2[2][1],-x1[2][1]*x2[2][1]],
#                [x1[3][0],x1[3][1],1,0,0,0,-x1[3][0]*x2[3][0],-x1[3][1]*x2[3][0]],
#                [0,0,0,x1[3][0],x1[3][1],1,-x1[3][0]*x2[3][1],-x1[3][1]*x2[3][1]]])
#     
#     b =   np.array([[x2[0][0]],[x2[0][1]],[x2[1][0]],[x2[1][1]],[x2[2][0]],[x2[2][1]],[x2[3][0]],[x2[3][1]]])  
# =============================================================================
    #affine 
    A =  np.array([[x1[0][0],x1[0][1],1,0,0,0],
               [0,0,0,x1[0][0],x1[0][1],1],
               [x1[1][0],x1[1][1],1,0,0,0],
               [0,0,0,x1[1][0],x1[1][1],1],
               [x1[2][0],x1[2][1],1,0,0,0],
               [0,0,0,x1[2][0],x1[2][1],1]])
     
    b =   np.array([[x2[0][0]],[x2[0][1]],[x2[1][0]],[x2[1][1]],[x2[2][0]],[x2[2][1]]])  
    
    return A,b

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    #ransac_thr = 5
    #ransac_iter = 100
    ##
    #check_list = []
    inlier_list = []
    best_A = np.zeros((8,1))
    for iteration in range(ransac_iter):   
        temp_list = []
        temp_list=random.sample(range(x1.shape[0]), 3)
        temp_list = sorted(temp_list)
# =============================================================================
#         while temp_list in check_list:
#             temp_list=random.sample(range(x1.shape[0]), 4)
#             temp_list=temp_list.sort()
# =============================================================================
        #check_list.append(temp_list)
        
        selected_x1 = np.zeros((3,2))
        selected_x2 = np.zeros((3,2))        
        for i in range(3):
            selected_x1[i,:]=x1[temp_list[i],:]
            selected_x2[i,:]=x2[temp_list[i],:]
         #to solveAh=b,define A
        A,b = get_Ab(selected_x1,selected_x2)
        #H = ((A.T*A).I)*(A.T)*b
        H = np.linalg.inv(np.matmul(A.transpose(), A))
        H = np.matmul(H, A.transpose())
        H = np.matmul(H, b)
        H = np.append(H,[[0],[0],[1]])
        H= np.reshape(H,(3,3))
        inlier = 0
        for j in range(0,x1.shape[0]):
            uv1 = np.array([[x1[j][0]],[x1[j][1]],[1]])
            uv1 = np.matmul(H, uv1)
            if np.sqrt(math.pow((uv1[0][0]-x2[j][0]),2)+math.pow((uv1[1][0]-x2[j][1]),2)) < ransac_thr:
                inlier+=1
        inlier_list.append(inlier)
        if inlier == max(inlier_list):
            best_A = H
        
    A = best_A
    #print(inlier_list)
    return A

def warp_image(img, A, output_size):
    # To do
    
    #
    img_warped = np.zeros(output_size,dtype=np.float64)
    
    for i in range(0,img_warped.shape[0]):
        for j in range(0,img_warped.shape[1]):
            uv1 = np.array([[j],[i],[1]])
            uv2 = np.matmul(A, uv1)
            if uv2[0][0]<img.shape[1] and uv2[1][0]<img.shape[0]:
                #img_warped[i][j] = img[int(uv2[0][0])][int(uv2[1][0])]
                img_warped[i][j] = img[int(uv2[1][0])][int(uv2[0][0])]
       
    return img_warped

def get_differential_filter():
    # To do
    #sobel filter
    filter_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
    filter_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    return filter_x, filter_y

def filter_image(im, filter):
    # To do
    # pad zero around 
    im_pad0 =np.zeros((im.shape[0]+2,im.shape[1]+2))
    for i in range(1,im_pad0.shape[0]-1):
        for j in range(1,im_pad0.shape[1]-1):
            im_pad0[i][j]=im[i-1][j-1]
    #filter image
    im_filtered0 = np.zeros((im.shape[0]+2,im.shape[1]+2))
    im_filtered = np.zeros((im.shape[0],im.shape[1]))
    #filter = [[1,0,-1],[1,0,-1],[1,0,-1]]
    for i in range(1,im_pad0.shape[0]-1):
        for j in range(1,im_pad0.shape[1]-1):
            #calculate each pixel
            im_filtered0[i][j]=im_pad0[i-1][j-1]*filter[0][0]+im_pad0[i-1][j]*filter[0][1]+im_pad0[i-1][j+1]*filter[0][2] \
            +im_pad0[i][j-1]*filter[1][0]+im_pad0[i][j]*filter[1][1]+im_pad0[i][j+1]*filter[1][2] \
            +im_pad0[i+1][j-1]*filter[2][0]+im_pad0[i+1][j]*filter[2][1]+im_pad0[i+1][j+1]*filter[2][2]
            
    #get rid of surrounding 0
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            im_filtered[i][j]=im_filtered0[i+1][j+1]
    
    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    #grad_magnitude
    grad_mag=np.zeros((im_dx.shape[0],im_dx.shape[1]))
    for i in range(0,im_dx.shape[0]):
        for j in range(0,im_dx.shape[1]):
            grad_mag[i][j]=math.sqrt(im_dx[i][j]*im_dx[i][j]+im_dy[i][j]*im_dy[i][j])
    
    return grad_mag



def align_image(template, target, A):
    # To do
    #A_refined, errors = align_image(template, target_list[0], A)
    #target = target_list[0]
    #get gradient of template image[Ix,Iy]
    template = template.astype('float') / 255.0    
    target = target.astype('float') / 255.0    
    #(filter_x, filter_y) = get_differential_filter()
    #im_dx = filter_image(template, filter_x)
    #im_dy = filter_image(template, filter_y)
    im_dx = cv2.Sobel(template,cv2.CV_64F,1,0,ksize=3)
    im_dy = cv2.Sobel(template,cv2.CV_64F,0,1,ksize=3)
    #get A
    D = np.empty((np.product(template.shape),6))
    index = 0
    for i in range (0,im_dx.shape[0]):
        for j in range(0,im_dx.shape[1]):
            #Ix*u(j)  Ix*v(i)  Ix  Iy*u Iy*v Iy
            D[index][0]=im_dx[i][j]*j
            D[index][1]=im_dx[i][j]*i
            D[index][2]=im_dx[i][j]
            D[index][3]=im_dy[i][j]*j
            D[index][4]=im_dy[i][j]*i
            D[index][5]=im_dy[i][j]
            #a = np.array([im_dx[i][j]*j,im_dx[i][j]*i ,im_dx[i][j] ,im_dy[i][j]*j, im_dy[i][j]*i, im_dy[i][j]])
            index+=1   
    #H=A.T*A
    H = np.matmul(D.transpose(), D)
    print(H)
    #initialize p
    dp = np.array([[A[0][0]-1],[A[0][1]],[A[0][2]],[A[1][0]],[A[1][1]-1],[A[1][2]]])
    iterations = 0
    pre = 0
    errors = []
    while( math.sqrt(dp[0][0]*dp[0][0]+dp[1][0]*dp[1][0]+dp[2][0]*dp[2][0]+dp[3][0]*dp[3][0]+dp[4][0]*dp[4][0]+dp[5][0]*dp[5][0]) > 0.0001):
        #Itgt = warp_image(target, A, template.shape)
        #Ierr = Itgt-template
        
        #Ierr = np.reshape(Ierr, (np.product(Ierr.shape),1))
# =============================================================================
        Itgt = np.zeros(template.shape,dtype=np.float64)
        Ierr = np.zeros(template.shape,dtype=np.float64)
        F = np.zeros((6,1))
        for i in range (0,Itgt.shape[0]):
            for j in range(0,Itgt.shape[1]):
                uv1 = np.array([[j],[i],[1]])
                uv2 = np.matmul(A, uv1)
                if uv2[0][0]<target.shape[1] and uv2[1][0]<target.shape[0]:
                     Itgt[i][j] = target[int(uv2[1][0])][int(uv2[0][0])]
                     Ierr[i][j] = Itgt[i][j]-template[i][j]
                     F[0][0]=F[0][0]+im_dx[i][j]*j*Ierr[i][j]
                     F[1][0]=F[1][0]+im_dx[i][j]*i*Ierr[i][j]
                     F[2][0]=F[2][0]+im_dx[i][j]*Ierr[i][j]
                     F[3][0]=F[3][0]+im_dy[i][j]*j*Ierr[i][j]
                     F[4][0]=F[4][0]+im_dy[i][j]*i*Ierr[i][j]
                     F[5][0]=F[5][0]+im_dy[i][j]*Ierr[i][j]
# =============================================================================
 #       dp = np.matmul(np.linalg.inv(H),D.transpose())
  #      dp = np.matmul(dp,Ierr)
        dp = np.matmul(np.linalg.inv(H),F)
        dA = np.array([[1+dp[0][0],dp[1][0],dp[2][0]],[dp[3][0],1+dp[4][0],dp[5][0]],[0,0,1]])#p1, p1+1
        A = np.matmul(A,np.linalg.inv(dA))
        print(np.linalg.norm(Ierr),np.linalg.norm(dp),iterations)
        errors.append(np.linalg.norm(Ierr))
        iterations+=1
        if iterations>800 or np.linalg.norm(Ierr)==pre: 
           break
        pre = np.linalg.norm(Ierr)
    A_refined = A

    return A_refined, errors


def track_multi_frames(template, img_list):
    # To do
    #img_list = target_list
    #
    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1, x2, 3, 5000)
    #
    A_List = []
    A_refined, errors = align_image(template, img_list[0], A)
    errors_1=errors
    A_List.append(A_refined)
    #warp_image(target_list[0], A, template.shape)
    template = warp_image(target_list[0], A_refined, template.shape)
    A_refined, errors = align_image(template, img_list[1], A_refined)
    A_List.append(A_refined)
    
    template = warp_image(target_list[1], A_refined, template.shape)
    A_refined, errors = align_image(template, img_list[2], A_refined)
    A_List.append(A_refined)
    
    template = warp_image(target_list[2], A_refined, template.shape)
    A_refined, errors = align_image(template, img_list[3], A_refined)
    A_List.append(A_refined)
        
    return A_List,errors_1


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    #boundary_t = np.hstack((np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]],
    #                                    [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
    #boundary_t = boundary_t*scale_factor2
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    #plt.plot(boundary_t[:, 0]+img1.shape[1]* scale_factor1, boundary_t[:, 1], 'r')
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    #target = target_list[0]
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./data/0.jpg', 0)  
    target_list = []
    for i in range(4):
        target = cv2.imread('./data/{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    #%matplotlib qt 
    
    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    A = align_image_using_feature(x1, x2, 3, 5000)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list,errors_1 = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)
    new_A = np.copy(A_list)
    for i in range(len(new_A)):
        new_A[i][0][0] = (new_A[i][0][0]+new_A[i][1][1])/2
        new_A[i][1][1] = new_A[i][0][0]
        new_A[i][0][1] = (abs(new_A[i][1][0])+abs(new_A[i][0][1]))/2
        new_A[i][1][0] = -new_A[i][0][1]
    visualize_track_multi_frames(template, target_list, new_A)
    
    
        
    #

    

