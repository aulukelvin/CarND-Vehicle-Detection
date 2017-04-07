import numpy as np
import cv2
import matplotlib.pyplot as plt

def warper(img, M):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def pltPairedShow(img1,title1, img2, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
        
    if (len(img1.shape)<3):
        ax1.imshow(img1,cmap ='gray')
    else:
        ax1.imshow(img1)
        
    ax1.set_title(title1, fontsize=50)
    
    if (len(img2.shape)<3):
        ax2.imshow(img2,cmap ='gray')
    else:
        ax2.imshow(img2)
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def pltShowFour(img1,title1, img2, title2, img3, title3, img4, title4):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax2.imshow(img2,cmap ='gray')
    ax2.set_title(title2)
    ax3.imshow(img3,cmap ='gray')
    ax3.set_title(title3)
    ax4.imshow(img4,cmap ='gray')
    ax4.set_title(title4)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
#def hls_color_thresh(img, Hthresh, Sthresh):
def hls_s_thresh(img, Hthresh=(0,255), Sthresh=(0,255)):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    binary_output = np.zeros_like(img)
    binary_output[(imgHLS[:,:,0] >= Hthresh[0])  & (imgHLS[:,:,0] <= Hthresh[1]) 
                  & ((imgHLS[:,:,2] >= Sthresh[0])  & (imgHLS[:,:,2] <= Sthresh[1]))] = 255
    return binary_output

def sobelX_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):   
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # r channel
#    sobelx = cv2.Sobel(img[:,:,0], cv2.CV_64F, 1,0, ksize=sobel_kernel)
    # V channel
    sobelx = cv2.Sobel(imgHLS[:,:,2], cv2.CV_64F, 1,0, ksize=sobel_kernel)
    scaled_sobel = np.uint8(255*sobelx / np.max(sobelx))
       
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255

    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):    
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = imgHLS[:,:,2]
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*gradmag / np.max(gradmag))
       
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255


    # 6) Return this mask as your binary_output image
    return binary_output

#Direction threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = imgHLS[:,:,2]
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    absgraddir = np.arctan2(abs_sobely, abs_sobelx) 

    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    return binary_output

#Both Magnitude and direction threshold
def mag_dir_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0,np.pi/2)):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = imgHLS[:,:,2]
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize=sobel_kernel) 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    absgraddir = np.arctan2(abs_sobely, abs_sobelx) 

    scaled_sobel = np.uint8(255*gradmag / np.max(gradmag))
       
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1]) & (absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1]) ] = 255
    
    return binary_output

def fitlines(binary_warped, pri_left_fitx, pri_right_fitx,lf_history, rf_history):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    confident_left = histogram[leftx_base] / (histogram[leftx_base] + histogram[rightx_base])
    confident_right = histogram[rightx_base] / (histogram[leftx_base] + histogram[rightx_base])
    
    primary_side = 'right'
    if confident_left>confident_right:
        primary_side='left'
    distance = rightx_base-leftx_base
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit = [0,0,0]
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
    
    if len(rightx) == 0:
        right_fit = [0,0,0]
    else:
        right_fit = np.polyfit(righty, rightx, 2)
    
    #consolidate the left and right curv using the confident factors
#    right_fit[0] = confident_left * left_fit[0] + confident_right * right_fit[0]
#    right_fit[1] = confident_left * left_fit[1] + confident_right * right_fit[1]
#    right_fit[2] = confident_left * (left_fit[2]-leftx_base) + confident_right * (right_fit[2] - rightx_base) + rightx_base
#    left_fit[0] = confident_left * left_fit[0] + confident_right * right_fit[0]
#    left_fit[1] = confident_left * left_fit[1] + confident_right * right_fit[1]
#    left_fit[2] = confident_left * (left_fit[2]-leftx_base) + confident_right * (right_fit[2] - rightx_base) + leftx_base
    
    if left_fit[0] > 0.0004 or left_fit[0] < -0.0005 or left_fit[1]>0.8 or left_fit[1] < -0.5:
        left_fit = lf_history[0]
        
    lf_history[4] = lf_history[3]
    lf_history[3] = lf_history[2]
    lf_history[2] = lf_history[1]
    lf_history[1] = lf_history[0]
    lf_history[0] = left_fit
    
    left_fit = np.mean(lf_history, axis=0)
    
    if right_fit[0] > 0.0004 or right_fit[0] < -0.0005 or right_fit[1]>0.8 or right_fit[1] < -0.5:
        right_fit = rf_history[0]

    rf_history[4] = rf_history[3]
    rf_history[3] = rf_history[2]
    rf_history[2] = rf_history[1]
    rf_history[1] = rf_history[0]
    rf_history[0] = right_fit
    
    right_fit = np.mean(rf_history, axis=0)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    print(len(leftx),' - ', len(rightx))
    
    if primary_side=='left':
        if len(leftx)<5000:
            print('restore the lane lines to the previous ones')
            left_fitx = pri_left_fitx
            right_fitx = pri_right_fitx
        else:
            if len(rightx)<2000:
                print('set right lane lines to the left one')
                right_fitx = left_fitx + max(distance, 750)
    else:
        if len(rightx)<5000:
            print('restore the lane lines to the previous ones')
            left_fitx = pri_left_fitx
            right_fitx = pri_right_fitx
        else:
            if len(leftx)<2000:
                print('set left lane lines to the right one')
                left_fitx = right_fitx - max(distance, 750)
                
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    
    return left_fitx, right_fitx, left_curverad, right_curverad, out_img, left_fit, right_fit

def curvature():
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def combinedThresholds(img):
    combined3 = np.zeros_like(img[:,:,2])

    hls_mask = hls_s_thresh(img, Hthresh=(0, 50), Sthresh=(120, 255))[:,:,2]
    sobelX_mask = sobelX_thresh(img, 15,(70,240))
    mag_mask = mag_thresh(img, 15,(25,100))
    dir_mask = dir_threshold(img, 15,(0.7, 1.2))

    combined3[(((hls_mask >=100 ) & (sobelX_mask >=100)) | ((mag_mask >= 100) & (dir_mask >= 100)))] = 255
#    combined3[(((hls_mask >=100 ) & (sobelX_mask >=100)))] = 255

    return combined3

def processingframe(image, left_fitx, right_fitx, M, Minv):
    
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale=1
    thickness=2
    
    combined = combinedThresholds(image)
    
    binary_warped = warper(combined, M)
    
    left_fitx, right_fitx, left_curverad, right_curverad, out_img = fitlines(binary_warped, left_fitx, right_fitx)

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warper(color_warp, Minv) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    middle = int((left_fitx[-1] + right_fitx[-1])//2)
    veh_pos = int(image.shape[1]//2)
    
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    dx = (veh_pos - middle)*xm_per_pix 
    
    #draw position lines    
    cv2.line(result, (veh_pos, 400), (veh_pos, 720), (255,0,0),2)
    #cv2.line(result, (veh_pos-20, 400), (veh_pos+20, 400), (200,200,200),2)
    cv2.line(result, (veh_pos-30, 450), (veh_pos+30, 450), (200,200,200),2)
    #cv2.line(result, (veh_pos-20, 500), (veh_pos+20, 500), (200,200,200),2)
    cv2.line(result, (middle, 550), (middle, 720), (200,200,200),4)

    img_out=np.zeros((576,1280,3), dtype=np.uint8)
    img_out[0:576,0:1024,:] =cv2.resize(result,(1024,576))
    #b) Threshold
    img_out[0:288,1024:1280, 0] =cv2.resize(combined,(256,288))
    img_out[0:288,1024:1280, 1] =cv2.resize(combined,(256,288))
    img_out[0:288,1024:1280, 2] =cv2.resize(combined,(256,288))
    #c)Birds eye view
    img_out[310:576,1024:1280,:] =cv2.resize(out_img,(256,266))
    
    #Write curvature and center in image
    TextL = "Left  curv: {:>.1f} km".format(left_curverad/1000)
    TextR = "Right curv: {:>.1f} km".format(right_curverad/1000)

    cv2.putText(img_out, TextL, (130,40), fontFace, fontScale,(200,200,200), thickness,  lineType = cv2.LINE_AA)
    cv2.putText(img_out, TextR, (130,70), fontFace, fontScale,(200,200,200), thickness,  lineType = cv2.LINE_AA)
    cv2.putText(img_out,'Position : %.2f m %s of center'%(abs(dx), 'left' if dx < 0 else 'right'),(50,110), 
                        fontFace, fontScale,(200,200,200),thickness,lineType = cv2.LINE_AA)

    cv2.putText(img_out, "Thresh. view", (1070,30), fontFace, .8,(200,200,0), thickness,  lineType = cv2.LINE_AA)
    cv2.putText(img_out, "Birds-eye view", (1080,305), fontFace, .8,(200,200,0), thickness,  lineType = cv2.LINE_AA)
    
    return img_out