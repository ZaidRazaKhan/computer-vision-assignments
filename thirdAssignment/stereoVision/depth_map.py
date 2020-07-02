import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

def graph_cut(disparity, image, threshold):
    for i in range(0, image.height):
        for j in range(0, image.width):
            if cv2.GetReal2D(disparity,i,j) > threshold:
                cv2.Set2D(disparity,i,j,cv2.Get2D(image,i,j))

def plot_disparity_map(left_image, right_image):
    disparity_left = cv2.CreateMat(left_image.height, left_image.width, cv2.CV_16S)
    disparity_right = cv2.CreateMat(right_image.height, right_image.width, cv2.CV_16S)
    state = cv2.CreateStereoGCState(16, 2)
    cv2.FindStereoCorrespondenceGC(left_image, right_image, disparity_left, disparity_right, state)
    disp_left_visual = cv2.CreateMat(left_image.height, left_image.width, cv2.CV_8U)
    cv2.ConvertScale(disparity_left, disp_left_visual, -16)
    cv2.Save("disparity.pgm", disp_left_visual)
    graph_cut(disp_left_visual, left_image, 120)
    cv2.NamedWindow('Disparity map', cv2.CV_WINDOW_AUTOSIZE)
    cv2.ShowImage('Disparit map', disp_left_visual)
    cv2.WaitKey()

# # loading the stereo pair
# left  = cv2.LoadImage('scene_l.bmp',cv2.CV_LOAD_IMAGE_GRAYSCALE)
# right = cv2.LoadImage('scene_r.bmp',cv2.CV_LOAD_IMAGE_GRAYSCALE)

# disparity_left  = cv2.CreateMat(left.height, left.width, cv2.CV_16S)
# disparity_right = cv2.CreateMat(left.height, left.width, cv2.CV_16S)

# # data structure initialization
# state = cv2.CreateStereoGCState(16,2)
# # running the graph-cut algorithm
# cv2.FindStereoCorrespondenceGC(left,right,
#                           disparity_left,disparity_right,state)

# disp_left_visual = cv2.CreateMat(left.height, left.width, cv2.CV_8U)
# cv2.ConvertScale( disparity_left, disp_left_visual, -16 )
# cv2.Save( "disparity.pgm", disp_left_visual )

# # cutting the object farthest of a threshold (120)
# cut(disp_left_visual,left,120)

# cv2.NamedWindow('Disparity map', cv2.CV_WINDOW_AUTOSIZE)
# cv2.ShowImage('Disparity map', disp_left_visual)
# cv2.WaitKey()



def plot_depth_map(file_path_left, file_path_right, numDisparities, blockSize):
    imgL = cv2.imread(file_path_left, 0)
    imgR = cv2.imread(file_path_right,0)

    stereo = cv2.createStereoBM(numDisparities, blockSize)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-fpl", "--file_path_left", required="True", help="Path to left image file for which you want to make depth map")
    ap.add_argument("-fpr", "--file_path_right", required="True", help="Path to right image file for which you want to make depth map")
    ap.add_argument('-nd', '--number_of_disparities', required=False, default=16, help="Number of disparities")
    ap.add_argument('-bs', '--block_size', required= False, default=10, help="Block size for stereo map")
    args = vars(ap.parse_args())

    plot_depth_map(args['file_path_left'], args['file_path_right'], args['number_of_disparities'], args['block_size'])