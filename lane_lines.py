import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import math
import os

kernel_size = 3 # Gaussian smoothing variable
low_threshold = 50
high_threshold = 150

# We are building trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85 
trap_top_width = 0.07
trap_height = 0.4  

# Hough Transform
rho = 2 
theta = 1 * np.pi/180 
threshold = 15	
min_line_length = 10 
max_line_gap = 20	

def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
def canny(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	mask = np.zeros_like(img)   

	if len(img.shape) > 2:
		channel_count = img.shape[2] 
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
	# If error, don't draw the lines
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True

	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0] 
		
		if x2 - x1 == 0.: 
			slope = 999. 
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines

	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2 
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)

	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False

	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)
	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	
	# Drawing  right and left lines on image
	if draw_right:
		cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
	if draw_left:
		cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
	
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((*img.shape, 3), dtype=np.uint8) 
	draw_lines(line_img, lines)
	return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	return cv2.addWeighted(initial_img, α, img, β, λ)

def filter_colors(image):

	# Filter white pixels
	white_threshold = 200 #130
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)

	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([90,100,100])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

	return image2

def annotate_image_array(image_in):
	
	image = filter_colors(image_in)
	gray = grayscale(image)
	blur_gray = gaussian_blur(gray, kernel_size)
	edges = canny(blur_gray, low_threshold, high_threshold)
	imshape = image.shape
	vertices = np.array([[\
		((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
		((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
		, dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)
	line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
	initial_image = image_in.astype('uint8')
	annotated_image = weighted_img(line_image, initial_image)
	
	return annotated_image

def annotate_image(input_file, output_file):
	annotated_image = annotate_image_array(mpimg.imread(input_file))
	plt.imsave(output_file, annotated_image)

def annotate_video(input_file, output_file):
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image_array)
	annotated_video.write_videofile(output_file, audio=False)

if __name__ == '__main__':
	from optparse import OptionParser

	# Configuring command line options
	parser = OptionParser()
	parser.add_option("-i", "--input_file", dest="input_file",
					help="Input video/image file")
	parser.add_option("-o", "--output_file", dest="output_file",
					help="Output (destination) video/image file")
	parser.add_option("-I", "--image_only",
					action="store_true", dest="image_only", default=False,
					help="Annotate image (defaults to annotating video)")

	options, args = parser.parse_args()

	input_file = options.input_file
	output_file = options.output_file
	image_only = options.image_only

	if image_only:
		annotate_image(input_file, output_file)
	else:
		annotate_video(input_file, output_file)
