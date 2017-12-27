
'''
AUTOMATE OPENCV TRAINING
Jeff Thompson | 2014 | www.jeffreythompson.org

A utility to automate the training of object detection
"cascade" files with various settings.

1. 	Count # of positive and negative images in txt files
2. 	Create negative file list of required length
3. 	Create vector files with specified # of negative images
4. 	Iterate all settings:
		- create cascade output directory (name clearly)
		- run training

This might take quite a long time (like days or longer) so
be patient!

Created during a residency generously supported by Impakt.nl

'''

import os 				# for creating directories, etc
import itertools		# for creating combinations of settings
import subprocess		# for running command line OpenCV commands
import time				# keep track of how long each combination takes


## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##
#  VARIABLES

# setup variables
object_to_detect = 	'PriceTag'					        # what are you detecting? auto-names files for clarity
collection_filename = 	'PriceTage_Collection.txt'				# file with bounded objects
negative_filename = 	'NegativeImages.txt'					# large file listing negative images
memory_allocation = 	4000							# how much RAM to use (in MB, should be no more than 1/2 total RAM)
log_filename =		'Log_' + object_to_detect + '.csv'	        	# record settings and results to file

# settings to iterate
pos = 			  [ 1, 5, 22 ] 			# how many positive images to use (will use in order of filename)
stages = 		  [ 10, 20 ]			# how many stages to run (smaller = faster but less accurate)
neg = 			  [ 100, 200 ]			# how many negative images to use
accept_rate = 	          [ 0.95, 0.99 ]		# acceptance % (higher = more accurate but MUCH slower to train)
width = 		  [ 120 ]			# output size for vector and training
height = 		  [ 80 ]

use_bg_color =	  False
bgColor = 		  0
bgThresh = 		  0

is_symmetrical =  False					# is the object being used symmetrical?

# set paths for output
local_path = 			os.path.dirname(os.path.realpath(__file__))		# path to script

# folders will be created below if they don't exist
collection_path = 		local_path + '/CollectionFiles_' + object_to_detect + '/'
negative_path = 		local_path + '/NegativeImageLists_' + object_to_detect + '/'
vector_path = 			local_path + '/VectorFiles_' + object_to_detect + '/'
cascade_path = 			local_path + '/CascadeFiles_' + object_to_detect + '/'


## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##
#  FUNCTIONS

# breaks apart the collection file into smaller pieces
# to run with fewer positive images
def create_collection_file(output_filename, num_pos):
	with open(collection_filename) as input_file:
		with open(output_filename, 'w') as output_file:
			for i, line in enumerate(input_file):
				
				# when we get to enough images, stop
				if i == num_pos:
					break
				output_file.write('../' + line)


# create a file of negative images at a specified length
def create_negative_file(output_filename, n):
	with open(negative_filename) as input_file:
		with open(output_filename, 'w') as output_file:
			for i, line in enumerate(input_file):
				
				# when we get to enough images, stop
				if i == n:
					break
				output_file.write('../' + line)


# create a vector file using the 'createsamples' program
# argument is number of negeative images to use
def create_vector_file(collection, negative_file, vector_file, num_pos, num_neg, w, h):
	commands = [ 
		'opencv_createsamples',
		 '-info', collection,
		 '-bg', negative_file,
		 '-vec', vector_file,
		 '-num', str(num_pos),
		 '-w', str(w),
		 '-h', str(h)
	]
	subprocess.check_output(commands)


# train OpenCV using OpenCV's 'trainsamples' program
def train_cascade(cascade_dir, num_stages, num_pos, num_neg, accept, w, h, vector_file, negative_file):
	
	print '- stages:       ', num_stages
	print '- pos images:   ', num_pos
	print '- neg images:   ', num_neg
	print '- accept rate:  ', accept
	print '- dims:         ', w, 'x', h

	# create output directory
	make_dir(cascade_dir)

	# set commands for training
	commands = [
		'opencv_traincascade',
		'-data', cascade_dir,
		'-vec', vector_file,
		'-bg', negative_file,
		'-numPos', str(num_pos),
		'-numNeg', str(num_neg),
		'-numStages', str(num_stages),
		'-mem', str(memory_allocation),
		'-maxHitRate', str(accept),
		'-w', str(w),
		'-h', str(h),
	]

	# if object isn't symmetrical, add to list of commands
	if not is_symmetrical:
		commands.append('-nosym')

	# use a background color?
	if use_bg_color:
		commands.extend( [ '-bgColor', str(bgColor), '-bgThresh', str(bgThresh) ] )

	# do it!
	subprocess.check_output(commands)


# create a directory, if it doesn't already exist
def make_dir(dir):
	if os.path.exists(dir):
		return False
	os.makedirs(dir)
	return True

# format a time in seconds into the largest unit
# ie 59 seconds will be returned in seconds, but 60 as 1 minute
def format_time_in_largest_unit(seconds, decimal_places):
	minutes = seconds / 60.0
	hours =   minutes / 60.0
	
	# probably a more elegant way to do this, but whatever :)
	if hours >= 1.0:
		out_time = hours
		unit = 'hours'
	elif minutes >= 1.0:
		out_time = minutes
		unit = 'minutes'
	else:
		out_time = seconds
		unit = 'seconds'
	
	# output in formatted string with N decimal places
	return '{0:.{1}f}'.format(out_time, decimal_places) + ' ' + unit


## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##
# MAIN PROGRAM

# let's go!
print '\n' + 'AUTOMATING OPENCV TRAINING' + '\n'

# make directories as needed
print 'MAKING DIRECTORIES (if needed)...'
if make_dir(collection_path):
	print '-', collection_path
if make_dir(negative_path):
	print '-', negative_path
if make_dir(vector_path):
	print '-', vector_path
if make_dir(cascade_path):
	print '-', cascade_path

# write header to log file, if it doesn't exist
if not os.path.exists(log_filename):
	with open(log_filename, 'w') as log:
		log.write('stages,num_pos,num_neg,accept_rate,width,height,processing_time_sec,cascade_filename' + '\n')

# get all combinations of settings (via: http://stackoverflow.com/a/798893/1167783)
s = [ stages, pos, neg, accept_rate, width, height ]
combinations = list(itertools.product(*s))

# count total number of positive/negative images
print '\n' + 'COUNTING INPUT IMAGES...'
total_pos = 0
# with open(collection_filename) as collection:
# 	for line in collection:
# 		d = line.split(' ')				# data is separated by spaces
#                 print(d)
# 		total_pos += int(d[1])			# number of bounding boxes is 2nd item listed
# print 'positive images: ', total_pos
with open(collection_filename) as collection:
	for line in collection:
		total_pos += 1			# number of bounding boxes is 2nd item listed
print 'positive images: ', total_pos

total_neg = 0
with open(negative_filename) as negative:
	for line in negative:
		total_neg += 1 					# just count the lines!
print 'negative images: ', total_neg


# train that sucker!
for i, combo in enumerate(combinations):

	print '\n' + ('- ' * 5) + '\n\n' + str(i+1) + '/' + str(len(combinations)) + '\n'
	start_time = time.time()

	# get settings from combination - makes things a little semantically easier :)
	num_stages = combo[0]
	num_pos =	 combo[1]
	num_neg = 	 combo[2]
	accept =  	 combo[3]
	w = 		 combo[4]
	h = 		 combo[5]

	# split collection file into specified length
	# if we don't have enough positive images, throw an error and skip
	# otherwise, generate the file
	print 'creating collection file...'
	if num_pos > total_pos:
		print "  can't create collection file with", num_pos, "images (there are only", total_pos, "available)!" + "\n"
		continue
	print (str(num_pos) + ' images:').ljust(12),
	collection_file = collection_path + object_to_detect + '_Collection-' + str(num_pos) + 'Images.txt'
	create_collection_file(collection_file, num_pos)
	print 'done!'

	# same as above: create negative image files of specified lengths
	print '\n' + 'creating negative image file...'
	if num_neg > total_neg:
		print "\n" + "can't create", num_neg, "negative files (there are only", total_neg, "available)!" + "\n"
		continue
	print (str(num_neg) + ' images:').ljust(12),
	negative_file = negative_path + 'NegativeImages_' + str(num_neg) + '.txt'
	create_negative_file(negative_file, num_neg)
	print 'done!'

	# create vector file with positive/negative images
	print '\n' + 'creating vector file...'
	print (str(num_pos) + 'pos/' + str(num_neg) + 'neg:').ljust(12),
	vector_file = vector_path + 'Vector_' + str(num_pos) + 'pos-' + str(num_neg) + 'neg.vec'
	create_vector_file(collection_file, negative_file, vector_file, num_pos, num_neg, w, h)
	print 'done!'

	# finally, train it!
	print '\n' + 'training cascade...'
	cascade_filename = cascade_path + 'CascadeOutput_' + str(num_stages) + 'Stages-' + str(num_pos) + 'Pos-' + str(num_neg) + 'Neg-' + str(accept) + 'AcceptRate-' + str(w) + 'W-' + str(h) + 'H'
	train_cascade(cascade_filename, num_stages, num_pos, num_neg, accept, w, h, vector_file, negative_file)

	# how long has elapsed? argument is time in seconds and number of decimal places
	end_time = time.time()
	processing_time = format_time_in_largest_unit(end_time - start_time, 3)
	print '\n' + 'processing time: ' + processing_time

	# save details to log file
	with open(log_filename, 'a') as log:
		log.write(str(num_stages) + ',' + str(num_pos) + ',' + str(num_neg) + ',' + str(accept) + ',' + str(w) + ',' + str(h) + ',' + str(end_time - start_time) + ',' + cascade_filename + '\n')


# all done!
print '\n' + ('- ' * 5) + '\n\n' + 'ALL DONE!' + ('\n' * 3)

