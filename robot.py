#!/usr/bin/env python

import rospy, json, os, random, math
import numpy as np

from std_msgs.msg      import Bool
from sensor_msgs.msg   import LaserScan
from nav_msgs.msg      import OccupancyGrid
from geometry_msgs.msg import PoseArray

from map_utils         import Map
from helper_functions  import *

from threading         import Lock
from sklearn.neighbors import NearestNeighbors

radians_in_circle = math.pi * 2

x = None

terminalStart = False

pose_publisher  = rospy.Publisher('/particlecloud',    PoseArray,     queue_size = 1)

class Particle():
	def __init__(self, map_data, particle_num):#, move_list):
		self.x = random.randrange(map_data.width)
		self.y = random.randrange(map_data.height)
		self.theta = random.uniform(0, radians_in_circle)
		while map_data.grid[self.y][self.x] == 1.:# or self.willCrash(move_list, map_data):
			self.x = random.randrange(map_data.width)
			self.y = random.randrange(map_data.height)

		
		self.weight = 1/float(particle_num) #must be float
		self.set_pose()

	def set_pose(self):
		self.pose = get_pose(self.x, self.y, self.theta)

	def willCrash(self, moveList, map_data):
		x, y, theta = self.x, self.y, self.theta
		for degrees, distance in moveList:
			theta += degrees
			y += (math.sin(theta) * distance)
			x += (math.cos(theta) * distance)
			x, y = int(x), int(y)
			if x < 0 or y < 0 or x >= map_data.height or y >= map_data.width or map_data.grid[x][y] == 1.:
				return True

		return False

	def move(self, move, width, height):
		degree   = move[0]
		distance = move[1]
		self.theta += math.radians(degree)
		self.x += math.cos(self.theta)
		self.y += math.sin(self.theta)
		self.x = sorted((0, self.x, width))[1]
		self.y = sorted((0, self.y, height))[1]
		self.x = int(self.x)
		self.y = int(self.y)
		self.pose = get_pose(self.x, self.y, self.theta)


	def normalize(self, factor):
		self.weight /= factor


	def noise(self, x, y, theta):
		self.x += random.gauss(0, x)
		self.y += random.gauss(0, y)
		self.theta += random.gauss(0, theta)
		self.pose = get_pose(self.x, self.y, self.theta)

	def adjustProbabilities(self, scan, field):
		height = field.height
		width = field.width

		theta = self.theta
		weight = self.weight
		for r in scan.ranges:
			x = int(math.cos(theta) * r + self.x)
			y = int(math.sin(theta) * r + self.y)
			if x < 0 or y < 0 or x >= width or y >= height:
				continue
			weight += field.grid[y][x]
			theta += scan.angle_increment

		self.weight = weight
		return weight



class Robot():
	def __init__(self):
		#rospy.wait_for_service('/map')
		self.config          = self.read_config()
		
		self.field_publisher = rospy.Publisher('/likelihood_field', OccupancyGrid, queue_size = 1, latch = True)
		self.lock = Lock()
		self.laserData = None
		self.firstTime = True
		

		moveList      = self.config['move_list']
		self.moveList = []

		for degree, distance, amount in moveList:
			if degree % 360. != 0.:
				self.moveList.append([degree, 0])
			for i in range(amount):
				self.moveList.append([0, distance])
		self.moveList.reverse()


		rospy.Subscriber("/map", OccupancyGrid, self.onMapRead)
		
		
	def read_config(self):
		if terminalStart:
			config_file_path = os.path.join(
				os.path.dirname(os.path.realpath(__file__)),
				'configuration.json'
			)
		else:
			config_file_path = os.path.join(
				os.path.dirname(os.path.realpath('__file__')),
				'configuration.json'
			)
		with open(config_file_path) as config_file:
			return json.load(config_file)


	def onLaserRead(self, scan):
		
		self.lock.acquire()
		
		if scan_equals(scan, self.laserData):
			#print 'releasing'
			self.lock.release()
			return
		print 'got lock'
		self.laserData = scan
		
		#adjusts weights
		new_weight = sum([p.adjustProbabilities(scan, self.field) for p in self.particles])
		#for p in self.particles:
		#	p.normalize(new_weight) 
		
		#picks new particles
		particleValues = [weighted_choice(self.particles, new_weight) for i in range(len(self.particles))]
		for clump, p in zip(particleValues, self.particles):
			x, y, theta = clump
			p.x = x
			p.y = y
			p.theta = theta
			p.set_pose()

		move = self.moveList.pop()
		move_function(move[0], move[1])
		for p in self.particles:
			p.move(move, self.field.width, self.field.height)

		#adds noise
		if self.firstTime:
			for p in self.particles:
				p.noise(self.config['first_move_sigma_x'], self.config['first_move_sigma_y'], self.config['first_move_sigma_angle'])
			self.firstTime = False
		else:
			for p in self.particles:
				p.noise(self.config['resample_sigma_x'], self.config['resample_sigma_y'], self.config['resample_sigma_angle'])


		

		#move particles
		if len(self.moveList) == 0:
			self.exit()

		print 'releasing'
		self.lock.release()
		
	def exit(self):
		print 'done'
		pass

	def onMapRead(self, data):
		self.data = data
		self.map = Map(data)
		particle_num = self.config['num_particles']
		#moveList = [(math.radians(degrees), distance) for degrees, distance in self.moveList]
		self.particles = [Particle(self.map, particle_num) for i in range(particle_num)]

		self.setupField()
		publishPoseArray(self.particles)

		self.laser = rospy.Subscriber("/base_scan", LaserScan, self.onLaserRead)

		#if len(self.moveList) == 0:
		#	return

		#initial_move = self.moveList.pop()
		#move_function(initial_move[0], initial_move[1])

		#while not rospy.is_shutdown() and len(self.moveList > 0):




	

	def setupField(self):
		self.field = Map(self.data)
		obstacles, free = self.splitMap(self.map.grid)
		obstacles = np.array(obstacles)
		free = np.array(free)
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(obstacles)
		distances, indices = nbrs.kneighbors(free)

		stdv = self.config['laser_sigma_hit']
		for (h,w), distance in zip(free, distances):
			val = pdf(stdv, 0, distance)
			self.field.set_cell(h, w, val)
		for (h,w) in obstacles:
			self.field.set_cell(h, w, 0)

		self.field_publisher.publish(self.field.to_message())



	def splitMap(self, grid):
		obstacles = []
		free = []
		for h in range(self.map.height):
			for w in range(self.map.width):
				x, y = self.map.cell_position(h,w)
				if grid[h][w] == 1. and isNotSurrounded(h,w,grid, self.map.height, self.map.width):
					obstacles.append([x,y])
				else:
					free.append([x,y])
		return obstacles, free
	
def isNotSurrounded(h, w, grid, height, width):
	if h - 1 >= 0:
		if w - 1 >= 0 and grid[h-1][w-1] != 1.:
			return True
		if grid[h-1][w] != 1.:
			return True
		if w + 1 < width and grid[h-1][w+1] != 1.:
			return True

	if w - 1 >= 0 and grid[h][w-1] != 1.:
		return True
	if w + 1 < width and grid[h][w+1] != 1.:
		return True

	if h + 1 < height:
		if w - 1 >= 0 and grid[h+1][w-1] != 1.:
			return True
		if grid[h+1][w] != 1.:
			return True
		if w + 1 < width and grid[h+1][w+1] != 1.:
			return True
	return False

def weighted_choice(choices, total):
   r = random.uniform(0, total)
   upto = 0
   for c in choices:
      if upto + c.weight >= r:
         return (c.x, c.y, c.theta)
      upto += c.weight
   return (c.x, c.y, c.theta)
   #assert False, "Shouldn't get here"

def pdf(stdv, x, mu):
	top = x - mu
	top = top * top * -1.
	bottom = 2. * stdv * stdv
	e = math.e ** (top / bottom)
	total = e / stdv
	total = total / math.sqrt(2. * math.pi)
	return total

def publishPoseArray(particles):
	pose_array = PoseArray()
	pose_array.header.stamp = rospy.Time.now()
	pose_array.header.frame_id='map'
	pose_array.poses = [p.pose for p in particles]
	pose_publisher.publish(pose_array)

def scan_equals(s1, s2):
	if s1 == None or s2 == None:
		return False
	if s1.angle_min != s2.angle_min:
		return False
	if s1.angle_max != s2.angle_max:
		return False
	if s1.angle_increment != s2.angle_increment:
		return False
	if s1.time_increment != s2.time_increment:
		return False
	if s1.scan_time != s2.scan_time:
		return False
	if s1.range_min != s2.range_min:
		return False
	if s1.range_max != s2.range_max:
		return False
	if s1.ranges != s2.ranges:
		return False
	if s1.intensities != s2.intensities:
		return False
	return True

if __name__ == '__main__':
	rospy.init_node('robot', anonymous = True)
	rospy.sleep(2)
	terminalStart = True

	r = Robot()
	rospy.spin()
	