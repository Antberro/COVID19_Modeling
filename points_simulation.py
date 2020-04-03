import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib
from random import choice
plt.style.use('dark_background')

class Points:
	"""
	Class that supports the creation of multiple points that can be plotted
	and animated using matplotlib. Using the varius step methods, these collections
	of points can move in different ways. Initial distribution of points can also
	be controlled by separate point generator functions.
	"""
	def __init__(self, matrix, width, height, radius, color='blue', dt=0.01, speed=10):
		self.N = matrix.shape[0]
		self.positions = matrix
		self.colors = np.array([color for _ in range(self.N)])
		self.dt = dt
		self.speed = speed
		self.width_bound = width # 10 deal with later
		self.height_bound = height  # 10 deal with later
		self.radius = radius # 0.2
		self.v = self._get_random_velocity()

	def _boundary_detection(self):
		"""
		Detect points that are out of bounds of screen. Returns a column vector
		of 1's or 0's. If the ith element is 1, then the ith point is out of bounds,
		otherwise 0 for inbounds.
		"""
		bounds = np.array([[self.width_bound-self.radius, self.height_bound-self.radius]])
		bounds = np.broadcast_to(bounds, (self.N, 2))
		# create boolean array to find points out of bounds
		upper = np.greater(self.positions, bounds)
		lower = np.less(self.positions, self.radius*np.ones(self.positions.shape))
		compare_upper = upper.any(axis=1, keepdims=True)
		compare_lower = lower.any(axis=1, keepdims=True)
		compare = np.logical_or(compare_upper, compare_lower)
		return np.where(compare == True, 1, 0)

	def _get_random_velocity(self):
		"""
		Generates a velocity vector with random direction
		and constant speed for each point. Returns matrix of 
		veloctity vectors. Used in combination with random_step().
		"""
		# create random matrix v where each row is velocity vector of each point
		v = np.random.uniform(-1, 1, (self.N, 2))
		# turn each vector in v into a unit vector
		mag = v**2
		mag = (mag[:,[0]] + mag[:,[1]])**0.5
		v_unit = v / mag
		# multiply each row in v by some constant speed 
		v_new = self.speed * v_unit
		return v_new
		
	def random_step(self):
		"""
		Updates positions of points after one timestep. Points move
		in random directions at a constant speed.
		"""
		# calculate new positions
		self.positions = self.positions + self.v * self.dt

		# detect any points that are put of bounds
		# negate their original velocities to keep them in bounds
		outofbounds = self._boundary_detection()
		self.positions = self.positions - outofbounds * self.v * self.dt
		
		# generate new random velocities
		self.v = self._get_random_velocity()

### Point Generator Functions ###
# Used to generate (n, 2) matrices that contain the x and y
# values for n points. Different functions generate matrices 
# with points in different distribution. Used as an input 
# parameter for the Points class.
def uniform_point_generator(width, height, spacing=1):
	"""
	Generates a 2d numpy array of uniformly distributed points
	across a grid size of width, height.
	"""
	x, y = [], []
	for i in np.linspace(spacing, width-spacing, int(width // spacing)):
		for j in np.linspace(spacing, height-spacing, int(height // spacing)):
			x.append(i)
			y.append(j)
	x = np.array([x]).T
	y = np.array([y]).T
	return np.hstack((x, y))

def random_point_generator(width, height, n):
	"""
	Generates a 2d numpy array of n uniformly distributed points 
	across a grid size of width, height.
	"""
	x = np.random.uniform(0, width, (n, 1))
	y = np.random.uniform(0, height, (n, 1))
	return np.hstack((x, y))
#########################################

class Simulation:
	"""
	Class used for running a matplotlib animation simulating a 
	collection of points (implemented using the Points class) 
	moving around a matplotlib figure. Takes in a dictionary of
	parameters.
	"""
	def __init__(self, kwargs):
		# unpack args
		self.frames = kwargs['frames'] if 'frames' in kwargs else 200
		self.interval = kwargs['interval'] if 'interval' in kwargs else 80
		self.width = kwargs['width'] if 'width' in kwargs else 10
		self.height = kwargs['height'] if 'height' in kwargs else 10
		self.pointgen = kwargs['pointgen'] if 'pointgen' in kwargs else uniform_point_generator
		self.num_points = kwargs['num_points'] if 'num_points' in kwargs else self.width * self.height
		self.spacing = kwargs['spacing'] if 'spacing' in kwargs else 1
		self.radius = kwargs['radius'] if 'radius' in kwargs else 0.2
		self.color = kwargs['color'] if 'color' in kwargs else 'blue'
		self.dt = kwargs['dt'] if 'dt' in kwargs else 0.01
		self.speed = kwargs['speed'] if 'speed' in kwargs else 10
		self.title = kwargs['title'] if 'title' in kwargs else 'Simulation'

		# define plt objects
		self.fig = plt.figure()
		self.ax = plt.axes(xlim=(0,self.width), ylim=(0,self.height))
		plt.axis('scaled')

		# create collection of points
		matrix = None
		if self.pointgen == uniform_point_generator:
			matrix = self.pointgen(self.width, self.height, self.spacing)
		if self.pointgen == random_point_generator:
			matrix = self.pointgen(self.width, self.height, self.num_points)

		self.point_collection = Points(matrix, self.width, self.height, self.radius, self.color, self.dt, self.speed)
		self.points = [plt.Circle((p[0],p[1]), radius=self.radius) for p in self.point_collection.positions]

		# figure visualization
		self.axis_color = 'white'

	def _sim_init(self):
		for point in self.points:
			self.ax.add_patch(point)
		return self.points

	def _sim_animate(self, i):
		self.point_collection.random_step()
		for j in range(len(self.points)):
			new_x = self.point_collection.positions[j,0]
			new_y = self.point_collection.positions[j,1]
			self.points[j].center = (new_x, new_y)
			self.points[j].set_color(self.point_collection.colors[j])
		return self.points

	def run(self, show=True, saveas=None):
		anim = FuncAnimation(self.fig, self._sim_animate, init_func=self._sim_init, frames=self.frames, interval=self.interval, blit=True, repeat=False)
		if show:
			self.ax.set_title(self.title, loc='center', color=self.axis_color)
			self.ax.set_xticks([], [])
			self.ax.set_yticks([], [])
			self.ax.spines['bottom'].set_color(self.axis_color)
			self.ax.spines['top'].set_color(self.axis_color) 
			self.ax.spines['right'].set_color(self.axis_color)
			self.ax.spines['left'].set_color(self.axis_color)
			plt.show()
		if saveas:
			pass  # add later

# examples of a default params dictionaries
# for different inital point distributions
DEFAULT_UNIFORM_PARAMS = {
	'frames': 200,
	'interval': 80,
	'width': 10,
	'height': 10,
	'pointgen': uniform_point_generator,
	'spacing': 1,
	'radius': 0.2,
	'color': 'blue',
	'dt': 0.01,
	'speed': 10,
	'title': 'Initial Uniform Distribution'
}
DEFAULT_RANDOM_PARAMS = {
	'frames': 200,
	'interval': 80,
	'width': 10,
	'height': 10,
	'pointgen': random_point_generator,
	'num_points': 10,
	'radius': 0.2,
	'color': 'blue',
	'dt': 0.01,
	'speed': 10,
	'title': 'Initial Random Distribution'
}