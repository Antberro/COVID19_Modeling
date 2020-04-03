import points_simulation as psim 
import matplotlib.pyplot as plt
from random import sample
from math import ceil
import numpy as np 

class EpidemicSimulation(psim.Simulation):
	def __init__(self, settings, params):
		super().__init__(settings)
		self.params = params
		
		# unpack parameter arguments
		self.init_infected = params['init_infected']
		self.trans_radius = params['trans_radius']
		self.p_infected = params['p_infected']
		self.p_removed = params['p_removed']
		
		# other parameters
		self.total_pop = self.point_collection.positions.shape[0]
		self.healthy = set()
		self.infected = set()
		self.removed = set()
		self.num_healthy = 0
		self.num_infected = 0
		self.num_removed = 0

		# for visualization
		self.healthy_color = 'blue'
		self.infected_color = 'red'
		self.removed_color = 'grey'

		# for plotting
		self.num_healthy_data = []
		self.num_infected_data = []
		self.num_removed_data = []

		# set up initial sets
		self._initiate_patient_zero()

	def _initiate_patient_zero(self):
		"""
		Initialized the initial patient zeros. Determines the 
		initial points/patients that are healthy and infected.
		Called once in the class's __init__ function.
		"""
		# initialize healthy set
		self.healthy = set([i for i in range(self.total_pop)])
		self.num_healthy = self.total_pop

		# initialize infected set
		for i in sample(self.healthy, self.init_infected):
			# remove from healthy
			self.healthy.remove(i)
			self.num_healthy -= 1
			# add to infected
			self.infected.add(i)
			self.num_infected += 1
			self.point_collection.colors[i] = self.infected_color

		# update data lists
		self.num_healthy_data.append(self.num_healthy)
		self.num_infected_data.append(self.num_infected)
		self.num_removed_data.append(self.num_removed)
		
	def _get_people_within_radius(self, center):
		"""
		Returns a set of indexes of the points within a circle 
		of the transmission radius centered at the index center.
		"""
		cx, cy = self.point_collection.positions[center,:]
		dist = lambda x,y: ((cx-x)**2 + (cy-y)**2)**0.5
		group = set()
		for i in range(self.total_pop):
			if i != center:
				x, y = self.point_collection.positions[i,:]
				if dist(x,y) < self.trans_radius:
					group.add(i)
		return group

	def _spread_disease(self):
		# for each infected person, determine the set of 
		# points within trans_radius away. A percentage of them
		# will become infected
		potential_infected = set()
		for i in self.infected:
			potential_infected.update(self._get_people_within_radius(i))
		n_infected = ceil(self.p_infected * len(potential_infected))
		if n_infected <= len(potential_infected):
			newly_infected = sample(potential_infected, n_infected)
		else:
			newly_infected = set()

		# update sets/colors/counts of those healthy, infected, removed
		if newly_infected:
			for i in newly_infected:
				if i in self.healthy:
					# remove from healthy
					self.healthy.remove(i)
					self.num_healthy -= 1
					# add to infected
					self.infected.add(i)
					self.num_infected += 1
					self.point_collection.colors[i] = self.infected_color

	def _remove_dead_and_recovered(self):
		# for each infected person, a percentage of them
		# will be removed
		n_removed = ceil(self.p_removed * self.num_infected)
		if n_removed <= self.num_infected:
			newly_removed = sample(self.infected, n_removed)
		else:
			newly_removed = set()

		# update sets/colors/counts of those healthy, infected, removed
		if newly_removed:
			for i in newly_removed:
				# remove from infected
				self.infected.remove(i)
				self.num_infected -= 1
				# add to removed
				self.removed.add(i)
				self.num_removed += 1
				self.point_collection.colors[i] = self.removed_color
				
	def _sim_animate(self, i):
		#self.point_collection.biased_step('up')
		self.point_collection.random_step()
		# determine who is healthy, infected, removed
		self._spread_disease()
		self._remove_dead_and_recovered()

		# update data_lists
		if self.num_infected > 0:
			self.num_healthy_data.append(self.num_healthy)
			self.num_infected_data.append(self.num_infected)
			self.num_removed_data.append(self.num_removed)

		# update points
		for j in range(len(self.points)):
			new_x = self.point_collection.positions[j,0]
			new_y = self.point_collection.positions[j,1]
			self.points[j].center = (new_x, new_y)
			self.points[j].set_color(self.point_collection.colors[j])
		return self.points

	def plot_growth(self, show=True, saveas=None):
		"""
		Plots the total number of healthy, infected and removed during 
		the simulation.
		"""
		# plot growth as lines
		fig2, out_ax = plt.subplots(1,2)
		t = [i for i in range(len(self.num_infected_data))]
		out_ax[0].set_title("Epidemic Growth")
		out_ax[0].set_xlabel("Time Steps")
		out_ax[0].set_ylabel("Number of People")
		out_ax[0].plot(t, self.num_healthy_data, 'blue')
		out_ax[0].plot(t, self.num_infected_data,'red')
		out_ax[0].plot(t, self.num_removed_data, 'grey')

		# plot growth as stackplot
		data = np.vstack([self.num_healthy_data, self.num_infected_data, self.num_removed_data])
		out_ax[1].stackplot(t, data, colors=[self.healthy_color, self.infected_color, self.removed_color], alpha=0.6)
		
		if show:
			plt.show()
		if saveas:
			pass

		


	


settings = {
	'frames': 2000,
	'interval': 5,
	'width': 40,
	'height': 40,
	'pointgen': psim.random_point_generator,
	'num_points': 400,
	'spacing': 0.5,
	'radius': 0.2,
	'color': 'blue',
	'dt': 0.1,
	'speed': 10,
	'title': 'Ahhhhhh Corona!!!'
}
params = {
	'init_infected': 5,
	'trans_radius': 10,
	'p_infected': 0.3,
	'p_removed': 0.2,
}


		