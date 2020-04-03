import epidemic_simulation as esim 
import points_simulation as psim 

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

sim = esim.EpidemicSimulation(settings, params)
sim.run()
sim.plot_growth()
