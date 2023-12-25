import random
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm

ONE_MAX_LENGTH = 50

POPULATION_SIZE = 100
P_CROSSOVER = .9
P_MUTATION = 0.02
MAX_GENERATIONS = 100


RANDOM_SEED = 42
random.seed(RANDOM_SEED)



class Individual:
	def __init__(self, chromosomes):
		self.chromosomes = chromosomes


def fitness(chromosomes):
	def f(x):
		return -(x - 11) ** 2

	return f(sum(chromosomes))


def generateIndividual():
	#return Individual([0 if random.random() < 0.01 else 1 for _ in range(ONE_MAX_LENGTH)])
	return Individual([random.randint(0, 1) for _ in range(ONE_MAX_LENGTH)])


def generatePopulation():
	return [generateIndividual() for _ in range(POPULATION_SIZE)]


def selection(population, size=None):
	size = len(population) if size is None else size
	offspring = []
	for _ in range(size):
		offspring.append(
			max(
				np.random.choice(population, size=3, replace=False).tolist(),
				key=lambda x: fitness(x.chromosomes)
			)
		)

	return offspring


def mutation(individual):
	for i in range(ONE_MAX_LENGTH):
		if random.random() < P_MUTATION:
			individual.chromosomes[i] = 1 - individual.chromosomes[i]


def crossover(population):
	new_generation = []
	for ind1, ind2 in zip(population[::2], population[1::2]):
		if random.random() < P_CROSSOVER:
			childs = list(map(copy.deepcopy, [ind1, ind1]))
			for child in childs:
				for i in range(ONE_MAX_LENGTH):
					if random.random() < .5:
						child.chromosomes[i] = ind2.chromosomes[i]

				mutation(child)
			new_generation.extend(childs)
		else:
			new_generation.extend([ind1, ind2])

	random.shuffle(new_generation)
	return new_generation


def convert_arr_to_decimal(arr):
	return int('0b' + ''.join([str(x) for x in arr]), 2)


population = generatePopulation()

best, mean = [], []
#for _ in tqdm(range(MAX_GENERATIONS)):
for _ in range(MAX_GENERATIONS):
	best.append(max([fitness(ind.chromosomes) for ind in population]))
	mean.append(sum([fitness(ind.chromosomes) for ind in population]) / len(population))
	population = crossover(selection(population))

	if _ % 100 == 0:
		print('Generation {generation}\tThe best value is {the_best_val}\n\tThe mean values is {mean_val}\n--------------------------------\n'.format(
				the_best_val=best[-1],
				mean_val=mean[-1],
				generation=_ + 1
			))

best.append(max([fitness(ind.chromosomes) for ind in population]))
mean.append(sum([fitness(ind.chromosomes) for ind in population]) / len(population))

plt.title('Genetic algorithm | Sureya Khalilova')
plt.xlabel('Epochs')
plt.ylabel('Fitness')
plt.plot(range(MAX_GENERATIONS + 1), best, label='the best', c='r')
plt.plot(range(MAX_GENERATIONS + 1), mean, label='mean', c='b')
plt.legend()



the_best_ind = population[[fitness(ind.chromosomes) for ind in population].index(
		max([fitness(ind.chromosomes) for ind in population])
	)]

print('\tThe best value is {the_best_val}\n\tThe mean values is {mean_val}\nThe best individual is {the_best_ind}'.format(
		the_best_val=best[-1],
		mean_val=mean[-1],
		the_best_ind=the_best_ind.chromosomes
	))

plt.savefig('ga.png')

plt.show()

