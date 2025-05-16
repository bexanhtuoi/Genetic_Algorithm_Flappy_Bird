import numpy as np

class GA:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best_weights = None
        self.best_fitness = -np.inf

    def selection_parents(self, fitness, weights):
        idx = np.argsort(fitness)[-2:]
        return weights[idx[0]], weights[idx[1]]
    
    def crossover(self, parent1, parent2):
        alpha = 0.5
        child = (alpha * parent1) + ((1 - alpha) * parent2)
        return child

    def mutate(self, weights):
        mutation_prob = self.mutation_rate
        if np.random.rand() < self.mutation_rate:
            weights += (np.random.uniform(-0.1, 0.1, size=weights.shape) * np.random.randint(0, 2, size=weights.shape))
        return weights

    def fit(self, fitness, weights):
        p1, p2 = self.selection_parents(fitness, weights)
        child = self.crossover(p1, p2)
        best_indices = np.argmax(fitness)
        best = weights[best_indices]
        best_fitness = fitness[best_indices]

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_weights = best.copy()

        new_population = []

        if self.best_weights is not None:
            for _ in range(self.population_size//10): # Lấy 10% Bird tốt nhất
                new_population.append(self.best_weights.copy())

        for _ in range(self.population_size//10): # Lấy 10% Bird tốt nhất được đột biến
            mutate_best_bird = self.mutate(self.best_weights.copy()) 
            new_population.append(mutate_best_bird)

        for _ in range(self.population_size//10): # Lấy 10% Bird lai ghép giữa cha, mẹ
            new_population.append(child)

        for _ in range(self.population_size//10): # Lấy 10% Bird lai ghép giữa cha, mẹ và đột biến
            mutate_prarents_bird = self.mutate(child)
            new_population.append(mutate_prarents_bird)

        for _ in range(self.population_size//10): # Lấy 10% Bird cha
            new_population.append(p1.copy())

        for _ in range(self.population_size//10): # Lấy 10% Bird mẹ
            new_population.append(p2.copy())

        for _ in range(self.population_size//10): # Lấy 10% Bird cha đột biến
            mutate_parent_bird = self.mutate(p1.copy())
            new_population.append(mutate_parent_bird)

        for _ in range(self.population_size//10): # Lấy 10% Bird mẹ đột biến
            mutate_parent_bird = self.mutate(p2.copy())
            new_population.append(mutate_parent_bird)

        while len(new_population) < self.population_size: # Lấy Bird còn lại ngẫu nhiên
            random_bird = np.random.normal(0, 0.1, size=weights.shape[1])
            new_population.append(random_bird)

        return np.array(new_population)     
