import random
import copy
from statistics import mean
import numpy as np
import os
import shutil
from game_models.base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork


class GEGameModel(BaseGameModel):

    model = None

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self,
                               game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.model = ConvolutionalNeuralNetwork(input_shape, action_space).model

    def _predict(self, state):
        if np.random.rand() < 0.02:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])


class GESolver(GEGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ge/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        GEGameModel.__init__(self,
                             game_name,
                             "GE testing",
                             input_shape,
                             action_space,
                             "./output/logs/" + game_name + "/ge/testing/" + self._get_date() + "/",
                             testing_model_path)
        self.model.load_weights(self.model_path)

    def move(self, state):
        return self._predict(state)


class GETrainer(GEGameModel):

    run = 0
    generation = 0
    selection_rate = 0.1
    mutation_rate = 0.01
    population_size = 100
    random_weight_range = 1.0
    parents = int(population_size * selection_rate)

    def __init__(self, game_name, input_shape, action_space):
        GEGameModel.__init__(self,
                             game_name,
                             "GE training",
                             input_shape,
                             action_space,
                             "./output/logs/" + game_name + "/ge/training/"+ self._get_date() + "/",
                             "./output/neural_nets/" + game_name + "/ge/" + self._get_date() + "/model.h5")
        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

    def move(self, state):
        pass

    def genetic_evolution(self, env):
        print "population_size: " + str(self.population_size) +\
              ", mutation_rate: " + str(self.mutation_rate) +\
              ", selection_rate: " + str(self.selection_rate) +\
              ", random_weight_range: " + str(self.random_weight_range)
        population = None

        while True:
            print('{{"metric": "generation", "value": {}}}'.format(self.generation))

            # 1. Selection
            parents = self._strongest_parents(population, env)

            self._save_model(parents)  # Saving main model based on the current best two chromosomes

            # 2. Crossover (Roulette selection)
            pairs = []
            while len(pairs) != self.population_size:
                pairs.append(self._pair(parents))

            # # 2. Crossover (Rank selection)
            # pairs = self._combinations(parents)
            # random.shuffle(pairs)
            # pairs = pairs[:self.population_size]

            base_offsprings = []
            for pair in pairs:
                offsprings = self._crossover(pair[0][0], pair[1][0])
                base_offsprings.append(offsprings[-1])

            # 3. Mutation
            new_population = self._mutation(base_offsprings)
            population = new_population
            self.generation += 1

    def _pair(self, parents):
        total_parents_score = sum([x[1] for x in parents])
        pick = random.uniform(0, total_parents_score)
        pair = [self._roulette_selection(parents, pick), self._roulette_selection(parents, pick)]
        return pair

    def _roulette_selection(self, parents, pick):
        current = 0
        for parent in parents:
            current += parent[1]
            if current > pick:
                return parent
        return random.choice(parents) # Fallback

    def _combinations(self, parents):
        combinations = []
        for i in range(0, len(parents)):
            for j in range(i, len(parents)):
                combinations.append((parents[i], parents[j]))
        return combinations

    def _strongest_parents(self, population, env):
        if population is None:
            population = self._initial_population()
        scores_for_chromosomes = []
        for i in range(0, len(population)):
            chromosome = population[i]
            scores_for_chromosomes.append((chromosome, self._gameplay_for_chromosome(chromosome, env)))

        scores_for_chromosomes.sort(key=lambda x: x[1])
        top_performers = scores_for_chromosomes[-self.parents:]
        top_scores = [x[1] for x in top_performers]
        print('{{"metric": "population", "value": {}}}'.format(mean([x[1] for x in scores_for_chromosomes])))
        print('{{"metric": "top_min", "value": {}}}'.format(min(top_scores)))
        print('{{"metric": "top_avg", "value": {}}}'.format(mean(top_scores)))
        print('{{"metric": "top_max", "value": {}}}'.format(max(top_scores)))
        return top_performers

    def _mutation(self, base_offsprings):
        offsprings = []
        for offspring in base_offsprings:
            offspring_mutation = copy.deepcopy(offspring)

            for a in range(0, len(offspring_mutation)):  # 10
                a_layer = offspring_mutation[a]
                for b in range(0, len(a_layer)):  # 8
                    b_layer = a_layer[b]
                    if not isinstance(b_layer, np.ndarray):
                        if np.random.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate]):
                            offspring_mutation[a][b] = self._random_weight()
                        continue
                    for c in range(0, len(b_layer)):  # 8
                        c_layer = b_layer[c]
                        if not isinstance(c_layer, np.ndarray):
                            if np.random.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate]):
                                offspring_mutation[a][b][c] = self._random_weight()
                            continue
                        for d in range(0, len(c_layer)):  # 4
                            d_layer = c_layer[d]
                            for e in range(0, len(d_layer)):  # 32
                                if np.random.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate]):
                                    offspring_mutation[a][b][c][d][e] = self._random_weight()
            offsprings.append(offspring_mutation)
        return offsprings

    def _crossover(self, x, y):
        offspring_x = x
        offspring_y = y

        for a in range(0, len(offspring_x)):  # 10
            a_layer = offspring_x[a]
            for b in range(0, len(a_layer)):  # 8
                b_layer = a_layer[b]
                if not isinstance(b_layer, np.ndarray):
                    if random.choice([True, False]):
                        offspring_x[a][b] = y[a][b]
                        offspring_y[a][b] = x[a][b]
                    continue
                for c in range(0, len(b_layer)):  # 8
                    c_layer = b_layer[c]
                    if not isinstance(c_layer, np.ndarray):
                        if random.choice([True, False]):
                            offspring_x[a][b][c] = y[a][b][c]
                            offspring_y[a][b][c] = x[a][b][c]
                        continue
                    for d in range(0, len(c_layer)):  # 4
                        d_layer = c_layer[d]
                        for e in range(0, len(d_layer)):  # 32
                            if random.choice([True, False]):
                                offspring_x[a][b][c][d][e] = y[a][b][c][d][e]
                                offspring_y[a][b][c][d][e] = x[a][b][c][d][e]
        return offspring_x, offspring_y

    def _gameplay_for_chromosome(self, chromosome, env):
        self.run += 1
        self.logger.add_run(self.run)

        self.model.set_weights(chromosome)
        state = env.reset()
        score = 0
        while True:
            action = self._predict(state)
            state_next, reward, terminal, info = env.step(action)
            score += np.sign(reward)
            state = state_next
            if terminal:
                self.logger.add_score(score)
                return score

    def _initial_population(self):
        weights = self.model.get_weights()
        chromosomes = []

        for i in range(0, self.population_size):
            chromosome = weights # 1 686 180 params
            for a in range(0, len(weights)): # 10
                a_layer = weights[a]
                for b in range(0, len(a_layer)):  # 8
                    b_layer = a_layer[b]
                    if not isinstance(b_layer, np.ndarray):
                        weights[a][b] = self._random_weight()
                        continue
                    for c in range(0, len(b_layer)):  # 8
                        c_layer = b_layer[c]
                        if not isinstance(c_layer, np.ndarray):
                            weights[a][b][c] = self._random_weight()
                            continue
                        for d in range(0, len(c_layer)):  # 4
                            d_layer = c_layer[d]
                            for e in range(0, len(d_layer)):  # 32
                                weights[a][b][c][d][e] = self._random_weight()
            chromosomes.append(chromosome)
        return chromosomes

    def _random_weight(self):
        return random.uniform(-self.random_weight_range, self.random_weight_range)

    def _save_model(self, parents):
        x = copy.deepcopy(parents[-1][0])
        y = copy.deepcopy(parents[-2][0])
        best_offsprings = self._crossover(x, y)
        self.model.set_weights(best_offsprings[-1])
        self.model.save_weights(self.model_path)
