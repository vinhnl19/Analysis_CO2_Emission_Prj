import random
import numpy as np

GA_POP_SIZE = 40
GA_GENERATIONS = 20
GA_MUTATION_RATE = 0.15
GA_CROSSOVER_RATE = 0.8

def ga_optimize_changes(
        feature_selection,
        predict_fn,
        predicted_co2, 
        co2_target
):
    feature_names = list(feature_selection.keys())
    n = len(feature_names)

    bounds = {}
    for f, info in feature_selection.items():
        bounds[f] = (-info["max_change_pct"], info["max_change_pct"])


    def init_individual():
        indiv = {}
        for f in feature_names:
            low, high = bounds[f]
            indiv[f] = random.uniform(low, high)

        return indiv
    
    def evaluate(indiv):
        new_pred = predict_fn(indiv)
        error = abs(new_pred - co2_target)

        cost_penalty = 0
        for f, delta in indiv.items():
            cost_level = feature_selection[f]["cost"]
            cost_penalty += cost_level * (abs(delta) / bounds[f][1])

        fitness_value = error + 0.1 * cost_penalty
        return fitness_value, new_pred
    
    def crossover(p1, p2):
        if random.random() > GA_CROSSOVER_RATE:
            return p1.copy(), p2.copy()
        
        c1, c2 = {}, {}
        for f in feature_names:
            if random.random() < 0.5:
                c1[f] = p1[f]
                c2[f] = p2[f]
            else:
                c1[f] = p2[f]
                c2[f] = p1[f]
        return c1, c2
    
    def mutate(indiv):
        for f in feature_names:
            if random.random() < GA_MUTATION_RATE:
                low, high = bounds[f]
                indiv[f] += random.uniform(-0.1 * high, 0.1 * high)

                indiv[f] = max(low, min(high, indiv[f]))

        return indiv
    
    def select(pop):
        k = 3
        candidates = random.sample(pop, k)
        candidates.sort(key=lambda x: x["fitness"])
        return candidates[0]
    
    population = []

    for _ in range(GA_POP_SIZE):
        indiv = init_individual()
        indiv_fitness, pred_val = evaluate(indiv=indiv)
        population.append({
            "indiv": indiv, 
            "fitness": indiv_fitness,
            "predicted": pred_val})

    for gen in range(GA_GENERATIONS):
        new_pop = []

        while len(new_pop) < GA_POP_SIZE:
            p1 = select(population)
            p2 = select(population)

            child1_indiv, child2_indiv = crossover(p1["indiv"], p2["indiv"])

            child1_indiv = mutate(child1_indiv)
            child2_indiv = mutate(child2_indiv)

            fit1, pred1 = evaluate(child1_indiv)
            fit2, pred2 = evaluate(child2_indiv)

            new_pop.append({
                "indiv": child1_indiv, 
                "fitness": fit1,
                "predicted": pred1})
            new_pop.append({
                "indiv": child2_indiv, 
                "fitness": fit2,
                "predicted": pred2})

        population = new_pop

    population.sort(key=lambda x: x["fitness"])
    best = population[0]

    return best["indiv"], best["fitness"], best["predicted"]