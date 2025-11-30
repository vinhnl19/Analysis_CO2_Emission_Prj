import numpy as np

DE_POP_SIZE = 100
DE_EVAL_LIMIT = 20000
DE_F_SCALE = 0.8
DE_CROSSOVER_RATE = 0.7

def de_optimizer_changes(
        feature_selection,
        fixed_features,
        predict_fn,
        co2_target,
):
    np.random.seed(42)

    dims = len(feature_selection)

    lower_bound = []
    upper_bound = []

    feature_names = []

    for item in feature_selection:
        feature_names.append(item["feature"])
        lower_bound.append(item["min_pct"])
        upper_bound.append(item["max_pct"])

    

    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    diff = upper_bound - lower_bound

    pop = lower_bound + diff * np.random.rand(DE_POP_SIZE, dims)

    def indiv_to_dict(indiv):
        return {feature_names[i]: indiv[i] for i in range(dims)}
    
    def evaluate(indiv):
        indiv_dict = indiv_to_dict(indiv)
        pred, x = predict_fn(indiv_dict, fixed_features)

        if pred > co2_target:
            error = 10 * (pred - co2_target)
        else:
            error = co2_target - pred
        
        fitness = error
        return fitness, pred, x
    
    fitness = []
    preds = []
    xs = []
    for ind in pop:
        fit, pred, x = evaluate(ind)
        fitness.append(fit)
        preds.append(pred)
        xs.append(x)

    fitness = np.array(fitness)
    preds = np.array(preds)
    xs = np.array(xs)
    eval_count = DE_POP_SIZE

    best_idx = np.argmin(fitness)
    best_ind = pop[best_idx]
    best_fit = fitness[best_idx]
    best_pred = preds[best_idx]
    best_x = xs[best_idx]

    max_gens = max(1, DE_EVAL_LIMIT // DE_POP_SIZE)

    for gen in range(max_gens):
        if eval_count >= DE_EVAL_LIMIT:
            break
        for i in range(DE_POP_SIZE):

            idxs = [idx for idx in range(DE_POP_SIZE) if idx !=i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + DE_F_SCALE * (b - c)

            mutant = np.clip(mutant, lower_bound, upper_bound)

            cross_points = np.random.rand(dims) < DE_CROSSOVER_RATE
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dims)] = True

            trial = np.where(cross_points, mutant, pop[i])

            trial_fit, trial_pred, trial_x = evaluate(trial)
            eval_count +=1

            if trial_fit < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fit
                preds[i] = trial_pred

                if trial_fit < best_fit:
                    best_idx = i
                    best_ind = trial.copy()
                    best_fit = trial_fit
                    best_pred = trial_pred
                    best_x = trial_x

        if best_pred <= co2_target:
            break

    best_change = indiv_to_dict(best_ind)

    return best_change, best_fit, best_pred, best_x
