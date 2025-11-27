import numpy as np
ES_POP_SIZE = 100
ES_EVAL_LIMIT = 20000
ES_SIGMA_INIT = 5.0
ES_C_INC = 1.2
ES_C_DEC = 0.6
def es_optimize_changes(
        feature_selection,
        predict_fn,
        predicted_co2,
        co2_target
):
    np.random.seed(42)
    
    dims = len(feature_selection)

    lower_bound = []
    upper_bound = []
    max_change = []
    cost_levels = []

    for f in feature_selection:
        max_pct = feature_selection[f]["max_change_pct"]
        cost_levels.append(feature_selection[f]["cost"])
        max_change.append(max_pct)
        lower_bound.append(-max_pct)
        upper_bound.append(max_pct)

    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    max_change = np.array(max_change)
    cost_levels = np.array(cost_levels)

    mu = np.zeros(dims)

    mu_fitness = predicted_co2 - co2_target

    sigma = ES_SIGMA_INIT
    eval_count = 1

    def indiv_to_dict(indiv):
        return {f: indiv[i] for i, f in enumerate(feature_selection)}
    
    def evaluate(indiv):
        f_dict = indiv_to_dict(indiv)
        pred = predict_fn(f_dict)

        if pred > co2_target:
            error = 10 * (pred - co2_target)
        else:
            error = co2_target - pred
        
        penalty = np.sum(cost_levels * (np.abs(indiv) / max_change))

        fitness = error + 0.1 * penalty
        return fitness, pred
    
    best_predicted = predicted_co2
    
    while eval_count < ES_EVAL_LIMIT:
        epsilon = np.random.randn(ES_POP_SIZE, dims)
        offsping = mu + sigma * epsilon

        offsping = np.clip(offsping, lower_bound, upper_bound)

        fitness = []
        preds = []
        for i in range(ES_POP_SIZE):
            fit, pred = evaluate(offsping[i])
            fitness.append(fit)
            preds.append(pred)

        fitness = np.array(fitness)
        preds = np.array(preds)
        eval_count += ES_POP_SIZE

        best_idx = np.argmin(fitness)
        best_child = offsping[best_idx]
        best_fit = fitness[best_idx]
        best_pred = preds[best_idx]

        if best_fit <= mu_fitness:
            mu = best_child.copy()
            mu_fitness = best_fit
            best_predicted = best_pred
            sigma *= ES_C_INC
        else:
            sigma *= ES_C_DEC

        if best_predicted <= co2_target:
            break
    
    best_change = indiv_to_dict(mu)

    return best_change, mu_fitness, best_predicted

