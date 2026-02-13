import numpy as np

def optimal_threshold(y_true, y_proba, cost_matrix):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_t, best_cost = 0.5, None

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()

        total = (
            fp * cost_matrix["FP"] +
            fn * cost_matrix["FN"] +
            tp * cost_matrix["TP"] +
            tn * cost_matrix["TN"]
        )
        if best_cost is None or total < best_cost:
            best_cost = float(total)
            best_t = float(t)

    return best_t, best_cost