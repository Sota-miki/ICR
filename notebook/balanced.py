class BalancedLogLoss:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        y_true = target.astype(int)
        y_pred = approxes[0].astype(float)
        
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        individual_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        class_weights = np.where(y_true == 1, np.sum(y_true == 0) / np.sum(y_true == 1), np.sum(y_true == 1) / np.sum(y_true == 0))
        weighted_loss = individual_loss * class_weights
        
        balanced_logloss = np.mean(weighted_loss)
        
        return balanced_logloss, 0.0