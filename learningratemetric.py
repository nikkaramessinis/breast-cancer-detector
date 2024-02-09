import tensorflow as tf

# Learning Rate
class LearningRateMetric(tf.keras.metrics.Metric):
    def __init__(self, name='learning_rate', **kwargs):
        super(LearningRateMetric, self).__init__(name=name, **kwargs)
        self.learning_rates = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(current_lr)

    def result(self):
        return self.learning_rates