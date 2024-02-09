import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1, gamma=2, reduction=tf.keras.losses.Reduction.AUTO, name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # Compute binary cross-entropy loss
        bce_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)

        # Calculate focal loss
        pt = tf.exp(-bce_loss)
        focal_loss = self.alpha * tf.pow(1 - pt, self.gamma) * bce_loss

        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(focal_loss)
        elif self.reduction == tf.keras.losses.Reduction.NONE:
            return focal_loss
        else:
            return tf.reduce_mean(focal_loss)