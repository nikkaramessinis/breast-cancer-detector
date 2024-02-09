
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    def __init__(self, target_accuracy=0.90):
        super(CustomCallback, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy >= self.target_accuracy:
            print(f"\nReached target accuracy ({self.target_accuracy}), stopping training!")
            self.model.stop_training = True