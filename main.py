import pandas as pd
import seaborn as sns
import warnings
import tensorflow as tf
import os
import io
import sys
import logging
import numpy as np
from plots import draw_plots


warnings.filterwarnings("ignore", category=DeprecationWarning, module="keras")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers  # Add this line
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2


from cycliclr import CyclicLR
from customcallback import CustomCallback
from metricscollection import collect_metrics
from breastcancerdataset import BreastCancerDataset
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from metricscollection import summary
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from focalloss import FocalLoss




def configure_logging(log_filename):

    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_available_filename(prefix='Debug'):
    counter = 1
    while True:
        log_filename = f"{prefix}{counter}.log"
        if not os.path.exists(log_filename):
            return log_filename
        counter += 1


def main():
    log_filename = find_available_filename()
    configure_logging(log_filename)
    main_folder = 'D:\\rsna-breast-cancer-detection\\train_images\\'  # Replace with the path to your image folder
    img_dir = 'D:\\rsna-breast-cancer-detection\\imgs\\'  # Replace with the path to your image folder

    df = pd.read_csv('D:\\rsna-breast-cancer-detection\\train.csv')
    df["path"] = img_dir + df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png"

    # Note that CC (craniocaudal) and MLO (mediolateral oblique) are by far the most common imaging views, and so the few unusual views were dropped as well.
    df = df[(df.view == "MLO") | (df.view == "CC")]

    # The not-malignant cancer cases were limited into biopsy cases.
    DF_train = df[df['biopsy'] == 1].reset_index(drop = True)
    DF_train.head()

    # The number of positive (malignant) and negative (not-malignat) cases should be the same
    # to create a balanced dataset.
    #undersampled_majority = DF_train[DF_train['cancer'] == 1].sample(500, random_state=42)
    #undersampled_minority = DF_train[DF_train['cancer'] == 0].sample(500, random_state=42)
    #DF_train = pd.concat([undersampled_majority, undersampled_minority])

    # we had this before
    #DF_train = DF_train.groupby(['cancer']).apply(lambda x: x.sample(500, replace = True)
    #                                                      ).reset_index(drop = True)

    draw_plots(DF_train)

    print('New Data Size:', DF_train.shape[0])


    # the number of not-malignant cancer cases from biopsy
    logging.debug(f"the number of not-malignant cancer cases from biopsy {len(DF_train[(DF_train['biopsy'] == 1) & (DF_train['cancer'] == 0)])}")

    # the number of malignant cancer cases from biopsy
    logging.debug(f"the number of malignant cancer cases from biopsy {len(DF_train[(DF_train['biopsy'] == 1) & (DF_train['cancer'] == 1)])}")

    images_ = DF_train[['patient_id', 'image_id', 'cancer']]
    images_['file_path'] = images_.apply(lambda x: os.path.join(main_folder, str(x.patient_id), str(x.image_id) + '.dcm'), axis=1)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )

    # Split the data
    train_images_df, val_images_df = train_test_split(images_, test_size=0.2)

    logging.debug(f"train_images_df  {train_images_df.head()}")
    logging.debug(f"train_images_df columns  {train_images_df.columns}")
    logging.debug(f"train_images_df  shape {train_images_df.shape}")

    logging.debug(f"val_images_df  {val_images_df.head()}")
    logging.debug(f"val_images_df columns  {val_images_df.columns}")
    logging.debug(f"val_images_df  shape {val_images_df.shape}")
    # Define batch size and target size

    batch_size = 32
    image_size = (512, 512)
    STEPS_PER_EPOCHS = train_images_df.shape[0] / batch_size
    logging.debug(f"batch_size {batch_size}")
    logging.debug("Number of Training Samples : {}".format(train_images_df.shape[0]))
    logging.debug("Number of Validation Samples : {}".format(val_images_df.shape[0]))
    logging.debug(f"Number of Samples in one Batch: {batch_size}")
    logging.debug(f'STEPS PER EPOCHS: {STEPS_PER_EPOCHS}')

    # Create data generators using tf.data
    train_dataset = BreastCancerDataset(train_images_df, batch_size, image_size, datagen=datagen)
    val_dataset = BreastCancerDataset(val_images_df, batch_size, image_size)

    train_data_loader = tf.data.Dataset.from_generator(
        lambda: train_dataset,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
        )).cache().prefetch(tf.data.AUTOTUNE)


    val_data_loader = tf.data.Dataset.from_generator(
        lambda: val_dataset,
        output_signature=(
            tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
        )).cache().prefetch(tf.data.AUTOTUNE)

    all_X = []
    all_y = []
    for data in val_data_loader:
        X, y = data
        all_X.append(X)
        all_y.append(y)

    # Concatenate all batches into a single batch
    merged_X = tf.concat(all_X, axis=0)
    merged_y = tf.concat(all_y, axis=0)

    # Convert the tensors to NumPy arrays
    X_numpy = merged_X.numpy()
    y_numpy = merged_y.numpy()

    logging.debug(X_numpy.shape)

    #custom_callback = CustomCallback(target_accuracy=0.85)

    early_stopper = EarlyStopping(monitor='val_accuracy',
                                  min_delta=0.01,
                                  patience=3,
                                  restore_best_weights=True
                                  )

    # Set up ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath="model_checkpoint.h5",  # Specify the file path where checkpoints will be saved
        monitor="accuracy",  # Choose the metric to monitor (e.g., validation loss)
        save_best_only=True,  # Save only the best models based on the monitored metric
        save_weights_only=True,  # Save only the model weights, not the entire model
        mode="auto",  # Mode can be "min", "max", or "auto" (determines the best checkpoint)
        verbose=1  # Set to 1 to see checkpoint saving information in the console
    )

    clr = CyclicLR(base_lr=1e-4, max_lr=1e-2, step_size=8, mode='triangular')

    num_classes = 1
    input_shape = (image_size[0], image_size[1], 3)
    my_dense_activation = 'sigmoid'# try others as well softmax
    logging.debug(f"input_shape {input_shape} my_dense_activation {my_dense_activation}")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the layers of the pre-trained model
    base_model.trainable = False

    # Create a new model on top of the pre-trained model
    model = Sequential([
        base_model,
        Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        MaxPooling2D((3, 3)),
        Flatten(),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation=my_dense_activation)
    ])

    alpha = 1  # balancing parameter, adjust as needed
    gamma = 2  # focusing parameter, adjust as needed
    focal_loss_criterion = FocalLoss(alpha=alpha, gamma=gamma)

    logging.debug("focal_loss as a model.compile loss")
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),loss=focal_loss_criterion,
                  #loss='binary_crossentropy',
                  metrics=['accuracy'])

    tensorboard_callback = TensorBoard(log_dir="logs")

    #hist = model.fit(train_data_loader, validation_data=val_data_loader, epochs=100, callbacks=[early_stopper, clr, custom_callback, checkpoint_callback, tensorboard_callback])
    hist = model.fit(train_data_loader, validation_data=val_data_loader, epochs=50, callbacks=[early_stopper, checkpoint_callback, tensorboard_callback])
    model.save(f'{log_filename}.keras')

    fig, axes = plt.subplots(ncols=2, figsize=(22, 7))
    sns.lineplot(hist.history['loss'], ax=axes[0], label='loss')
    sns.lineplot(hist.history['val_loss'], ax=axes[0], label='Validation loss', linestyle='--')
    axes[0].set_xlabel('Number of epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Chart')

    sns.lineplot(hist.history['accuracy'], ax=axes[1], label='Training Accuracy')
    sns.lineplot(hist.history['val_accuracy'], ax=axes[1], label='Validation Accuracy', linestyle='--')
    axes[1].set_xlabel('Number of epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Chart')

    plt.suptitle('Model Performance')
    plt.tight_layout()
    plt.savefig(f'{log_filename}.png')

    summary(model)
    predictions = model.predict(X_numpy)
    y_pred = (predictions > 0.5).astype(int)
    y_true = y_numpy
    collect_metrics(y_pred, y_true)





def prediction_phase():
    # Predict on the test data
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Assuming a classification task with softmax activation

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    logging.debug(f"Accuracy: {accuracy}")



    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    logging.debug(f"{classification_report(y_test, y_pred)}")

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    #multiprocessing.freeze_support()
    main()