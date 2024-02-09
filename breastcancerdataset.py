from tensorflow.keras.utils import Sequence
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import dicomsdl  # or from dicomsdl import some_function
from datetime import datetime
import cv2
import numpy as np
import dicom

from tensorflow.keras.applications.vgg16 import preprocess_input

class BreastCancerDataset(Sequence):
    def __init__(self, df, batch_size, image_size, shuffle=True, datagen=None, breast_side: str='R',):
        self.df = df
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.datagen = datagen
        self.indices = np.arange(len(self.df))
        self.breast_side = breast_side

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def flip_colouring(self, image):
        # Count the number of white and black pixels
        num_white_pixels = np.sum(image == 255)
        num_black_pixels = np.sum(image == 0)

        # Decide whether to flip the colors based on the condition
        if num_white_pixels > num_black_pixels:
            inverted_image = cv2.bitwise_not(image)
            return inverted_image
        else:
            return image


    def flip_breast_side(self, img):
        img_breast_side = self._determine_breast_side(img)
        if img_breast_side == self.breast_side:
            return img
        else:
            #print("Flipping")
            return np.fliplr(img)

    # Determine the current breast side
    def _determine_breast_side(self, img):
        col_sums_split = np.array_split(np.sum(img, axis=0), 2)
        left_col_sum = np.sum(col_sums_split[0])
        right_col_sum = np.sum(col_sums_split[1])
        if left_col_sum > right_col_sum:
            #print("L")
            return 'L'
        else:
            #print("R")
            return 'R'


    def enhance_contrast(self, image):
        # Convert the input image to 8-bit unsigned char (if it's not already)
        if image.dtype != np.uint8:
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = np.uint8(image)

        image = self.flip_breast_side(image)
        image = self.flip_colouring(image)
        # Convert RGB image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


        # Apply CLAHE to each channel of the LAB image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_channels = [clahe.apply(channel) for channel in cv2.split(lab_image)]

        # Merge the enhanced channels back to LAB image
        enhanced_lab_image = cv2.merge(enhanced_channels)

        # Convert back to RGB
        enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2RGB)

        enhanced_rgb_image = enhanced_rgb_image.reshape((self.image_size[0], self.image_size[1], 3))

        # Preprocess the image using VGG16 preprocess_input function
        preprocessed_image = preprocess_input(enhanced_rgb_image)

        return preprocessed_image

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indices = self.indices[start:end]

        # Use ProcessPoolExecutor for parallel image loading and preprocessing
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(
                executor.map(self.load_and_preprocess_image, [self.df.iloc[idx]['file_path'] for idx in batch_indices]))

        #results = list(map(self.load_and_preprocess_image, [self.df.iloc[idx]['file_path'] for idx in batch_indices]))

        X = np.array([result[0] for result in results])
        y = np.array([result[1] for result in results])
        current_time = datetime.now()
        # Format the output
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        # Print the formatted time
        print("Formatted Time:", formatted_time)

        return X, y.reshape(-1, 1)

    def load_scan(path):
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices

    def get_pixels_hu(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    def resample(self, image, scan, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing
        spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing

    def load_and_preprocess_image(self, file_path):

        def normalize_image(image):
            min_val = np.min(image)
            max_val = np.max(image)
            image = (image - min_val) / (max_val - min_val)

            return image

        try:
            image = dicomsdl.open(file_path).pixelData()
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            #dicom_data.pixel_array.astype(np.uint8)
            #image = np.expand_dims(image, axis=-1)



            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            #self.show_image(image)


            image = self.enhance_contrast(image)
            #cv2.imshow('image', image)
            #c = cv2.waitKey()
            #image = normalize_image(image) #for one channel
            #image = np.repeat(image, 3, axis=-1)

            #if self.datagen:
            #    image = self.datagen.random_transform(image)
            label = self.df[self.df['file_path'] == file_path]['cancer'].values[0]
            return image, label
        except Exception as e:
            print(f'Error loading image from {file_path}: {str(e)}')
            return None, None
