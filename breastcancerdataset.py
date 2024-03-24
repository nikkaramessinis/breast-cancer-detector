from tensorflow.keras.utils import Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import dicomsdl  # or from dicomsdl import some_function
from datetime import datetime
import cv2
import numpy as np
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
            return np.fliplr(img)

    # Determine the current breast side
    def _determine_breast_side(self, img):
        col_sums_split = np.array_split(np.sum(img, axis=0), 2)
        left_col_sum = np.sum(col_sums_split[0])
        right_col_sum = np.sum(col_sums_split[1])
        if left_col_sum > right_col_sum:
            return 'L'
        else:
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

    def get_pixels_hu(self, slice, image):
        try:
            # Convert to int16 (from sometimes int16),
            # should be possible as values should always be low enough (<32k)
            image = image.astype(np.int16)

            # Set outside-of-scan pixels to 0
            # The intercept is usually -1024, so air is approximately 0
            image[image == -2000] = 0

            # Convert to Hounsfield units (HU)
            intercept = slice.RescaleIntercept
            slope = slice.RescaleSlope

            if slope is not None and slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(np.int16)

                image += np.int16(intercept)

        except Exception as e:
            print(f"Intercept: {intercept}, Slope: {slope}")
            print(f"get_pixels_hu {image}: {e}")
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

    def resize_with_letterbox(self, image, target_size):
        resized_image = image
        # Get the original image size
        try:
            original_height, original_width = image.shape[:2]
            largest_roi_dim = max(image.shape[:2])

            # Step 2: Create an empty square box
            empty_box = np.zeros((largest_roi_dim, largest_roi_dim, 3), dtype=np.uint8)

            # Step 3: Place extracted ROI on the box
            # Calculate position to place ROI in the center of the empty box
            x_offset = (largest_roi_dim - image.shape[1]) // 2
            y_offset = (largest_roi_dim - image.shape[0]) // 2
            empty_box[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

            resized_image = cv2.resize(empty_box, (target_size[0], target_size[1]))
        except Exception as e:
            print(f"Here Error loading DICOM file {image}: {e}")
        return resized_image

    def load_and_preprocess_image(self, file_path):

        def normalize_image(image):
            min_val = np.min(image)
            max_val = np.max(image)
            image = (image - min_val) / (max_val - min_val)
            return image

        try:
            slice = dicomsdl.open(file_path)

            image = slice.pixelData()
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = self.enhance_contrast(image)
            label = self.df[self.df['file_path'] == file_path]['cancer'].values[0]
            return image, label
        except Exception as e:
            print(f'Error loading image from {file_path}: {str(e)}')
            return None, None