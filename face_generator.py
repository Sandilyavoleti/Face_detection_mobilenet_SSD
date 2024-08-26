import sys
import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from copy import deepcopy
from PIL import Image
import csv
import os
from bs4 import BeautifulSoup
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from termcolor import colored

bb_expanded = False


def save_bb(path, filename, results, prediction=True):
    _SIZ = 300

    # Load the image without resizing
    img = image.load_img(filename)
    img_height, img_width = img.size[1], img.size[0]
    img_array = image.img_to_array(img) / 255.0  # Normalize the image

    # Adjust filename if not prediction
    if not prediction:
        filename = filename.rsplit(".", 1)[0] + "_gt.jpg"

    # Set up the plot
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)

    # Get colors for bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    print(colored(f"Total number of bounding boxes: {len(results)}", "yellow"))

    for i, result in enumerate(results):
        # Parse the outputs
        if prediction:
            det_label, det_conf, det_xmin, det_xmax, det_ymin, det_ymax = result
        else:
            det_label, det_xmin, det_xmax, det_ymin, det_ymax = result[:5]
            det_xmin *= img_width / _SIZ
            det_xmax *= img_width / _SIZ
            det_ymin *= img_height / _SIZ
            det_ymax *= img_height / _SIZ
        xmin, ymin, xmax, ymax = map(int, [det_xmin, det_ymin, det_xmax, det_ymax])
        if xmin<=0 or ymin<=0 or xmax>= img_height-1 or ymax>=img_width-1:
            continue

        # Set up label and text
        label_name = "face"  # Replace with actual label name mapping if available
        display_txt = f'{det_conf:.2f}, {label_name}' if prediction else label_name

        # Draw bounding box and label
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        color = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        ax.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5}, fontsize=8, color='white')

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Save the figure
    output_path = os.path.join(path, os.path.basename(filename))
    plt.savefig(output_path)
    print('Saved:', output_path)

    plt.close(fig)

def _translate(image, horizontal=(0, 40), vertical=(0, 10)):
    # Ensure image has the correct dimensions
    rows, cols, ch = image.shape

    # Randomly choose translation values within the given ranges
    x = np.random.randint(horizontal[0], horizontal[1] + 1)
    y = np.random.randint(vertical[0], vertical[1] + 1)
    
    # Randomly decide the direction of translation (left/right and up/down)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])

    # Define the translation matrix
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

    # Apply the translation using OpenCV's warpAffine function
    translated_image = cv2.warpAffine(image, M, (cols, rows))

    # Return the translated image along with the shift values
    return translated_image, x_shift, y_shift


def _flip(image, orientation='horizontal'):
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)


def _scale(image, min=0.9, max=1.1):
    rows, cols, ch = image.shape

    # Randomly select a scaling factor from the range passed.
    scale = np.random.uniform(min, max)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows)), M, scale


def _brightness(image, min=0.5, max=2.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min, max)

    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def histogram_eq(image):

    image1 = np.copy(image)

    image1[:, :, 0] = cv2.equalizeHist(image1[:, :, 0])
    image1[:, :, 1] = cv2.equalizeHist(image1[:, :, 1])
    image1[:, :, 2] = cv2.equalizeHist(image1[:, :, 2])

    return image1


class BatchGenerator:

    def __init__(self,
                 images_path,
                 include_classes='all',
                 box_output_format=None):
        # Initialize paths and options
        self.images_path = images_path
        self.include_classes = include_classes
        self.box_output_format = box_output_format if box_output_format else ['class_id', 'xmin', 'xmax', 'ymin', 'ymax']

        # Variables for CSV parsing
        self.labels_path = None
        self.input_format = None

        # Variables for XML parsing
        self.annotations_path = None
        self.image_set_path = None
        self.image_set = None
        self.classes = None

        # Outputs from the parsers
        self.filenames = []  # All unique image filenames
        self.labels = []  # Ground truth bounding boxes per image

    def parse_csv(self, labels_path=None, input_format=None, ret=False):
        # Set labels_path and input_format if provided
        if labels_path is not None:
            self.labels_path = labels_path
        if input_format is not None:
            self.input_format = input_format

        # Ensure that labels_path and input_format are set
        if self.labels_path is None or self.input_format is None:
            raise ValueError("`labels_path` and `input_format` must be set before parsing.")

        # Clear any previously parsed data
        self.filenames = []
        self.labels = []

        data = []

        # Read and process the CSV file
        with open(self.labels_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue  # Skip the header row
                if self.include_classes == 'all' or int(row[self.input_format.index('class_id')].strip()) in self.include_classes:
                    # Collect the image name and bounding box information
                    entry = [row[self.input_format.index('image_name')].strip()]
                    entry.extend(int(row[self.input_format.index(item)].strip()) for item in self.box_output_format)
                    data.append(entry)

        # Sort data by image filename
        data.sort()

        # Compile the actual samples and labels lists
        current_file = None
        current_labels = []

        for i in range(len(data)):
            image_name = data[i][0]
            if image_name == current_file:
                current_labels.append(data[i][1:])
            else:
                if current_file is not None:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
                current_file = image_name
                current_labels = [data[i][1:]]

            if i == len(data) - 1:  # Check if this is the last item in the list
                self.labels.append(np.stack(current_labels, axis=0))
                self.filenames.append(current_file)

        if ret:
            return self.filenames, self.labels

    def parse_xml(self,
                  annotations_path=None,
                  image_set_path=None,
                  image_set=None,
                  classes=None,
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False,
                  debug=False):
        # Set paths and classes if provided
        if annotations_path is not None:
            self.annotations_path = annotations_path
        if image_set_path is not None:
            self.image_set_path = image_set_path
        if image_set is not None:
            self.image_set = image_set
        if classes is None:
            classes = ['background',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat',
                       'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']
        self.classes = classes

        # Clear previous data
        self.filenames = []
        self.labels = []

        # Load the annotation data
        data = np.load(annotations_path, allow_pickle=True).item()

        n_train_samples = len(data)
        for train_cnt, key in enumerate(data):
            sys.stdout.flush()

            img_path = image_set_path
            img_name = data[key][1]
            image_id = key
            filename = os.path.join(img_path, img_name)

            boxes = []  # Store all boxes for this image

            # Check if the image file exists
            if not os.path.exists(filename):
                continue

            img_test = cv2.imread(filename)
            if img_test is None:
                continue

            height, width, channels = img_test.shape
            n_objects = len(data[key]) - 3

            num_valid_objects = 0
            for obj in range(n_objects):
                class_id = data[key][3 + obj][1]
                class_name = self.classes[class_id]

                # Bounding box coordinates
                xmin = data[key][3 + obj][0][0]
                xmax = data[key][3 + obj][0][1]
                ymin = data[key][3 + obj][0][2]
                ymax = data[key][3 + obj][0][3]

                # Calculate bounding box dimensions in the resized format
                bb_width = (xmax - xmin) * float(512) / width
                bb_height = (ymax - ymin) * float(512) / height

                if bb_width > 8 and bb_height > 8:  # Ensure the box is not too small
                    num_valid_objects += 1

                    # Store box details in a dictionary
                    item_dict = {
                        'folder': None,
                        'image_name': filename,
                        'image_id': image_id,
                        'class_name': class_name,
                        'class_id': class_id,
                        'pose': None,
                        'truncated': False,
                        'difficult': False,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    }

                    # Create the box based on the desired output format
                    box = [item_dict[item] for item in self.box_output_format]
                    boxes.append(box)

            if num_valid_objects > 0:
                self.filenames.append(filename)
                self.labels.append(boxes)

        if ret:
            return self.filenames, self.labels

    def generate(self,
                 batch_size=32,
                 train=True,
                 ssd_box_encoder=None,
                 equalize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 gray=False,
                 limit_boxes=True,
                 include_thresh=0.3,
                 diagnostics=False):
        
        # Shuffle the data before we begin
        self.filenames, self.labels = shuffle(self.filenames, self.labels)
        current = 0

        # Find out the indices of the box coordinates in the label data
        xmin = self.box_output_format.index('xmin')
        xmax = self.box_output_format.index('xmax')
        ymin = self.box_output_format.index('ymin')
        ymax = self.box_output_format.index('ymax')

        while True:
            batch_X, batch_y = [], []

            # Shuffle the data after each complete pass
            if current >= len(self.filenames):
                self.filenames, self.labels = shuffle(self.filenames, self.labels)
                current = 0

            for filename in self.filenames[current:current + batch_size]:
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_X.append(img)

            batch_y = deepcopy(self.labels[current:current + batch_size])
            this_filenames = self.filenames[current:current + batch_size]

            if diagnostics:
                original_images = np.copy(batch_X)  # Original, unaltered images
                original_labels = deepcopy(batch_y)  # Original, unaltered labels

            current += batch_size

            # Image transformations
            batch_items_to_remove = []

            for i in range(len(batch_X)):
                img_height, img_width, _ = batch_X[i].shape
                batch_y[i] = np.array(batch_y[i])  # Ensure labels are in array form

                if equalize:
                    batch_X[i] = histogram_eq(batch_X[i])

                if brightness:
                    if np.random.rand() >= (1 - brightness[2]):
                        batch_X[i] = _brightness(batch_X[i], min=brightness[0], max=brightness[1])

                if flip and np.random.rand() >= (1 - flip):
                    batch_X[i] = _flip(batch_X[i])
                    batch_y[i][:, [xmin, xmax]] = img_width - batch_y[i][:, [xmax, xmin]]

                if translate and np.random.rand() >= (1 - translate[2]):
                    batch_X[i], xshift, yshift = _translate(batch_X[i], translate[0], translate[1])
                    batch_y[i][:, [xmin, xmax]] += xshift
                    batch_y[i][:, [ymin, ymax]] += yshift
                    if limit_boxes:
                        batch_y[i] = self._limit_boxes(batch_y[i], img_width, img_height, include_thresh, xmin, xmax, ymin, ymax)

                if scale and np.random.rand() >= (1 - scale[2]):
                    batch_X[i], M, scale_factor = _scale(batch_X[i], scale[0], scale[1])
                    batch_y[i] = self._scale_boxes(batch_y[i], M, img_width, img_height, scale_factor, limit_boxes, include_thresh, xmin, xmax, ymin, ymax)

                if random_crop:
                    if np.random.rand() < 0.3:
                        batch_X[i], batch_y[i] = self._random_crop(batch_X[i], batch_y[i], random_crop, limit_boxes, include_thresh, xmin, xmax, ymin, ymax)
                        img_height, img_width, _ = batch_X[i].shape

                if crop:
                    batch_X[i], batch_y[i] = self._crop(batch_X[i], batch_y[i], crop, img_width, img_height, limit_boxes, include_thresh, xmin, xmax, ymin, ymax)

                if resize:
                    batch_X[i] = cv2.resize(batch_X[i], resize)
                    batch_y[i][:, [xmin, xmax]] = (batch_y[i][:, [xmin, xmax]] * (resize[0] / img_width)).astype(np.int64)
                    batch_y[i][:, [ymin, ymax]] = (batch_y[i][:, [ymin, ymax]] * (resize[1] / img_height)).astype(np.int64)

                if gray:
                    batch_X[i] = np.expand_dims(cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY), axis=-1)

            for j in sorted(batch_items_to_remove, reverse=True):
                batch_X.pop(j)
                batch_y.pop(j)

            if train:
                if ssd_box_encoder is None:
                    raise ValueError("`ssd_box_encoder` cannot be `None` in training mode.")
                y_true = ssd_box_encoder.encode_y(batch_y)
                yield np.array(batch_X), y_true
            else:
                yield np.array(batch_X), batch_y, this_filenames

    def _limit_boxes(self, boxes, img_width, img_height, include_thresh, xmin, xmax, ymin, ymax):
        """Limit box coordinates to stay within image boundaries."""
        before_limiting = deepcopy(boxes)
        boxes[:, [xmin, xmax]] = np.clip(boxes[:, [xmin, xmax]], 0, img_width - 1)
        boxes[:, [ymin, ymax]] = np.clip(boxes[:, [ymin, ymax]], 0, img_height - 1)
        return self._filter_boxes(boxes, before_limiting, include_thresh, xmin, xmax, ymin, ymax)

    def _scale_boxes(self, boxes, M, img_width, img_height, scale_factor, limit_boxes, include_thresh, xmin, xmax, ymin, ymax):
        """Apply scaling to bounding boxes."""
        toplefts = np.array([boxes[:, xmin], boxes[:, ymin], np.ones(boxes.shape[0])])
        bottomrights = np.array([boxes[:, xmax], boxes[:, ymax], np.ones(boxes.shape[0])])
        new_toplefts = np.dot(M, toplefts).T
        new_bottomrights = np.dot(M, bottomrights).T
        boxes[:, [xmin, ymin]] = new_toplefts.astype(np.int64)
        boxes[:, [xmax, ymax]] = new_bottomrights.astype(np.int64)
        if limit_boxes and scale_factor > 1:
            return self._limit_boxes(boxes, img_width, img_height, include_thresh, xmin, xmax, ymin, ymax)
        return boxes

    def _random_crop(self, img, boxes, crop_dims, limit_boxes, include_thresh, xmin, xmax, ymin, ymax):
        img_height, img_width, _ = img.shape
        crop_height, crop_width, min_1_object, max_trials = crop_dims

        y_range = img_height - crop_height
        x_range = img_width - crop_width

        min_1_object_fulfilled = False
        trial_counter = 0

        while (not min_1_object_fulfilled) and (trial_counter < max_trials):
            crop_ymin = np.random.randint(0, max(1, y_range + 1))
            crop_xmin = np.random.randint(0, max(1, x_range + 1))

            if y_range >= 0 and x_range >= 0:
                patch_X = np.copy(img[crop_ymin:crop_ymin + crop_height, crop_xmin:crop_xmin + crop_width])
                patch_y = np.copy(boxes)
                patch_y[:, [ymin, ymax]] -= crop_ymin
                patch_y[:, [xmin, xmax]] -= crop_xmin
            elif y_range >= 0 and x_range < 0:
                patch_X = np.copy(img[crop_ymin:crop_ymin + crop_height])
                canvas = np.zeros((crop_height, crop_width, patch_X.shape[2]), dtype=np.uint8)
                canvas[:, crop_xmin:crop_xmin + img_width] = patch_X
                patch_X = canvas
                patch_y = np.copy(boxes)
                patch_y[:, [ymin, ymax]] -= crop_ymin
                patch_y[:, [xmin, xmax]] += crop_xmin
            elif y_range < 0 and x_range >= 0:
                patch_X = np.copy(img[:, crop_xmin:crop_xmin + crop_width])
                canvas = np.zeros((crop_height, crop_width, patch_X.shape[2]), dtype=np.uint8)
                canvas[crop_ymin:crop_ymin + img_height, :] = patch_X
                patch_X = canvas
                patch_y = np.copy(boxes)
                patch_y[:, [ymin, ymax]] += crop_ymin
                patch_y[:, [xmin, xmax]] -= crop_xmin
            else:
                patch_X = np.copy(img)
                canvas = np.zeros((crop_height, crop_width, patch_X.shape[2]), dtype=np.uint8)
                canvas[crop_ymin:crop_ymin + img_height, crop_xmin:crop_xmin + img_width] = patch_X
                patch_X = canvas
                patch_y = np.copy(boxes)
                patch_y[:, [ymin, ymax]] += crop_ymin
                patch_y[:, [xmin, xmax]] += crop_xmin

            if limit_boxes:
                patch_y = self._limit_boxes(patch_y, crop_width, crop_height, include_thresh, xmin, xmax, ymin, ymax)

            if min_1_object == 0:
                return patch_X, patch_y
            elif len(patch_y) > 0:
                min_1_object_fulfilled = True

            trial_counter += 1

        if min_1_object_fulfilled:
            return patch_X, patch_y
        else:
            return img, boxes  # Return the original image and boxes if no valid crop was found


    def _crop(self, img, boxes, crop, img_width, img_height, limit_boxes, include_thresh, xmin, xmax, ymin, ymax):
        """Apply fixed cropping to image and adjust bounding boxes."""
        img = np.copy(img[crop[0]:img_height - crop[1], crop[2]:img_width - crop[3]])
        boxes[:, [xmin, xmax]] -= crop[2]
        boxes[:, [ymin, ymax]] -= crop[0]
        img_height -= crop[0] + crop[1]
        img_width -= crop[2] + crop[3]
        if limit_boxes:
            return img, self._limit_boxes(boxes, img_width, img_height, include_thresh, xmin, xmax, ymin, ymax)
        return img, boxes

    def _filter_boxes(self, boxes, before_limiting, include_thresh, xmin, xmax, ymin, ymax):
        """Filter out boxes that are too small after transformations."""
        before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (before_limiting[:, ymax] - before_limiting[:, ymin])
        after_area = (boxes[:, xmax] - boxes[:, xmin]) * (boxes[:, ymax] - boxes[:, ymin])
        if include_thresh == 0:
            return boxes[after_area > include_thresh * before_area]
        else:
            return boxes[after_area >= include_thresh * before_area]

    def get_filenames_labels(self):
        return self.filenames, self.labels

    def get_n_samples(self):
        return len(self.filenames)

    def process_offline(self,
                    dest_path='',
                    start=0,
                    stop='all',
                    crop=False,
                    equalize=False,
                    brightness=False,
                    flip=False,
                    translate=False,
                    scale=False,
                    resize=False,
                    gray=False,
                    limit_boxes=True,
                    include_thresh=0.3,
                    diagnostics=False):

        targets_for_csv = []
        if stop == 'all':
            stop = len(self.filenames)

        if diagnostics:
            processed_images = []
            original_images = []
            processed_labels = []

        xmin = self.box_output_format.index('xmin')
        xmax = self.box_output_format.index('xmax')
        ymin = self.box_output_format.index('ymin')
        ymax = self.box_output_format.index('ymax')

        for k, filename in enumerate(self.filenames[start:stop]):
            i = k + start
            img = Image.open(os.path.join(self.images_path, filename))
            image = np.array(img)
            targets = np.copy(self.labels[i])

            if diagnostics:
                original_images.append(image)

            img_height, img_width, ch = image.shape

            if equalize:
                image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if brightness:
                p = np.random.uniform(0, 1)
                if p >= (1 - brightness[2]):
                    image = self._brightness(image, min=brightness[0], max=brightness[1])

            if flip:
                p = np.random.uniform(0, 1)
                if p >= (1 - flip):
                    image = self._flip(image)
                    targets[:, [xmin, xmax]] = img_width - targets[:, [xmax, xmin]]

            if translate:
                p = np.random.uniform(0, 1)
                if p >= (1 - translate[2]):
                    image, xshift, yshift = self._translate(image, translate[0], translate[1])
                    targets[:, [xmin, xmax]] += xshift
                    targets[:, [ymin, ymax]] += yshift
                    if limit_boxes:
                        before_limiting = np.copy(targets)
                        x_coords = targets[:, [xmin, xmax]]
                        x_coords[x_coords >= img_width] = img_width - 1
                        x_coords[x_coords < 0] = 0
                        targets[:, [xmin, xmax]] = x_coords
                        y_coords = targets[:, [ymin, ymax]]
                        y_coords[y_coords >= img_height] = img_height - 1
                        y_coords[y_coords < 0] = 0
                        targets[:, [ymin, ymax]] = y_coords
                        before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * \
                                      (before_limiting[:, ymax] - before_limiting[:, ymin])
                        after_area = (targets[:, xmax] - targets[:, xmin]) * (targets[:, ymax] - targets[:, ymin])
                        targets = targets[after_area >= include_thresh * before_area]

            if scale:
                p = np.random.uniform(0, 1)
                if p >= (1 - scale[2]):
                    image, M, scale_factor = self._scale(image, scale[0], scale[1])
                    toplefts = np.array([targets[:, xmin], targets[:, ymin], np.ones(targets.shape[0])])
                    bottomrights = np.array([targets[:, xmax], targets[:, ymax], np.ones(targets.shape[0])])
                    new_toplefts = (np.dot(M, toplefts)).T
                    new_bottomrights = (np.dot(M, bottomrights)).T
                    targets[:, [xmin, ymin]] = new_toplefts.astype(np.int64)
                    targets[:, [xmax, ymax]] = new_bottomrights.astype(np.int64)
                    if limit_boxes and scale_factor > 1:
                        before_limiting = np.copy(targets)
                        x_coords = targets[:, [xmin, xmax]]
                        x_coords[x_coords >= img_width] = img_width - 1
                        x_coords[x_coords < 0] = 0
                        targets[:, [xmin, xmax]] = x_coords
                        y_coords = targets[:, [ymin, ymax]]
                        y_coords[y_coords >= img_height] = img_height - 1
                        y_coords[y_coords < 0] = 0
                        targets[:, [ymin, ymax]] = y_coords
                        before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * \
                                      (before_limiting[:, ymax] - before_limiting[:, ymin])
                        after_area = (targets[:, xmax] - targets[:, xmin]) * (targets[:, ymax] - targets[:, ymin])
                        targets = targets[after_area >= include_thresh * before_area]

            if crop:
                image = image[crop[0]:img_height - crop[1], crop[2]:img_width - crop[3]]
                if limit_boxes:
                    before_limiting = np.copy(targets)
                    if crop[0] > 0:
                        targets[:, [ymin, ymax]] = np.maximum(targets[:, [ymin, ymax]] - crop[0], 0)
                    if crop[1] > 0:
                        targets[:, [ymin, ymax]] = np.minimum(targets[:, [ymin, ymax]], img_height - crop[1] - 1)
                    if crop[2] > 0:
                        targets[:, [xmin, xmax]] = np.maximum(targets[:, [xmin, xmax]] - crop[2], 0)
                    if crop[3] > 0:
                        targets[:, [xmin, xmax]] = np.minimum(targets[:, [xmin, xmax]], img_width - crop[3] - 1)
                    before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * \
                                  (before_limiting[:, ymax] - before_limiting[:, ymin])
                    after_area = (targets[:, xmax] - targets[:, xmin]) * (targets[:, ymax] - targets[:, ymin])
                    targets = targets[after_area >= include_thresh * before_area]
                img_height -= (crop[0] + crop[1])
                img_width -= (crop[2] + crop[3])

            if resize:
                image = cv2.resize(image, resize)
                targets[:, [xmin, xmax]] = (targets[:, [xmin, xmax]] * (resize[0] / img_width)).astype(np.int64)
                targets[:, [ymin, ymax]] = (targets[:, [ymin, ymax]] * (resize[1] / img_height)).astype(np.int64)

            if gray:
                image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=-1)

            if diagnostics:
                processed_images.append(image)
                processed_labels.append(targets)

            img = Image.fromarray(image.astype(np.uint8))
            img.save(os.path.join(dest_path, filename), 'JPEG', quality=90)
            del image, img
            gc.collect()

            for target in targets:
                target = list(target)
                target = [filename] + target
                targets_for_csv.append(target)

        with open(os.path.join(dest_path, 'labels.csv'), 'w', newline='') as csvfile:
            labelswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            labelswriter.writerow(['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
            labelswriter.writerows(targets_for_csv)

        if diagnostics:
            print("Image processing completed.")
            return np.array(processed_images), np.array(original_images), np.array(targets_for_csv), processed_labels
        else:
            print("Image processing completed.")