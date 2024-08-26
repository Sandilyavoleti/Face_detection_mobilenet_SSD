import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from ssd_box_encode_decode_utils import convert_coordinates  # Import necessary utility

class TileLayer(Layer):
    def __init__(self, **kwargs):
        super(TileLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.tile(inputs, [tf.shape(inputs)[0], 1, 1, 1, 1])

class AnchorBoxes(Layer):
    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.75, 1.0, 1.5],
                 two_boxes_for_ar1=True,
                 limit_boxes=True,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):
        super(AnchorBoxes, self).__init__(**kwargs)
        
        if this_scale <= 0 or next_scale <= 0 or this_scale > 1:
            raise ValueError(f"`this_scale` must be in (0, 1] and `next_scale` must be > 0, but got `this_scale`={this_scale} and `next_scale`={next_scale}")

        if len(variances) != 4 or np.any(np.array(variances) <= 0):
            raise ValueError(f"4 variance values must be positive, but got {variances}")

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords

        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)

    def build(self, input_shape):
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x):
        aspect_ratios = np.array(self.aspect_ratios)
        size = min(self.img_height, self.img_width)
        wh_list = []

        for ar in aspect_ratios:
            if ar == 1 and self.two_boxes_for_ar1:
                w = self.this_scale * size * np.sqrt(ar)
                h = self.this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))

                w = np.sqrt(self.this_scale * self.next_scale) * size * np.sqrt(ar)
                h = np.sqrt(self.this_scale * self.next_scale) * size / np.sqrt(ar)
                wh_list.append((w, h))
            else:
                w = self.this_scale * size * np.sqrt(ar)
                h = self.this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))

        wh_list = np.array(wh_list)

        feature_map_height, feature_map_width = tf.shape(x)[1], tf.shape(x)[2]

        cell_height = self.img_height / tf.cast(feature_map_height, tf.float32)
        cell_width = self.img_width / tf.cast(feature_map_width, tf.float32)

        cx = tf.linspace(cell_width / 2, self.img_width - cell_width / 2, feature_map_width)
        cy = tf.linspace(cell_height / 2, self.img_height - cell_height / 2, feature_map_height)
        cx_grid, cy_grid = tf.meshgrid(cx, cy)

        cx_grid = tf.expand_dims(cx_grid, -1)
        cy_grid = tf.expand_dims(cy_grid, -1)

        boxes_tensor = tf.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor = tf.concat([
            tf.tile(cx_grid, [1, 1, self.n_boxes]),  # Set cx
            tf.tile(cy_grid, [1, 1, self.n_boxes]),  # Set cy
            tf.tile(tf.expand_dims(wh_list[:, 0], axis=0), [feature_map_height, feature_map_width, 1]),  # Set w
            tf.tile(tf.expand_dims(wh_list[:, 1], axis=0), [feature_map_height, feature_map_width, 1])  # Set h
        ], axis=-1)

        boxes_tensor = convert_coordinates(boxes_tensor.numpy(), start_index=0, conversion='centroids2minmax')
        boxes_tensor = tf.constant(boxes_tensor, dtype=tf.float32)

        if self.limit_boxes:
            boxes_tensor = tf.concat([
                tf.clip_by_value(boxes_tensor[..., 0:1], 0, self.img_width),
                tf.clip_by_value(boxes_tensor[..., 1:2], 0, self.img_height),
                tf.clip_by_value(boxes_tensor[..., 2:3], 0, self.img_width),
                tf.clip_by_value(boxes_tensor[..., 3:4], 0, self.img_height)
            ], axis=-1)

        if self.normalize_coords:
            boxes_tensor = tf.concat([
                boxes_tensor[..., 0:1] / self.img_width,
                boxes_tensor[..., 1:2] / self.img_height,
                boxes_tensor[..., 2:3] / self.img_width,
                boxes_tensor[..., 3:4] / self.img_height
            ], axis=-1)

        if self.coords == 'centroids':
            boxes_tensor = convert_coordinates(boxes_tensor.numpy(), start_index=0, conversion='minmax2centroids')
            boxes_tensor = tf.constant(boxes_tensor, dtype=tf.float32)

        variances_tensor = tf.ones_like(boxes_tensor) * self.variances
        boxes_tensor = tf.concat([boxes_tensor, variances_tensor], axis=-1)

        boxes_tensor = tf.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = TileLayer()(boxes_tensor)
        return boxes_tensor

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.n_boxes, 8)
