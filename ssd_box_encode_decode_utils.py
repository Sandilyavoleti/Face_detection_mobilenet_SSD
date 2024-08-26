import numpy as np

def iou(boxes1, boxes2, coords='centroids'):
    if len(boxes1.shape) > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
    if len(boxes2.shape) > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

    if len(boxes1.shape) == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.".format(boxes1.shape[1], boxes2.shape[1]))

    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2minmax')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2minmax')
    elif coords != 'minmax':
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    # Compute intersection
    intersection_xmin = np.maximum(boxes1[:, 0], boxes2[:, 0])
    intersection_ymin = np.maximum(boxes1[:, 2], boxes2[:, 2])
    intersection_xmax = np.minimum(boxes1[:, 1], boxes2[:, 1])
    intersection_ymax = np.minimum(boxes1[:, 3], boxes2[:, 3])

    intersection_area = np.maximum(0, intersection_xmax - intersection_xmin) * np.maximum(0, intersection_ymax - intersection_ymin)

    # Compute union
    area1 = (boxes1[:, 1] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 2])
    area2 = (boxes2[:, 1] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 2])
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    iou = intersection_area / np.maximum(union_area, np.finfo(float).eps)

    return iou

def convert_coordinates(tensor, start_index, conversion='minmax2centroids'):
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float64)
    
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind]  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2]  # Set h
        
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0  # Set ymax

    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")
    
    return tensor1

def convert_coordinates2(tensor, start_index, conversion='minmax2centroids'):
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float64)
    
    if conversion == 'minmax2centroids':
        # Convert from (xmin, xmax, ymin, ymax) to (cx, cy, w, h)
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0  # cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0  # cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind]  # w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2]  # h

    elif conversion == 'centroids2minmax':
        # Convert from (cx, cy, w, h) to (xmin, xmax, ymin, ymax)
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0  # xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0  # xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0  # ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0  # ymax

    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1

def greedy_nms(y_pred_decoded, iou_threshold=0.3, coords='minmax'):
    y_pred_decoded_nms = []
    for batch_item in y_pred_decoded: 
        boxes_left = np.copy(batch_item)
        maxima = [] 
        while boxes_left.shape[0] > 0: 
            maximum_index = np.argmax(boxes_left[:, 1]) 
            maximum_box = np.copy(boxes_left[maximum_index]) 
            maxima.append(maximum_box) 
            boxes_left = np.delete(boxes_left, maximum_index, axis=0) 
            if boxes_left.shape[0] == 0:
                break 
            similarities = iou(boxes_left[:, 2:], maximum_box[2:], coords=coords) 
            boxes_left = boxes_left[similarities <= iou_threshold] 
        y_pred_decoded_nms.append(np.array(maxima))

    return y_pred_decoded_nms

def _greedy_nms(predictions, iou_threshold=0.3, coords='minmax'):
    if len(predictions) == 0:
        return np.array([])

    remaining_boxes = np.copy(predictions)
    selected_boxes = []

    while remaining_boxes.shape[0] > 0:
        max_conf_index = np.argmax(remaining_boxes[:, 0])
        best_box = remaining_boxes[max_conf_index]
        selected_boxes.append(best_box)
        remaining_boxes = np.delete(remaining_boxes, max_conf_index, axis=0)

        if remaining_boxes.shape[0] == 0:
            break

        ious = iou(remaining_boxes[:, 1:], best_box[1:], coords=coords)
        remaining_boxes = remaining_boxes[ious <= iou_threshold]

    return np.array(selected_boxes)

def _greedy_nms2(predictions, iou_threshold=0.3, coords='minmax'):
    if predictions.shape[0] == 0:
        return np.array([])

    boxes_left = np.copy(predictions)
    maxima = []

    while boxes_left.shape[0] > 0:
        maximum_index = np.argmax(boxes_left[:, 1])  # Get the index of the box with the highest confidence
        maximum_box = boxes_left[maximum_index]  # Copy that box
        maxima.append(maximum_box)  # Append it to `maxima`
        
        # If there's only one box left, we're done
        if boxes_left.shape[0] == 1:
            break
        
        # Remove the maximum box from the list
        boxes_left = np.delete(boxes_left, maximum_index, axis=0)
        
        # Compute IoU between the remaining boxes and the maximum box
        similarities = iou(boxes_left[:, 2:], maximum_box[2:], coords=coords)
        
        # Keep only boxes with IoU <= iou_threshold
        boxes_left = boxes_left[similarities <= iou_threshold]

    return np.array(maxima)

def decode_y(y_pred,
             confidence_thresh=0.7,
             iou_threshold=0.3,
             top_k=200,
             input_coords='centroids',
             normalize_coords=False,
             img_height=None,
             img_width=None):

    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, img_height and img_width must be provided.")

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    y_pred_decoded_raw = np.copy(y_pred[:, :, :-8])  # Slice out the classes and the four offsets

    if input_coords == 'centroids':
        y_pred_decoded_raw[:, :, -2:] = np.exp(y_pred_decoded_raw[:, :, -2:] * y_pred[:, :, -2:])
        y_pred_decoded_raw[:, :, -2:] *= y_pred[:, :, -6:-4]
        y_pred_decoded_raw[:, :, -4:-2] *= y_pred[:, :, -4:-2] * y_pred[:, :, -6:-4]
        y_pred_decoded_raw[:, :, -4:-2] += y_pred[:, :, -8:-6]
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:, :, -4:] *= y_pred[:, :, -4:]
        y_pred_decoded_raw[:, :, -4:-2] *= np.expand_dims(y_pred[:, :, -7] - y_pred[:, :, -8], axis=-1)
        y_pred_decoded_raw[:, :, -2:] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -6], axis=-1)
        y_pred_decoded_raw[:, :, -4:] += y_pred[:, :, -8:-4]
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax' and 'centroids'.")

    # 2: Convert normalized box coordinates back to absolute coordinates if needed
    if normalize_coords:
        y_pred_decoded_raw[:, :, -4:-2] *= img_width
        y_pred_decoded_raw[:, :, -2:] *= img_height

    # 3: Apply confidence thresholding and non-maximum suppression per class
    n_classes = y_pred_decoded_raw.shape[-1] - 4  # The number of classes
    y_pred_decoded = []  # Store the final predictions in this list

    for batch_item in y_pred_decoded_raw:
        pred = []
        for class_id in range(1, n_classes):  # Skip background class (ID 0)
            single_class = batch_item[:, [class_id, -4, -3, -2, -1]]
            threshold_met = single_class[single_class[:, 0] > confidence_thresh]

            if threshold_met.shape[0] > 0:
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='minmax')
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1))
                maxima_output[:, 0] = class_id  # Write class ID
                maxima_output[:, 1:] = maxima  # Write NMS maxima
                pred.append(maxima_output)

        if pred:
            pred = np.concatenate(pred, axis=0)
            if pred.shape[0] > top_k:  # Apply top_k filtering
                top_k_indices = np.argpartition(pred[:, 1], kth=pred.shape[0] - top_k, axis=0)[pred.shape[0] - top_k:]
                pred = pred[top_k_indices]
            y_pred_decoded.append(pred)
        else:
            y_pred_decoded.append(np.array([]))  # Append an empty array if no predictions

    return y_pred_decoded

def decode_y2(y_pred,
              confidence_thresh=0.7,
              iou_threshold=0.3,
              top_k='all',
              input_coords='centroids',
              normalize_coords=False,
              img_height=None,
              img_width=None):

    # Debugging: print the shape of y_pred
    print("y_pred shape:", y_pred.shape)

    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError(f"If relative box coordinates are supposed to be converted to absolute coordinates, img_height and img_width must be provided.")

    # 1: Convert the classes from one-hot encoding to their class ID
    if y_pred.shape[-1] != 6:
        raise ValueError(f"Expected y_pred to have 6 elements in the last dimension, but got {y_pred.shape[-1]}")

    y_pred_converted = np.zeros_like(y_pred[:, :, :6])  # Create a placeholder for the converted predictions

    # Convert one-hot encoding to class ID and store the confidence values
    y_pred_converted[:, :, 0] = np.argmax(y_pred[:, :, :2], axis=-1)  # Class ID
    y_pred_converted[:, :, 1] = np.amax(y_pred[:, :, :2], axis=-1)    # Confidence

    # Copy the predicted box coordinates directly
    y_pred_converted[:, :, 2:] = y_pred[:, :, 2:]

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if input_coords == 'centroids':
        y_pred_converted[:, :, 4:] = np.exp(y_pred_converted[:, :, 4:])  # Convert width and height
        y_pred_converted[:, :, 2:4] += y_pred[:, :, 2:4]  # Adjust by anchor box center coordinates
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=2, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        y_pred_converted[:, :, 2:] += y_pred[:, :, 2:]  # Convert to absolute coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported values are 'minmax' and 'centroids'.")

    # 3: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
    if normalize_coords:
        y_pred_converted[:, :, 2:4] *= img_width  # Convert xmin, xmax back to absolute coordinates
        y_pred_converted[:, :, 4:] *= img_height  # Convert ymin, ymax back to absolute coordinates

    # 4: Decode our huge `(batch, #boxes, 6)` tensor into a list of length `batch` where each list entry is an array containing only the positive predictions
    y_pred_decoded = []
    for batch_item in y_pred_converted:
        # Filter out background class (class ID 0) and low-confidence boxes
        boxes = batch_item[batch_item[:, 0] > 0]  # Remove background class
        boxes = boxes[boxes[:, 1] >= confidence_thresh]  # Filter out low-confidence boxes

        # Perform Non-Maximum Suppression (NMS) if IoU threshold is set
        if iou_threshold and boxes.shape[0] > 0:
            boxes = _greedy_nms2(boxes, iou_threshold=iou_threshold, coords='minmax')

        # Keep only top_k results if specified
        if top_k != 'all' and boxes.shape[0] > top_k:
            top_k_indices = np.argpartition(boxes[:, 1], kth=boxes.shape[0] - top_k, axis=0)[-top_k:]
            boxes = boxes[top_k_indices]

        y_pred_decoded.append(boxes)

    return y_pred_decoded



class SSDBoxEncoder:

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 limit_boxes=True,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 pos_iou_threshold=0.6,
                 neg_iou_threshold=0.4,
                 coords='centroids',
                 normalize_coords=False):
        
        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.predictor_sizes = np.array(predictor_sizes)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.variances = np.array(variances)
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Ensure predictor_sizes has the correct shape
        if len(self.predictor_sizes.shape) == 1:
            self.predictor_sizes = np.expand_dims(self.predictor_sizes, axis=0)

        # Validate scales
        if (self.min_scale is None or self.max_scale is None) and self.scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if self.scales:
            if len(self.scales) != len(self.predictor_sizes) + 1:
                raise ValueError(f"Scales must be of length {len(self.predictor_sizes) + 1}, but received length {len(self.scales)}.")

        # Validate aspect_ratios_per_layer
        if self.aspect_ratios_per_layer:
            if len(self.aspect_ratios_per_layer) != len(self.predictor_sizes):
                raise ValueError(f"Aspect ratios per layer must be of length {len(self.predictor_sizes)}, but received length {len(self.aspect_ratios_per_layer)}.")

        # Validate variances
        if len(self.variances) != 4:
            raise ValueError(f"4 variance values must be provided, but received {len(self.variances)}.")
        if np.any(self.variances <= 0):
            raise ValueError(f"All variances must be >0, but received variances {self.variances}.")

        # Validate IoU thresholds
        if self.neg_iou_threshold > self.pos_iou_threshold:
            raise ValueError("Negative IoU threshold must be <= positive IoU threshold.")

        # Validate coordinates format
        if self.coords not in ['minmax', 'centroids']:
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

        # Compute the number of boxes per cell
        if self.aspect_ratios_per_layer:
            self.n_boxes = []
            for aspect_ratios in self.aspect_ratios_per_layer:
                if 1 in aspect_ratios and self.two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if 1 in self.aspect_ratios_global and self.two_boxes_for_ar1:
                self.n_boxes = len(self.aspect_ratios_global) + 1
            else:
                self.n_boxes = len(self.aspect_ratios_global)

    def generate_anchor_boxes(self,
                          batch_size,
                          feature_map_size,
                          aspect_ratios,
                          this_scale,
                          next_scale,
                          diagnostics=False):
        # Sort aspect ratios
        aspect_ratios = np.sort(aspect_ratios)
        size = min(self.img_height, self.img_width)

        # Compute the box widths and heights for all aspect ratios
        wh_list = []
        n_boxes = len(aspect_ratios)
        for ar in aspect_ratios:
            if ar == 1 and self.two_boxes_for_ar1:
                # Regular anchor box for aspect ratio 1
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
                # Compute one slightly larger version using the geometric mean of this scale and the next
                w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
                h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
                wh_list.append((w, h))
                n_boxes += 1
            else:
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
        wh_list = np.array(wh_list)

        # Compute the grid of box center points, identical for all aspect ratios
        cell_height = self.img_height / feature_map_size[0]
        cell_width = self.img_width / feature_map_size[1]
        cx = np.linspace(cell_width / 2, self.img_width - cell_width / 2, feature_map_size[1])
        cy = np.linspace(cell_height / 2, self.img_height - cell_height / 2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2minmax')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.limit_boxes:
            boxes_tensor[:, :, :, [0, 2]] = np.clip(boxes_tensor[:, :, :, [0, 2]], 0, self.img_width)
            boxes_tensor[:, :, :, [1, 3]] = np.clip(boxes_tensor[:, :, :, [1, 3]], 0, self.img_height)

        # If `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # If `coords` is 'centroids', convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
        if self.coords == 'centroids':
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='minmax2centroids')

        # Prepend one dimension to `boxes_tensor` to account for the batch size and tile it
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = np.tile(boxes_tensor, (batch_size, 1, 1, 1, 1))

        # Reshape the 5D tensor into a 3D tensor of shape `(batch, feature_map_height * feature_map_width * n_boxes, 4)`
        boxes_tensor = np.reshape(boxes_tensor, (batch_size, -1, 4))

        if diagnostics:
            return boxes_tensor, wh_list, (int(cell_height), int(cell_width))
        else:
            return boxes_tensor

    def generate_encode_template(self, batch_size, diagnostics=False):
    # Get the anchor box scaling factors for each conv layer
        if not self.scales:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)

        boxes_tensor = []
        if diagnostics:
            wh_list = []
            cell_sizes = []
            for i in range(len(self.predictor_sizes)):
                if self.aspect_ratios_per_layer:
                    aspect_ratios = self.aspect_ratios_per_layer[i]
                else:
                    aspect_ratios = self.aspect_ratios_global

                boxes, wh, cells = self.generate_anchor_boxes(
                    batch_size=batch_size,
                    feature_map_size=self.predictor_sizes[i],
                    aspect_ratios=aspect_ratios,
                    this_scale=self.scales[i],
                    next_scale=self.scales[i + 1],
                    diagnostics=True
                )
                boxes_tensor.append(boxes)
                wh_list.append(wh)
                cell_sizes.append(cells)
        else:
            for i in range(len(self.predictor_sizes)):
                if self.aspect_ratios_per_layer:
                    aspect_ratios = self.aspect_ratios_per_layer[i]
                else:
                    aspect_ratios = self.aspect_ratios_global

                boxes = self.generate_anchor_boxes(
                    batch_size=batch_size,
                    feature_map_size=self.predictor_sizes[i],
                    aspect_ratios=aspect_ratios,
                    this_scale=self.scales[i],
                    next_scale=self.scales[i + 1],
                    diagnostics=False
                )
                boxes_tensor.append(boxes)

        # Concatenate the anchor tensors from the individual layers into one
        boxes_tensor = np.concatenate(boxes_tensor, axis=1)

        # Create a template tensor to hold the one-hot class encodings
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # Create a tensor to contain the variances
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances

        # Concatenate the classes, boxes, and variances tensors to form the final template
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encode_template, wh_list, cell_sizes
        else:
            return y_encode_template


    def encode_y(self, ground_truth_labels):
        # 1: Generate the template for y_encoded
        y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
        y_encoded = np.copy(y_encode_template)  # Copy the template to start encoding ground truth data

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        class_vector = np.eye(self.n_classes)  # Identity matrix for one-hot encoding of class labels

        for i in range(y_encode_template.shape[0]):  # For each batch item...
            available_boxes = np.ones((y_encode_template.shape[1]))  # Track available anchor boxes
            negative_boxes = np.ones((y_encode_template.shape[1]))  # Track potential negative boxes

            for true_box in ground_truth_labels[i]:  # For each ground truth box...
                true_box = true_box.astype(np.float64)
                if (true_box[2] - true_box[1] == 0) or (true_box[4] - true_box[3] == 0):
                    continue  # Skip boxes with zero width or height

                # Normalize coordinates if needed
                if self.normalize_coords:
                    true_box[1:3] /= self.img_width  # Normalize xmin and xmax
                    true_box[3:5] /= self.img_height  # Normalize ymin and ymax

                # Convert coordinates if needed
                if self.coords == 'centroids':
                    true_box = convert_coordinates(true_box, start_index=1, conversion='minmax2centroids')

                # Compute IoU similarity between ground truth and anchor boxes
                similarities = iou(y_encode_template[i, :, -12:-8], true_box[1:], coords=self.coords)

                # Mark overlapping boxes that can't be negatives
                negative_boxes[similarities >= self.neg_iou_threshold] = 0

                # Filter out anchor boxes already assigned to another ground truth box
                similarities *= available_boxes

                # Assign boxes that meet the IoU threshold
                assign_indices = np.nonzero(similarities >= self.pos_iou_threshold)[0]

                if len(assign_indices) > 0:  # If we have any matches...
                    y_encoded[i, assign_indices, :-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0)
                    available_boxes[assign_indices] = 0  # Mark these boxes as used
                else:  # No matches, assign to the best available box
                    best_match_index = np.argmax(similarities)
                    y_encoded[i, best_match_index, :-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0)
                    available_boxes[best_match_index] = 0
                    negative_boxes[best_match_index] = 0  # This box is no longer a negative

            # Assign all remaining boxes as background class (class_id 0)
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i, background_class_indices, 0] = 1  # Set class_id 0 (background)

        # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
        if self.coords == 'centroids':
            y_encoded[:, :, -12:-11] -= y_encode_template[:, :, -12:-11]  # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:, :, -12:-11] /= y_encode_template[:, :, -10:-9] * y_encode_template[:, :, -4:-3]  # Normalize by anchor dimensions and variances
            y_encoded[:, :, -10:-9] /= y_encode_template[:, :, -10:-9]  # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:, :, -10:-9] = np.log(y_encoded[:, :, -10:-9]) / y_encode_template[:, :, -2:-1]  # Log scaling for width and height
        else:
            y_encoded[:, :, -12:-8] -= y_encode_template[:, :, -12:-8]  # gt - anchor
            y_encoded[:, :, [-12, -11]] /= np.expand_dims(y_encode_template[:, :, -11] - y_encode_template[:, :, -12], axis=-1)
            y_encoded[:, :, [-10, -9]] /= np.expand_dims(y_encode_template[:, :, -9] - y_encode_template[:, :, -10], axis=-1)
            y_encoded[:, :, -12:-8] /= y_encode_template[:, :, -4:]  # Normalize by anchor dimensions and variances

        return y_encoded
