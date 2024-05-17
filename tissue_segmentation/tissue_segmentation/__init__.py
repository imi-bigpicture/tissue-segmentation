import os

import numpy as np
import skimage
from scipy import ndimage
from tensorflow import convert_to_tensor, keras


def apply_network(image: np.ndarray) -> np.ndarray:
    # load model
    model = keras.models.load_model(
        os.path.join(os.path.dirname(__file__), "fcnn_c.keras")
    )

    # create mask
    result = model(convert_to_tensor(np.expand_dims(image, axis=0))).numpy()
    mask = result[0, :, :, 1]

    # rescale
    pads, downsamples = __get_padding_and_downsamples(image.shape, model)
    mask = skimage.transform.rescale(
        mask,
        (downsamples[0], downsamples[1]),
        preserve_range=True,
        order=True,
        mode="edge",
    )
    mask = np.pad(mask, ((pads[2], pads[3]), (pads[0], pads[1])), "constant")
    return mask


def create_tissue_mask(
    image: np.ndarray,
    dilation_disk_size: int = 32,
    confidence_threshold: float = 0.8,
    dilate_mask: bool = True,
    apply_hole_filling: bool = True,
    select_largest_tissue_objects: bool = True,
) -> np.ndarray:
    # preprocess
    image_preprocessed = image / 255.0

    # apply network
    mask = apply_network(image_preprocessed)

    # threshold
    confidence_threshold = confidence_threshold
    mask[mask < confidence_threshold] = 0.0
    mask[mask >= confidence_threshold] = 1.0

    # dilate mask
    if dilate_mask:
        structure_element = skimage.morphology.disk(dilation_disk_size)
        mask = skimage.morphology.binary_dilation(mask, structure_element)

    # apply hole filling
    if apply_hole_filling:
        mask = ndimage.binary_fill_holes(mask)

    # select largest tissue objects
    if select_largest_tissue_objects:
        labeled_mask, num_objects = ndimage.label(mask)
        object_areas = ndimage.sum(mask, labeled_mask, range(num_objects + 1))
        sorted_indices = np.argsort(object_areas)
        sorted_object_areas = object_areas[sorted_indices]
        differences_in_areas = np.diff(sorted_object_areas)
        index_of_largest_diff = np.argmax(differences_in_areas)
        largest_areas_labels = sorted_indices[index_of_largest_diff + 1 :]
        mask = np.isin(labeled_mask, largest_areas_labels)

    return mask


def __get_padding_and_downsamples(input_shape, model):
    layers = [
        l
        for l in model.layers
        if not any(
            isinstance(l, layer)
            for layer in [
                keras.layers.InputLayer,
                keras.layers.Dropout,
                keras.layers.Cropping2D,
                keras.layers.ReLU,
                keras.layers.LeakyReLU,
                keras.layers.PReLU,
                keras.layers.ELU,
                keras.layers.ThresholdedReLU,
                keras.layers.Softmax,
                keras.layers.BatchNormalization,
                keras.layers.Concatenate,
                keras.layers.Reshape,
                keras.layers.Activation,
            ]
        )
    ]
    lost = []
    downsamples = []
    last_shape = np.array([input_shape[1], input_shape[2]])
    for l in layers:
        lost_this_layer = np.array([0, 0, 0, 0], dtype=np.float32)
        if isinstance(l, keras.layers.Conv2DTranspose):
            next_shape = last_shape * 2.0
            downsamples.append([0.5, 0.5])
        elif isinstance(l, keras.layers.UpSampling2D):
            next_shape = last_shape * 2.0
            downsamples.append([0.5, 0.5])
        elif isinstance(l, keras.layers.GlobalAvgPool2D) or isinstance(
            l, keras.layers.GlobalMaxPool2D
        ):
            next_shape = np.asarray([1, 1])
            downsamples.append([l.kernel_size, l.kernel_size])
        else:
            cur_stride = np.array(l.strides)

            if (
                isinstance(l, keras.layers.Conv2D)
                or isinstance(l, keras.layers.MaxPool2D)
                or isinstance(l, keras.layers.AvgPool2D)
            ):
                if isinstance(l, keras.layers.Conv2D):
                    kernel_size = l.kernel_size
                else:
                    kernel_size = l.pool_size
                if l.padding == "same":
                    next_shape = np.ceil(last_shape / cur_stride)
                elif l.padding == "valid":
                    next_shape = np.floor((last_shape - kernel_size) / cur_stride) + 1
                cutoff = (last_shape - kernel_size) % cur_stride
                if l.padding == "valid":
                    lost_this_layer[0] = (kernel_size[1] - cur_stride[1]) / 2
                    lost_this_layer[2] = (kernel_size[0] - cur_stride[0]) / 2
                    lost_this_layer[1] = (kernel_size[1] - cur_stride[1]) / 2 + cutoff[
                        0
                    ]
                    lost_this_layer[3] = (kernel_size[0] - cur_stride[0]) / 2 + cutoff[
                        1
                    ]
                elif l.padding == "same":
                    lost_this_layer[1] = cutoff[0]
                    lost_this_layer[3] = cutoff[1]
                else:
                    raise Exception(str(l.padding), str(l))
            downsamples.append(cur_stride)

        last_shape = next_shape
        lost.append(lost_this_layer)

    downsamples = [np.array([float(x), float(y)]) for x, y in downsamples]
    lost = [x.astype(float) for x in lost]

    for i in range(1, len(downsamples)):
        downsamples[i] *= downsamples[i - 1]
        lost[i][0:2] *= downsamples[i - 1][0]
        lost[i][2:] *= downsamples[i - 1][1]

    lost_total = np.array(lost).sum(axis=0).astype(np.float32).tolist()
    lost_total[0::2] = np.floor(lost_total[0::2])
    lost_total[1::2] = np.ceil(lost_total[1::2])

    return np.array(lost_total, dtype=np.int32), downsamples[-1]
