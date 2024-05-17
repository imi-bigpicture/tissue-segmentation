from tensorflow import keras
import numpy as np
from glob import glob
import tifffile
import skimage
from PIL import Image
from scipy import ndimage


def get_padding_and_downsamples(input_shape, model):
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


model = keras.models.load_model("/app/fcnn_c.keras")
# print(model.summary())

# parameters
level = 4

for image_path in glob("/data/*.tiff"):
    print(f"processing {image_path}")
    image = tifffile.imread(image_path, key=level)
    image_preprocessed = image / 255.0
    result = model.predict(np.expand_dims(image_preprocessed, axis=0))
    mask = result[0, :, :, 1]
    # rescale
    pads, downsamples = get_padding_and_downsamples(image_preprocessed.shape, model)
    mask = skimage.transform.rescale(
        mask,
        (downsamples[0], downsamples[1]),
        preserve_range=True,
        order=True,
        mode="edge",
    )
    mask = np.pad(mask, ((pads[2], pads[3]), (pads[0], pads[1])), "constant")

    # threshold
    confidence_threshold = 0.8
    mask[mask < confidence_threshold] = 0.0
    mask[mask >= confidence_threshold] = 1.0

    # apply object size filter
    min_object_size = 50000

    labeled_mask, num_features = ndimage.label(mask)
    object_sizes = ndimage.sum(mask, labeled_mask, range(num_features + 1))
    filtered_mask = np.zeros_like(mask)
    for i, size in enumerate(object_sizes):
        if size >= min_object_size:
            filtered_mask[labeled_mask == i] = 1
    mask_filtered = filtered_mask

    # apply hole filling
    max_hole_size = 50000

    mask_inverted = np.logical_not(mask_filtered)

    labeled_mask, num_features = ndimage.label(mask_inverted)
    object_sizes = ndimage.sum(mask_inverted, labeled_mask, range(num_features + 1))
    filtered_mask = np.zeros_like(mask)
    for i, size in enumerate(object_sizes):
        if size <= max_hole_size:
            filtered_mask[labeled_mask == i] = 1
    filled_holes_mask = filtered_mask

    # save level image
    image_pil = Image.fromarray(image)
    image_pil.save(image_path.replace(".tiff", f".png"))
    # save result image
    mask_image = Image.fromarray((255 * filled_holes_mask).astype(np.uint8))
    mask_image.save(image_path.replace(".tiff", "_mask.png"))
