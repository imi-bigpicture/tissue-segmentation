import os

import numpy as np
import skimage.io as io

from tissue_segmentation import create_tissue_mask


def test_create_tissue_mask():
    # the test data does not have the correct resolution, which should ideally be around 8 microns
    # - network was trained using 8 microns resolution
    # - additionally the dilation of the mask has a default disk size of 32 pixels
    #   (can be adhusted using dilation_disk_size parameter)
    image = io.imread(os.path.join(os.path.dirname(__file__), "data", "0.jpg"))
    mask = create_tissue_mask(image)
    mask = 255 * mask.astype(np.uint8)
    mask_ref = io.imread(
        os.path.join(os.path.dirname(__file__), "data", "0_mask_ref.png")
    )
    assert np.sum(mask) == np.sum(mask_ref)
