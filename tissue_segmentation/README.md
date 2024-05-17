# Tissue Segmentation Python Package
The python package provides functionalities for tissue segmentation with and without post processing, which only chooses the biggest connected segmentation.

## Quickstart

Install the `tissue-segmentation` package 

```shell
pip3 install tissue-segmentation
```

and use it like this:

```python
from tissue_segmentation import create_tissue_mask

mask = create_tissue_mask(image)
```

- `image` should be a numpy array holding a rgb image with values from 0-255, pixel size should be around 0.008 mm
- optional: `dilation_disk_size` parameter (the dilation of the mask has a default disk size of 32 pixels)

## Build as python package

Use 
```
poetry install
poetry build --format=wheel
```
to build a local `.whl` file. It is stored in `dist/`