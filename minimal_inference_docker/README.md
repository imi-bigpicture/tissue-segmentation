# Minimal Inference Docker
The docker image outputs the biggest connected tissue segmentation, which can be used to ignore control tissue in the image.

## Build docker image

```shell
./build.sh
```

## Process tiff files

Running

```shell
docker run -v /some/folder/containing/tiffs:/data --rm minimal_inference_docker
```

searches for tiff files (`*.tiff`) in the mounted folder and saves the used level image and the resulting mask next to it (`*.png` and `*_mask.png`).
