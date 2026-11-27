# Imaging

AIVA's raster painting capability. The `aiva-imagegen` Python package provides
a client for the local MossInfer `moss-server` sidecar and registers the `image`
action with [Canvas](https://github.com/Project-AGi/canvas).

This package is a library and CLI, not a ROS node. The Telepathy receptor imports
it when drawing needs raster painting. Canvas remains usable without Imaging.

## Install and use

Install Canvas first, then install this checkout into the same environment:

```sh
python -m pip install -e ../canvas
python -m pip install -e .
imagegen check
imagegen paint "a foggy harbour at dawn" -o painting.png
```

`imagegen check` verifies the sidecar executable and required weights; it does
not generate an image. The model weights and MossInfer binary are separate
installations and are not included here.

Set `MOSS_PAINT_SERVER_BIN` and `MOSS_PAINT_WEIGHTS_ROOT` for their local paths.
Optional settings include `MOSS_PAINT_DEVICE` (default `1`),
`MOSS_PAINT_VRAM_MIB` (default `4096`), and `MOSS_PAINT_TIMEOUT_SECONDS`
(default `300`). `MOSS_PAINT_RELEASE=1` releases GPU memory after a painting.

The current integration's subprocess tests live with the Telepathy receptor;
they emulate the sidecar protocol without CUDA or model weights. Actual image
quality, VRAM usage and CUDA compatibility require a local generation run.

## License

See [LICENSE](LICENSE).
