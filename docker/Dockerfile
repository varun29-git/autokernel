# Supply a CUDA devel/Python image pinned by digest; no default floating image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN python -m pip install --no-cache-dir triton==3.4.0
WORKDIR /tmp
