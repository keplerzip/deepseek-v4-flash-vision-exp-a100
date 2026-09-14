# Sources and licenses

Deployment and acceptance scripts: Apache-2.0. vLLM source, the renderer override, and derived patches retain their Apache-2.0 headers; see `source/LICENSE-vllm`.

Model metadata and reference code originate from DeepSeek-V4-Flash-Vision-Exp. The upstream MIT license is included at `source/model-metadata/LICENSE`. Model weights are external and are not distributed in this package.

The SM80 vision port includes work based on publicly available PixelML patches; source revisions and inherited provenance are recorded in `source/provenance.json`. Synthetic image fixtures and client checks were adapted from the glm-5.3-flash-a100 deployment tests.

CUDA, PyTorch, NVIDIA user-space libraries, Python packages, and all other third-party components remain subject to their respective licenses included in the container. The deployment-script license does not replace those licenses.
