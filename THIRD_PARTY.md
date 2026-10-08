# Sources and licenses

Deployment and acceptance scripts: Apache-2.0. vLLM source, the renderer override, and derived patches retain their Apache-2.0 headers; see `source/LICENSE-vllm`.

Model metadata and reference code originate from DeepSeek-V4-Flash-Vision-Exp. The upstream MIT license is included at `source/model-metadata/LICENSE`. Model weights are external and are not distributed in this package.

The SM80 vision port includes work based on publicly available PixelML patches; source revisions and inherited provenance are recorded in `source/provenance.json`. Synthetic image fixtures and client checks were adapted from the glm-5.3-flash-a100 deployment tests.

CUDA, PyTorch, NVIDIA user-space libraries, Python packages, and all other third-party components remain subject to their respective licenses included in the container. The deployment-script license does not replace those licenses.

R3.7 ports the focused vLLM changes listed with exact revisions in `source/provenance.json`. `tests/upstream/test_responses_utils.py` and `test_streaming_events.py` derive from vLLM PR #57322 at `89c71911c736b6ba47e204db52c72a8a8b9048f2`, retaining Apache-2.0/SPDX headers. Import paths were adapted to the frozen runtime. The local regression and GPU gate tests do not imply upstream certification of this deployment.

R3.8 additionally ports focused vLLM changes #46257, #53739, #58291, #58296 and #58295 as listed in `source/provenance.json`. The carried Python code retains upstream Apache-2.0/SPDX notices. The offline image also includes pre-existing third-party runtime dependencies; `source/python-dependencies.json` records their versions.
