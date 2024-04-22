# Speech-Text Alignment

Scripts to align speech audio with their text transcriptions in time. 

## Setup - OSX

```bash
brew install ffmpeg sox
```

```bash
conda create -n sta python=3.11 -y &&
    conda activate sta &&
    pip install -r requirements.txt
```

## Setup - Linux - `TODO`

> Complete section when running on Artemis / Poseidon. CUDA information below:
> `NVIDIA-SMI 525.89.02    Driver Version: 525.89.02    CUDA Version: 12.0`

## Performing Alignment

> ...

## Dev Containers - `TODO`

We provide a _devcontainer.json_ to allow for development inside a containerised environment via VSCode. This enables installation of Linux binary dependencies such as [FFmpeg](https://ffmpeg.org/) or [SoX](https://en.wikipedia.org/wiki/SoX) via `apt` without requiring sudo permissions on the host machine. 

References:
- the [devcontainers/cli documentation](https://github.com/devcontainers/cli/blob/c1c8b08263c6dca7cd79c97a2d0bc581fcef4f6c/README.md#try-out-the-cli)
  - specifically the the subsection titled [Try out the CLI](https://github.com/devcontainers/cli/tree/main?tab=readme-ov-file#try-out-the-cli) to get set up
  - see also the [documentation on the Dev Containers CLI](https://containers.dev/implementors/reference/)
- the [Specification: Dev Container metadata reference](https://containers.dev/implementors/json_reference/) to customize the dev container configuration
- Many [Development Container Templates](https://containers.dev/templates) are available for immediate use
  - see for example the [template to use `ubuntu:jammy`](https://github.com/devcontainers/templates/tree/main/src/ubuntu)
- [how to use a Dockerfile or Docker Compose YAML config](https://containers.dev/guide/dockerfile) in place of a pre-built image on a container registry

VSCode provides substantial integration for Dev Containers. See their overview of [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers), [quickstart tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) and related sections in their documentation (e.g. on [creating a dev container](https://code.visualstudio.com/docs/devcontainers/create-dev-container)). 

## TODO

- Fix dev container integration