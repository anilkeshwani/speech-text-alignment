# Speech-Text Alignment

Scripts to align speech audio with their text transcriptions in time. 

In addition to this README, broader [documentation can be found in docs/](/docs/).

## Setup

### Clone Repository

```bash
git clone git@github.com:anilkeshwani/speech-text-alignment.git && 
    cd speech-text-alignment &&
    git submodule update --init --recursive --progress
```

### Set Up Environment

Ensure the necessary binary requirements are installed:

```bash
apt install sox ffmpeg
```

Install the package and with it all dependencies including useful dependencies for development; specified via "dev" option to `pip install`.

```bash
conda create -n sardalign python=3.10.6 -y &&
    pip install pip==24.0 &&
    conda activate sardalign &&
    pip install -e .["dev"]
```

> Note: We do not install the _dataclasses_ library as per the [fairseq MMS README](https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/examples/mms/data_prep/README.md) it ships out of the box with Python 3.10.6.

<details>
  <summary>Note: When running on Artemis / Poseidon, ensure support for CUDA is provided.</summary>
  
  At the time of writing, NVIDIA / CUDA drivers were:
  - NVIDIA-SMI: 525.89.02
  - Driver Version: 525.89.02
  - CUDA Version: 12.0
  
</details>

## Data Processing and Performing Tasks

Documentation for performing data processing steps or tasks is found in [scripts/README.md](/scripts/README.md).
