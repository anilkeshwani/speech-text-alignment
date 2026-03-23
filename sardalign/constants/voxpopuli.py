import os


VOXPOPULI_HF_DATASET_REPO: str = "facebook/voxpopuli"
# NOTE VOXPOPULI_LOCAL_DISK_DIR specific to our mnt fs
# NOTE must concat dataset subset to path e.g. Path(VOXPOPULI_LOCAL_DISK_DIR) / "en"
# subsets comprise: cs  de  en  es  et  fi  fr  hr  hu  it  lt  nl  pl  ro  sk  sl
VOXPOPULI_LOCAL_DISK_DIR: str = os.environ.get(
    "VOXPOPULI_LOCAL_DISK_DIR", "/mnt/scratch-artemis/shared/datasets/facebook/voxpopuli/"
)
