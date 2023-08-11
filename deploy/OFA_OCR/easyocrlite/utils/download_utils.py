import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm.auto import tqdm

FILENAME = "craft_mlt_25k.pth"
URL = (
    "https://xc-models.oss-cn-zhangjiakou.aliyuncs.com/modelscope/studio/easyocr/craft_mlt_25k.zip"
)
MD5SUM = "2f8227d2def4037cdb3b34389dcf9ec1"
MD5MSG = "MD5 hash mismatch, possible file corruption"


logger = logging.getLogger(__name__)


def calculate_md5(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def print_progress_bar(t: tqdm) -> Callable[[int, int, Optional[int]], None]:
    last = 0

    def update_to(
        count: int = 1, block_size: int = 1, total_size: Optional[int] = None
    ):
        nonlocal last
        if total_size is not None:
            t.total = total_size
        t.update((count - last) * block_size)
        last = count

    return update_to


def download_and_unzip(
    url: str, filename: str, model_storage_directory: Path, verbose: bool = True
):
    zip_path = model_storage_directory / "temp.zip"
    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, disable=not verbose
    ) as t:
        reporthook = print_progress_bar(t)
        urlretrieve(url, str(zip_path), reporthook=reporthook)
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extract(filename, str(model_storage_directory))
    zip_path.unlink()


def prepare_model(model_storage_directory: Path, download=True, verbose: bool = True) -> bool:
    model_storage_directory.mkdir(parents=True, exist_ok=True)

    detector_path = model_storage_directory / FILENAME

    # try get model path
    model_available = False
    if not detector_path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {detector_path} and downloads disabled")
        logger.info(
            "Downloading detection model, please wait. "
            "This may take several minutes depending upon your network connection."
        )
    elif calculate_md5(detector_path) != MD5SUM:
        logger.warning(MD5MSG)
        if not download:
            raise FileNotFoundError(
                f"MD5 mismatch for {detector_path} and downloads disabled"
            )
        detector_path.unlink()
        logger.info(
            "Re-downloading the detection model, please wait. "
            "This may take several minutes depending upon your network connection."
        )
    else:
        model_available = True

    if not model_available:
        download_and_unzip(URL, FILENAME, model_storage_directory, verbose)
        if calculate_md5(detector_path) != MD5SUM:
            raise ValueError(MD5MSG)
        logger.info("Download complete")

    return detector_path