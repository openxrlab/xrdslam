"""Download datasets and specific captures from the datasets."""
from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import tyro
from typing_extensions import Annotated

from slam.configs.base_config import PrintableConfig


@dataclass
class DatasetDownload(PrintableConfig):
    """Download a dataset."""

    capture_name = None

    save_dir: Path = Path('data/')
    """The directory to save the dataset to"""
    def download(self, save_dir: Path) -> None:
        """Download the dataset."""
        raise NotImplementedError


# pylint: disable=line-too-long
slam_downloads = {
    'f1_desk':
    'https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz',
    'f1_desk2':
    'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz',
    'f1_room':
    'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz',
    'f2_xyz':
    'https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz',
    'f3_office':
    'https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz',
    'replica': 'https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip',
    'neural_rgbd_data':
    'http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip',
    'apartment': 'https://cvg-data.inf.ethz.ch/nice-slam/data/Apartment.zip',
    'mh01': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip',
    'mh02': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip',
    'mh03': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip',
    'mh04': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip',
    'mh05': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip',
    'v101': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip',
    'v102': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip',
    'v103': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip',
    'v201': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip',
    'v202': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip',
    'v203': 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip',
    'all': None,
}

CaptureName = tyro.extras.literal_type_from_choices(slam_downloads.keys())


@dataclass
class SLAMDatasetDownload(DatasetDownload):
    """Download the dataset."""

    capture_name: CaptureName = 'f1_desk'

    def download(self, save_dir: Path):
        if self.capture_name == 'all':
            for capture_name in slam_downloads:
                if capture_name != 'all':
                    SLAMDatasetDownload(
                        capture_name=capture_name).download(save_dir)
            return

        assert (self.capture_name in slam_downloads
                ), f'Capture name {self.capture_name} not found \
            in {slam_downloads.keys()}'

        url = slam_downloads[self.capture_name]

        target_path = str(save_dir / self.capture_name)
        os.makedirs(target_path, exist_ok=True)

        format = url[-4:]

        download_path = Path(f'{target_path}{format}')
        tmp_path = str(save_dir / '.temp')
        shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)

        os.system(f'curl -L {url} > {download_path}')
        if format == '.tar':
            with tarfile.open(download_path, 'r') as tar_ref:
                tar_ref.extractall(str(tmp_path))
        elif format == '.zip':
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(str(tmp_path))
        elif format == '.tgz':
            with tarfile.open(download_path, 'r:gz') as tar_ref:
                tar_ref.extractall(str(tmp_path))
        else:
            raise NotImplementedError

        inner_folders = os.listdir(tmp_path)
        assert len(
            inner_folders
        ) == 1, 'There is more than one folder inside this zip file.'
        folder = os.path.join(tmp_path, inner_folders[0])
        shutil.rmtree(target_path)
        shutil.move(folder, target_path)
        shutil.rmtree(tmp_path)
        os.remove(download_path)


Commands = Annotated[SLAMDatasetDownload, tyro.conf.subcommand()]


def main(dataset: DatasetDownload, ):
    """Script to download existing datasets."""
    dataset.save_dir.mkdir(parents=True, exist_ok=True)

    dataset.download(dataset.save_dir)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color('bright_yellow')
    main(tyro.cli(Commands))


if __name__ == '__main__':
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
