from __future__ import annotations

import tyro

from deepslam.configs.config_utils import convert_markup_to_ansi
from deepslam.configs.input_config import AnnotatedBaseConfigUnion
from deepslam.engine.xrdslamer import XRDSLAMer, XRDSLAMerConfig


def main(config: XRDSLAMerConfig) -> None:
    """Main function."""

    # set data type and path
    config.xrdslam.data = config.data
    config.xrdslam.data_type = config.data_type

    # print and save config
    config.print_to_terminal()
    config.save_config()

    slam = XRDSLAMer(config)
    slam.setup()
    slam.run()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        ))


if __name__ == '__main__':
    entrypoint()
