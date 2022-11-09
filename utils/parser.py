import argparse
import os


class Parser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI argument parser.
    More options can be added specific by passing this object and calling
    ''add_arg()'' or add_argument'' on it.
    :param add_preprocessing_args:
        (default False) initializes the default arguments for Data Preprocessing package.
    """

    def __init__(
        self, add_preprocessing_args=False,
        description='Command Line parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
            add_help=add_preprocessing_args,
        )
        self.home_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['HOME_DIR'] = self.home_dir

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_preprocessing_args:
            self.add_preprocessing_args()

    def add_preprocessing_args(self, args=None):
        """
        Add common BLINK args across all scripts.
        """
        parser = self.add_argument_group("Preprocessing Arguments")
        parser.add_argument(
            "--data_home_dir", type=str, help="Directory where the data is stored"
        )

        parser.add_argument(
            "--dataset_name", type=str, help="name of AND dataset that you want to preprocess"
        )