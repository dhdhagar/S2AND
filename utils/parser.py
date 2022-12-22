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
        self,
            add_preprocessing_args=False,
            add_training_args=False,
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

        if add_training_args:
            self.add_training_args()

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

    def add_training_args(self, args=None):
        self.add_argument(
            "--wandb_sweep_name", type=str, required=True,
            help="Wandb sweep name",
        )
        self.add_argument(
            "--wandb_sweep_id", type=str,
            help="Wandb sweep id (optional -- if run is already started)",
        )
        self.add_argument(
            "--wandb_sweep_method", type=str, default="bayes",
            help="Wandb sweep method (bayes/random/grid)",
        )
        self.add_argument(
            "--wandb_project", type=str, default="missing-values",
            help="Wandb project name",
        )
        self.add_argument(
            "--wandb_entity", type=str, default="dhdhagar",
            help="Wandb entity name",
        )
        self.add_argument(
            "--wandb_sweep_params", type=str, required=True,
            help="Path to wandb sweep parameters JSON",
        )
        self.add_argument(
            "--wandb_sweep_metric_name", type=str, default="dev_auroc",
            help="Wandb sweep metric to optimize (dev_auroc/dev_loss/dev_f1)",
        )
        self.add_argument(
            "--wandb_sweep_metric_goal", type=str, default="maximize",
            help="Wandb sweep metric goal (maximize/minimize)",
        )
        self.add_argument(
            "--wandb_no_early_terminate", action="store_true",
            help="Whether to prevent wandb sweep early terminate or not",
        )
        self.add_argument(
            "--wandb_max_runs", type=int, default=600,
            help="Maximum number of runs to try in the sweep",
        )
