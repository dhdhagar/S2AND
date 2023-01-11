import argparse
import os


class Parser(argparse.ArgumentParser):
    def __init__(
        self,
        add_preprocessing_args=False,
        add_training_args=False,
        description='ProbEntRes command-line parser',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
        )
        # TODO: Check if `home_dir` is used for anything
        self.home_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['HOME_DIR'] = self.home_dir

        if add_preprocessing_args:
            self.add_preprocessing_args()

        if add_training_args:
            self.add_training_args()

    def add_preprocessing_args(self):
        """
        Add arguments related to data preprocessing.
        """
        parser = self.add_argument_group("Preprocessing Arguments")
        parser.add_argument(
            "--data_home_dir", type=str, help="Directory where the data is stored"
        )
        parser.add_argument(
            "--dataset_name", type=str, help="name of AND dataset that you want to preprocess"
        )

    def add_training_args(self):
        """
        Add arguments related to model training.
        """
        parser = self.add_argument_group("Training Arguments")
        parser.add_argument(
            "--dataset", type=str, default="pubmed",
            help="Dataset name (pubmed/qian/zbmath/arnetminer)",
        )
        parser.add_argument(
            "--dataset_random_seed", type=int, default=1,
            help="S2AND random seed for dataset splits (1/2/3/4/5)",
        )
        parser.add_argument(
            "--run_random_seed", type=int, default=17,
            help="Random seed for everything except the dataset",
        )
        parser.add_argument(
            "--wandb_sweep_name", type=str,
            help="Wandb sweep name",
        )
        parser.add_argument(
            "--wandb_tags", type=str,
            help="Comma-separated list of tags to add to a wandb run"
        )
        parser.add_argument(
            "--wandb_group", type=str,
            help="Group name to add the wandb run to"
        )
        parser.add_argument(
            "--wandb_sweep_id", type=str,
            help="Wandb sweep id (optional -- if run is already started)",
        )
        parser.add_argument(
            "--wandb_sweep_method", type=str, default="bayes",
            help="Wandb sweep method (bayes/random/grid)",
        )
        parser.add_argument(
            "--wandb_project", type=str, default="prob-ent-resolution",
            help="Wandb project name",
        )
        parser.add_argument(
            "--wandb_entity", type=str,
            help="Wandb entity name",
        )
        parser.add_argument(
            "--wandb_run_params", type=str,
            help="Path to wandb single-run parameters JSON",
        )
        parser.add_argument(
            "--wandb_sweep_params", type=str,
            help="Path to wandb sweep parameters JSON",
        )
        parser.add_argument(
            "--wandb_sweep_metric_name", type=str, default="dev_b3_f1",
            help="Wandb sweep metric to optimize (dev_vmeasure / dev_b3_f1)",
        )
        parser.add_argument(
            "--wandb_sweep_metric_goal", type=str, default="maximize",
            help="Wandb sweep metric goal (maximize/minimize)",
        )
        parser.add_argument(
            "--wandb_no_early_terminate", action="store_true",
            help="Whether to prevent wandb sweep early terminate or not",
        )
        parser.add_argument(
            "--wandb_max_runs", type=int, default=600,
            help="Maximum number of runs to try in the sweep",
        )
        parser.add_argument(
            "--cpu", action='store_true',
            help="Run on CPU regardless of CUDA-availability",
        )
        parser.add_argument(
            "--save_model", action='store_true',
            help="Whether to save the model (locally in the wandb run dir & in wandb cloud storage)",
        )
        parser.add_argument(
            "--load_model_from_wandb_run", type=str,
            help="Load model state_dict from a previous wandb run",
        )
        parser.add_argument(
            "--load_model_from_fpath", type=str,
            help="Load model state_dict from a local file path",
        )
        parser.add_argument(
            "--silent", action='store_true',
            help="Whether to prevent outputting logging statements to the console or not",
        )
        parser.add_argument(
            "--eval_only_split", type=str,
            help="Run script in inference-only mode on a particular data split (train / dev / test)",
        )
        parser.add_argument(
            "--skip_initial_eval", action='store_true',
            help="Whether to skip dev evaluation before training starts",
        )
