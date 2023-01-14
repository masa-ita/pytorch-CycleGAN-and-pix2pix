from mlflow import MlflowClient

class MlflowWriter():
    def __init__(self, experiment_name, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_args(self, args):
        for param_name, element in vars(args).items():
            self.log_param(param_name, element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step=None):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def log_artifacts(self, local_dir, artifact_path):
        self.client.log_artifacts(self.run_id, local_dir, artifact_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)
