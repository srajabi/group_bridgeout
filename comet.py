import os
from comet_ml import Experiment


def get_comet_experiment():
    comet_api_key = os.environ.get('COMET_API_KEY', 'dummy_key')
    project_name = "structuredsparsity"
    workspace = "srajabi"

    experiment_name = "text-silvertip"

    if comet_api_key:
        experiment = Experiment(
            api_key=comet_api_key, project_name=project_name, workspace=workspace, parse_args=False)
        experiment.set_name(experiment_name)
        experiment.display()
    else:
        experiment = Experiment(api_key='dummy_key', disabled=True)
    return experiment