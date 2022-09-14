from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

hyperparameters = {"epochs": "2", "batch-size": "64", "test-batch-size": "128", "learning-rate": 0.001}  # TODO: Include the hyperparameters your script will need over here.

estimator = PyTorch(
    entry_point="pytorch_cifar.py",
    base_job_name="sagemaker-script-mode_exercise",
    role=get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.large",
    hyperparameters=hyperparameters,
    framework_version="1.8",
    py_version="py36",
)  # TODO: Create your estimator here. You can use Pytorch or any other framework.

#TODO: Start Training
estimator.fit(wait=True)