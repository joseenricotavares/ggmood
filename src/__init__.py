import mlflow
import os

os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'
os.makedirs("../mlruns", exist_ok=True)
mlflow.set_tracking_uri("../mlruns")

from .data.resample import *
from .models.train import *
from .models.evaluation import *