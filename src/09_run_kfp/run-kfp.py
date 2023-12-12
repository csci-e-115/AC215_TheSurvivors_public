import os
import random
import string
import argparse
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_REGION = os.environ["GCP_REGION"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WHYLABS_DEFAULT_ORG_ID = os.environ["WHYLABS_DEFAULT_ORG_ID"]
WHYLABS_API_KEY = os.environ["WHYLABS_API_KEY"]
WHYLABS_DEFAULT_DATASET_ID = os.environ["WHYLABS_DEFAULT_DATASET_ID"]
GIT_LOGIN = os.environ["GIT_LOGIN"]
GCS_BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{GCS_BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
NUM_GPU = os.environ["NUM_GPU"]
TEMPLATE_PATH = "pipeline.yaml"
DVC_TAG_DEFAULT = "dataset_v09"

C1_ENVVAR = f"GCS_BUCKET_NAME={GCS_BUCKET_NAME} "
C2_ENVVAR = f"GCS_BUCKET_NAME={GCS_BUCKET_NAME} "+\
            f"WHYLABS_DEFAULT_ORG_ID={WHYLABS_DEFAULT_ORG_ID} "+\
            f"WHYLABS_API_KEY={WHYLABS_API_KEY} " +\
            f"WHYLABS_DEFAULT_DATASET_ID={WHYLABS_DEFAULT_DATASET_ID} "
C3_ENVVAR = ""

C1_IMG = "crispinnosidam/01-get-raw-data"
C2_IMG = "crispinnosidam/02-preprocess-data"
C3_IMG = "crispinnosidam/03-train-model"
C4_IMG = "sahilsakhuja/dermaid-versioning"

C1_PARAM=""
C2_PARAM=""

def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


@dsl.container_component
def get_raw_data():
    return dsl.ContainerSpec(
        image=C1_IMG,
        command=[
        ],
        args=[
            "-c",
            f"{C1_ENVVAR} pipenv run python get_data.py {C1_PARAM}"
        ]
    )

@dsl.container_component
def preprocess_data():
    return dsl.ContainerSpec(
        image=C2_IMG,
        command=[
        ],
        args=[
            "-c",
            f"{C2_ENVVAR} pipenv run python preprocess_data.py {C2_PARAM}"
        ]
    )

@dsl.container_component
def version_data(DVC_TAG: str):
    return dsl.ContainerSpec(
        image=C4_IMG,
        command=[
        ],
        args=[GCP_PROJECT, GCS_BUCKET_NAME, GIT_LOGIN, DVC_TAG]
    )


@dsl.container_component
def train_model(DVC_TAG: str):
    return dsl.ContainerSpec(
        image=C3_IMG,
        command=[
        ],
        args=[GCP_PROJECT, GCS_BUCKET_NAME, GIT_LOGIN, WANDB_API_KEY, DVC_TAG, NUM_GPU]
    )

def main(args=None):
    print("CLI Arguments:", args)

    @dsl.pipeline
    def dermaid_pl():
        container_tasks = []
        if 1 in args.run_containers:
            container_tasks.append(get_raw_data().set_display_name("01-get_raw-data"))
        if 2 in args.run_containers:
            container_tasks.append(preprocess_data().set_display_name("02-preprocess-data")\
                .set_cpu_limit('16')\
                .set_memory_limit('128G')\
                .add_node_selector_constraint("NVIDIA_TESLA_T4")\
                .set_gpu_limit('4'))
        if 4 in args.run_containers:
            container_tasks.append(version_data(DVC_TAG = args.tag).set_display_name("08-versioning"))
        if 3 in args.run_containers:
            container_tasks.append(train_model(DVC_TAG = args.tag).set_display_name("03-train-model")\
                                   .set_cpu_limit('16')\
                                   .set_memory_limit('128G')\
                                   .add_node_selector_constraint("NVIDIA_TESLA_T4")\
                                   .set_gpu_limit('4'))

        print(container_tasks)
        for task_no, task in enumerate(container_tasks):
            #print(f"task_no={task_no} {container_tasks[task_no].__dict__}")
            if task_no == 0: continue
            task=task.after(container_tasks[task_no-1])


    # Build yaml file for pipeline
    compiler.Compiler().compile(dermaid_pl, package_path=TEMPLATE_PATH)

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, location=GCP_REGION, staging_bucket=GCS_BUCKET_URI)

    job = aip.PipelineJob(
        display_name="dermaid-pipeline" + generate_uuid(),
        template_path=TEMPLATE_PATH,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    job.submit(service_account=GCS_SERVICE_ACCOUNT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-r",
        "--run_containers",
        action="append",
        type=int,
        required=True,
        help="Containers to run, can specify more than once",
    )

    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default=DVC_TAG_DEFAULT,
        help="Tag for DVC Versioning",
    )

    main(parser.parse_args())