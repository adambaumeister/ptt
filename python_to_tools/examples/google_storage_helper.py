"""This example uses a simple Wrapper class for the Google Cloud Storage API, and shows how you can use agents and
the standard TaskFlowBehavior to implement a simple file uploader/downloader with Google Cloud Storage!

To use this example, you must set the following environment variables (or adapt it for your use case)

    * GOOGLE_CLOUD_PROJECT
"""
import os
import pathlib
from typing import Annotated

from google.cloud import storage
from google.cloud.storage import Bucket

from python_to_tools.ai.generic_models import ToolParameter
from python_to_tools.ai.utils import get_model_by_environment_variables
from python_to_tools.ptt import TaskFlowBehavior


class GoogleStorageWrapper:
    def __init__(self):
        self.client = storage.Client()

    def list_buckets(self):
        """List all the google cloud storage buckets. Buckets are used to store objects."""
        return ", ".join([b.name for b in self.client.list_buckets()])

    def auth_to_gcloud(self):
        """Authenticate to Google Cloud Storage. This function does nothing, as authentication isn't required when
        running using gcloud app default credentials!"""
        return "ok!"

    def upload_file(
            self,
            file: Annotated[pathlib.Path, ToolParameter(type="string", description="Path to the file to upload")],
            bucket_name: Annotated[
                str, ToolParameter(
                    type="string",
                    description="The name of the bucket to upload the file to. Buckets can be listed using list_buckets."
                )
            ],
    ):
        """Upload a given file to google cloud storage, to the given bucket."""
        file = pathlib.Path(file)
        if not file.is_file():
            raise FileNotFoundError(f"File {file} not found")

        bucket: Bucket = self.client.bucket(bucket_name)
        bucket.blob(file.name).upload_from_filename(file)
        return f"File {file.name} uploaded to {bucket_name} successfully."

def list_local_files(
        directory: Annotated[str, ToolParameter(type="string", description="Directory to look for files")]
):
    """List all the files in the given directory."""
    path = pathlib.Path(directory)
    if not path.is_dir():
        raise ValueError(f"{directory} is not a directory!")

    return os.listdir(path)

def main():
    storage = GoogleStorageWrapper()
    model = get_model_by_environment_variables()
    behavior = TaskFlowBehavior(root_ai_model=model)
    behavior.handler_agent.add_tool(list_local_files)
    behavior.handler_agent.add_tool(storage.list_buckets)
    behavior.handler_agent.add_tool(storage.upload_file)
    behavior.handler_agent.add_tool(storage.auth_to_gcloud)

    print(
        behavior.resolve_from_text("Upload the file example_file.txt to google cloud using the bucket 'bowyridesblog' thanks")
    )

if __name__ == '__main__':
    main()
