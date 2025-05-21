import os
import boto3
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlparse


import os
import boto3
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlparse


def download_s3_from_uri(
    s3_uri: str, dotenv_path: str, local_dir: str = "datasets"
) -> None:
    """
    Downloads a file or all files in a folder from S3 to a local directory,
    skipping any files that already exist.

    :param s3_uri: Full S3 URI (e.g. s3://bucket-name/path/to/file_or_folder/)
    :param dotenv_path: Path to the .env file containing AWS credentials
    :param local_dir: Local directory where files should be saved
    """
    # Parse S3 URI
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket_name = parsed.netloc
    key = parsed.path.lstrip("/")

    # Load AWS credentials from .env
    load_dotenv(dotenv_path=Path(dotenv_path))
    access_key: Optional[str] = os.getenv("ACCESS_KEY_ID")
    secret_key: Optional[str] = os.getenv("SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        raise ValueError("Missing ACCESS_KEY_ID or SECRET_ACCESS_KEY in .env file")

    # Set up S3 client
    s3 = boto3.client(
        "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )
    os.makedirs(local_dir, exist_ok=True)

    #if key ends with /, treat as folder; else, could be file
    if not key or key.endswith("/"):
        # Folder download
        print(f"Listing objects with prefix: {key}")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=key):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                filename = os.path.basename(s3_key)
                if not filename:
                    continue  # Skip folders

                local_path = os.path.join(local_dir, filename)
                if os.path.exists(local_path):
                    print(f"Already exists: {local_path}")
                    continue

                print(f"Downloading {s3_key} to {local_path}")
                s3.download_file(bucket_name, s3_key, local_path)
    else:
        print('Try single file download')
        # Try single file download
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)

        if not os.path.exists(local_path):
            try:
                print(f"Downloading file {key} to {local_path}")
                s3.download_file(bucket_name, key, local_path)
            except s3.exceptions.NoSuchKey:
                raise FileNotFoundError(f"File not found: {s3_uri}")
        else:
            print(f"Already exists: {local_path}")

    print("Download complete.")



def main() -> None:
    #s3_uri = "s3://zrive-ds-data/groceries/sampled-datasets/"
    s3_uri = "s3://zrive-ds-data/groceries/box_builder_dataset/"

    dotenv_path = "/home/antonio/zrive-ds/.env"
    local_dir = "/home/antonio/zrive-ds/src/module_2/groceries_datasets2/raw/"

    try:
        download_s3_from_uri(
            s3_uri=s3_uri, dotenv_path=dotenv_path, local_dir=local_dir
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
