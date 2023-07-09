import json
import requests
import configparser
import pandas as pd
from transformers import pipeline
import argparse
import logging


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("log/pipeline.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def run_feature_extraction_pipeline(model_id, hf_token, texts):
    api_url = (
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    )
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as errh:
        logger.error(f"HTTP Error: {errh}")
        raise
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logger.error(f"Request Exception: {err}")
        raise


def save_embeddings_to_csv(embeddings, output_file):
    try:
        embeddings.to_csv(output_file, index=False)
        logger.info(f"Embeddings saved to {output_file}. Shape of embeddings: {embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to save embeddings to {output_file}: {str(e)}")


def embedding_query(json_file_path, model_id, hf_token, output_file):
    with open(json_file_path, "r") as file:
        texts = json.load(file)

    sentences = [entry["sentence"] for entry in texts["sentences"]]

    try:
        output = run_feature_extraction_pipeline(model_id, hf_token, sentences)
        embeddings = pd.DataFrame(output)
        save_embeddings_to_csv(embeddings, output_file)
    except Exception as e:
        logger.error(f"Embedding query failed: {str(e)}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="CLI Application for embedding query.")
    parser.add_argument("config_file", help="Path to the config file")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = configparser.ConfigParser()
    config.read(args.config_file)

    json_file_path = config.get("Settings", "json_file_path")
    model_id = config.get("Settings", "model_id")
    hf_token = config.get("Settings", "hf_token")
    output_file = config.get("Settings", "output_file")

    embedding_query(json_file_path, model_id, hf_token, output_file)


if __name__ == "__main__":
    logger = setup_logger()
    main()
