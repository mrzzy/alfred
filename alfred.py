#
# Alfred
# Entrypoint
#


from argparse import ArgumentParser
from pathlib import Path
from typing import List, cast

from langchain_milvus import Milvus


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser(description="Alfred LLM Tutoring assistant")
    parser.add_argument(
        "catalog",
        type=Path,
        help="Path to built Knowledge Catalogue Alfred will reference.",
    )
    parser.add_argument(
        "-o",
        "-provider",
        type=str,
        default="anthropic",
        choices=["anthropic"],
        help="Provider of the LLM model. Ensure that credentials are supplied for the provider.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="claude-3-haiku-20240307",
        help="Name of the LLM model to use to generate responses.",
    )
    args = parser.parse_args()

    # setup LLM model
    if args.provider == "anthropic":
        # expects ANTHROPIC_API_KEY env var to pass api key for authenticating with anthropic api
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model_name=args.model,
            temperature=0,
            max_tokens_to_sample=1024,
            timeout=None,
            max_retries=2,
            stop=None,
        )
    else:
        raise ValueError(f"Unsupported model provider: {args.model_provider})")

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    # ai_msg = model.invoke(messages)
    # print(ai_msg.content)
