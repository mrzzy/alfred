#
# Alfred
# Entrypoint
#


from pathlib import Path
from langchain_anthropic import ChatAnthropic
from argparse import ArgumentParser


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser(description="Alfred LLM Tutoring assistant")
    parser.add_argument(
        "content_dir",
        type=Path,
        help="Path to a directory of course content documents used by Alfred "
        "to understand the course. Document formats supported: PDF",
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="anthropic",
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
    # expects ANTHROPIC_API_KEY env var to pass api key for authenticating with anthropic api
    if args.provider == "anthropic":
        model = ChatAnthropic(
            model_name=args.model,
            temperature=0,
            max_tokens_to_sample=1024,
            timeout=None,
            max_retries=2,
            stop=None,
        )
    else:
        raise ValueError(f"Unsupported model provider: {args.provider})")

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = model.invoke(messages)
    print(ai_msg.content)
