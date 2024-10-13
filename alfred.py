#
# Alfred
# Entrypoint
#


from langchain_anthropic import ChatAnthropic


if __name__ == "__main__":
    # expects ANTHROPIC_API_KEY env var to pass api key for authenticating with anthropic api
    llm = ChatAnthropic(
        model_name="claude-3-haiku-20240307",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
