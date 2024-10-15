#
# Alfred
# Entrypoint
#


from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import List, Optional, cast

from huggingface_hub.hf_api import json
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import (
    Runnable,
    RunnableAssign,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_milvus import Milvus
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.graph import RunnableConfig
from pydantic import BaseModel, Field

from catalog import EMBEDDING_JSON, VECTOR_STORE
from embedding import Embedding, build_embedding


# Model Input / Response Schemas
class Metadata(BaseModel):
    doc_id: str = Field(description="Snippet ID of the document")
    source: str = Field(description="Source of the document")
    page: Optional[int] = Field(description="Page number (if applicable")


class Document(BaseModel):
    content: str = Field(description="Text content of the document")
    metadata: Metadata


class Source(BaseModel):
    reference: Metadata
    relevance: str = Field(
        description="Brief explanation of how this snippet is relevant to the answer"
    )


class Output(BaseModel):
    response: str = Field(description="Your response to the the prompt")
    sources: list[Source]


def extract_json(message: AIMessage) -> AIMessage:
    content = str(message.content)
    # assume bonds of json are delimited by brackets
    message.content = content[content.find("{") : content.rfind("}") + 1]
    return message


# Graph State schema
class State(MessagesState):
    pass


def build_chain(vector_store: VectorStore, model: BaseChatModel) -> Runnable:
    # build prompt template
    sys_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                args.system_prompt,
                input_variables=["messages", "documents"],
                partial_variables={
                    "doc_schema": json.dumps(Document.model_json_schema()),
                    "out_schema": json.dumps(Output.model_json_schema()),
                },
            ),
            HumanMessagePromptTemplate.from_template("User Prompt:\n {prompt}"),
        ]
    )

    # build langchain
    doc_retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    return (
        RunnableAssign(
            RunnableParallel(
                {
                    # extract prompt from last message, assumed to be from the human user
                    "prompt": RunnableLambda(
                        lambda state: str(state["messages"][-1].content)
                    ),
                    "prior_messages": RunnableLambda(
                        lambda state: state["messages"][:-1]
                    ),
                }
            )
        )
        | RunnableAssign(
            RunnableParallel(
                {
                    "documents": RunnableLambda(
                        lambda state: doc_retriever.invoke(state["prompt"])
                    ),
                }
            )
        )
        | sys_prompt
        | model
        # clip extract elaboration produced by the model to only JSON returned
        | RunnableLambda(extract_json)
    )


def run_chain(state: State, config: RunnableConfig, chain: Runnable) -> dict:
    ai_message = chain.invoke(state)
    # state update: append returned message to messages
    return {"messages": [ai_message]}


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser(description="Alfred LLM Tutoring assistant")
    parser.add_argument(
        "catalog",
        type=Path,
        help="Path to built Knowledge Catalogue Alfred will reference.",
    )
    parser.add_argument(
        "-p",
        "--provider",
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
    parser.add_argument(
        "-s",
        "--system-prompt",
        type=Path,
        default=Path("prompts") / "qna.md",
        help="Path to the system prompt that instruct's alfreds behaviour.",
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

    # load knowledge catalogue vector store
    with open(args.catalog / EMBEDDING_JSON, "r") as f:
        embedding = build_embedding(Embedding(**json.load(f)))
    vector_store = Milvus(
        embedding_function=embedding,
        connection_args={"uri": str(args.catalog / VECTOR_STORE)},
    )

    chain = build_chain(vector_store, model)

    # build & compile execution graph
    graph = StateGraph(state_schema=State)
    graph.add_edge(START, "chain")
    graph.add_node("chain", partial(run_chain, chain=chain))
    graph.add_edge("chain", END)

    memory = MemorySaver()
    run = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "0"}}
