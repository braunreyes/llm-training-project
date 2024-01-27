import os

import typer
from langchain.chains import LLMChain, SequentialChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from rich.prompt import Prompt

# app = typer.Typer()

API_KEY = os.getenv("OPEN_AI_KEY")
LLM = ChatOpenAI(api_key=API_KEY)

# CODE_PROMPT = PromptTemplate(
#     template="Write a very short {language} function that will {task}",
#     input_variables=["language", "task"],
# )

# TEST_PROMPT = PromptTemplate(
#     template="Write a test for the following {language} code:\n{code}",
#     input_variables=["language", "task"],
# )

# CODE_CHAIN = LLMChain(llm=LLM, prompt=CODE_PROMPT, output_key="code")
# TEST_CHAIN = LLMChain(llm=LLM, prompt=TEST_PROMPT, output_key="test")

# SEQ = SequentialChain(
#     chains=[CODE_CHAIN, TEST_CHAIN],
#     input_variables=["language", "task"],
#     output_variables=["code", "test"],
# )

CHAT_TEMPLATE = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are an assistant that answers questions with short responses"
        ),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

LLM_CHAT = LLMChain(llm=LLM, prompt=CHAT_TEMPLATE)


def main():
    while True:
        content = Prompt.ask(">> ")
        result = LLM_CHAT({"content": content})
        print(result["text"])


if __name__ == "__main__":
    typer.run(main)
