import os

import typer
from langchain.chains import LLMChain, SequentialChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

app = typer.Typer()

API_KEY = os.getenv("OPEN_AI_KEY")
LLM = OpenAI(api_key=API_KEY)

CODE_PROMPT = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

TEST_PROMPT = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "task"],
)

CODE_CHAIN = LLMChain(llm=LLM, prompt=CODE_PROMPT, output_key="code")
TEST_CHAIN = LLMChain(llm=LLM, prompt=TEST_PROMPT, output_key="test")

SEQ = SequentialChain(
    chains=[CODE_CHAIN, TEST_CHAIN],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)


@app.command()
def main(task: str = typer.Option(), language: str = "python"):
    result = SEQ({"language": language, "task": task})
    print(result)


if __name__ == "__main__":
    app()
