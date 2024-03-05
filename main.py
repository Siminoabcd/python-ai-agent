from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_eng import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import f_revolution_engine

load_dotenv()

#imbd-tool
imbd_path = os.path.join("data", "imbd.csv")
imbd_df = pd.read_csv(imbd_path)

imbd_query_engine = PandasQueryEngine(
    df=imbd_df, verbose=True, instruction_str=instruction_str
)
imbd_query_engine.update_prompts({"pandas_prompt": new_prompt})

#population-tool
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(query_engine=imbd_query_engine, 
        metadata=ToolMetadata(
            name="imbd_data",
            description="this gives information at the movies and movie statistics",
        ),
    ),
    QueryEngineTool(query_engine=f_revolution_engine, 
        metadata=ToolMetadata(
            name="french_revolution_data",
            description="this is a pdf containing knowledge about French Revolution for our agent to be able answer the queries"
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)