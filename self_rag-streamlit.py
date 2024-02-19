import os



import pprint
os.environ['OPENAI_API_KEY']=st.secrets["OPENAI_API_KEY"]
# Get the value of the OPENAI_API_KEY environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Check if the environment variable is set
if api_key is not None:
    # print("OPENAI_API_KEY:", api_key)
    pprint.pprint("OPENAI_API_KEY setted")
else:
    pprint.pprint("OPENAI_API_KEY is not set.")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import pdfplumber
from dataclasses import dataclass
from typing import Dict, TypedDict

from langchain_core.messages import BaseMessage

import json
import operator
from typing import Annotated, Sequence, TypedDict
import streamlit as st
from langchain import hub
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
st.set_page_config(
    page_title="LANGGRAPH",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def upload_file():
    uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type="pdf")
    if uploaded_file is not None:
        st.sidebar.write("Uploaded file details:")
        st.sidebar.write({"Filename": uploaded_file.name, "File size": uploaded_file.size})
    return uploaded_file
    
@dataclass
class Document:
    page_content: str
    metadata: dict

def main():
    st.header(":rainbow[LangGraph]",divider='rainbow')
    

    try:
        
        file_path=upload_file()
        if file_path is not None:
            st.write("Chunking the document(s)")
            
    # Load the content of the PDF file using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=150, chunk_overlap=10
            )
            doc_splits = text_splitter.split_text(pdf_text)

            # Create a list of Document instances where each instance represents a document chunk
            documents = [Document(page_content=chunk, metadata={'source': f'{file_path}.pdf'}) for chunk in doc_splits]

            # Add to vectorDB
            vectorstore = Chroma.from_documents(
                documents=documents,
                collection_name="rag-chroma",
                embedding=OpenAIEmbeddings(),
            )
            retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6,"k":15})
            pprint.pprint("Stored in vector-DB")
            #                                               -------------Graph------------
            pprint.pprint("defining a Graph")

            st.sidebar.write("Stored in vector-DB")

            # Graph
            st.sidebar.write("Defining a Graph")
            class GraphState(TypedDict):
                """
                Represents the state of our graph.

                Attributes:
                    keys: A dictionary where each key is a string.
                """

                keys: Dict[str, any]

            pprint.pprint("defined Class for graphState")
            question = st.text_area("Enter your question here:")

                
            if st.button("Run Workflow"):
                inputs = {"keys": {"question": question}}
                
                def retrieve(state):
                    """
                    Retrieve documents

                    Args:
                        state (dict): The current graph state

                    Returns:
                        state (dict): New key added to state, documents, that contains retrieved documents
                    """
                    print("---RETRIEVE---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    documents = retriever.get_relevant_documents(question)
                    return {"keys": {"documents": documents, "question": question}}

                def generate(state):
                    """
                    Generate answer

                    Args:
                        state (dict): The current graph state

                    Returns:
                        state (dict): New key added to state, generation, that contains LLM generation
                    """
                    print("---GENERATE---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    documents = state_dict["documents"]
                    print("---------------------------------------------------------------------     documents ----------------------------------------------------")
                    print("documents", documents)
                    print("---------------------------------------------------------------------     documents ----------------------------------------------------")
                    prompt = hub.pull("rlm/rag-prompt")
                    # basic_prompt = prompt.messages[0].prompt.template
                    # print(f"prompt---end {prompt}")
                    # prompt.messages[0].prompt.template="As a senior proposal writer, I need your expertise to create an outline for a response to an RFP. Based on the context given, generate a well-organized outline. Concentrate on the technical, management and business aspects outlined in the RFP. Ensure the outline follows the structure detailed in the RFP. Details such as legal disclaimers, place of performance, introductory sections, summary sections, conclusion sections, terms and conditions etc. are not necessary for the creation of outline. Omit them and focus on the technical, management and business components needed for an outline"
                    
                    # prompt.messages[0].prompt.template="Act as an expert RFP reader, extract the details where the instructions are given like what and how the response will be written in what format,how many sections what section,subsections, factors etc.So that user keep those instructions in consideration and start writting response for it.Dont give generic sections,pay more attention towards the given documents."


                    prompt.messages[0].prompt.template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Give the details as detaileda s possible.\nQuestion: {question} \nContext: {context} \nAnswer:"
                    print()
                    print()
                    print()
                    print()
                    print(f"prompt---Start {prompt}")
                    print()
                    print()
                    pprint.pprint("--------- end of prompt text---------")
                    # LLM
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.3)
                    print("llm  --- {llm}")
                    # Post-processing
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    # Chain
                    rag_chain = prompt | llm | StrOutputParser()

                    # Run
                    generation = rag_chain.invoke({"context": documents, "question": question})
                    return {
                        "keys": {"documents": documents, "question": question, "generation": generation}
                    }
                def grade_documents(state):
                    """
                    Determines whether the retrieved documents are relevant to the question.

                    Args:
                        state (dict): The current graph state

                    Returns:
                        state (dict): Updates documents key with relevant documents
                    """

                    print("---CHECK RELEVANCE---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    documents = state_dict["documents"]

                    # Data model
                    class grade(BaseModel):
                        """Binary score for relevance check."""

                        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

                    # LLM
                    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", streaming=True)

                    # Tool
                    grade_tool_oai = convert_to_openai_tool(grade)

                    # LLM with tool and enforce invocation
                    llm_with_tool = model.bind(
                        tools=[convert_to_openai_tool(grade_tool_oai)],
                        tool_choice={"type": "function", "function": {"name": "grade"}},
                    )

                    # Parser
                    parser_tool = PydanticToolsParser(tools=[grade])

                    # Prompt
                    prompt = PromptTemplate(
                        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
                        Here is the retrieved document: \n\n {context} \n\n
                        Here is the user question: {question} \n
                        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
                        input_variables=["context", "question"],
                    )

                    # Chain
                    chain = prompt | llm_with_tool | parser_tool

                    # Score
                    filtered_docs = []
                    for d in documents:
                        score = chain.invoke({"question": question, "context": d.page_content})
                        grade = score[0].binary_score
                        if grade == "yes":
                            print("---GRADE: DOCUMENT RELEVANT---")
                            filtered_docs.append(d)
                        else:
                            print("---GRADE: DOCUMENT NOT RELEVANT---")
                            continue

                    return {"keys": {"documents": filtered_docs, "question": question}}
                def transform_query(state):
                    """
                    Transform the query to produce a better question.

                    Args:
                        state (dict): The current graph state

                    Returns:
                        state (dict): Updates question key with a re-phrased question
                    """

                    print("---TRANSFORM QUERY---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    documents = state_dict["documents"]

                    # Create a prompt template with format instructions and the query
                    prompt = PromptTemplate(
                        template="""You are generating questions that is well optimized for retrieval. \n 
                        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
                        Here is the initial question:
                        \n ------- \n
                        {question} 
                        \n ------- \n
                        Formulate an improved question: """,
                        input_variables=["question"],
                    )
                    print(" -- Query Transformed -- ")
                    print(prompt)
                    print()
                    print(" -- Query Transformed -- ")
                    # Grader
                    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", streaming=True)

                    # Prompt
                    chain = prompt | model | StrOutputParser()
                    better_question = chain.invoke({"question": question})
                    print(f"Here is the improved question:- {better_question}")
                    return {"keys": {"documents": documents, "question": better_question}}
                def prepare_for_final_grade(state):
                    """
                    Passthrough state for final grade.

                    Args:
                        state (dict): The current graph state

                    Returns:
                        state (dict): The current graph state
                    """

                    print("---FINAL GRADE---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    documents = state_dict["documents"]
                    generation = state_dict["generation"]

                    return {
                        "keys": {"documents": documents, "question": question, "generation": generation}
                    }
                ### Edges ###
                def decide_to_generate(state):
                    """
                    Determines whether to generate an answer, or re-generate a question.

                    Args:
                        state (dict): The current state of the agent, including all keys.

                    Returns:
                        str: Next node to call
                    """

                    print("---DECIDE TO GENERATE---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    filtered_documents = state_dict["documents"]

                    if not filtered_documents:
                        # All documents have been filtered check_relevance
                        # We will re-generate a new query
                        print("---DECISION: TRANSFORM QUERY---")
                        return "transform_query"
                    else:
                        # We have relevant documents, so generate answer
                        print("---DECISION: GENERATE---")
                        return "generate"
                def grade_generation_v_documents(state):
                    """
                    Determines whether the generation is grounded in the document.

                    Args:
                        state (dict): The current state of the agent, including all keys.

                    Returns:
                        str: Binary decision
                    """

                    print("---GRADE GENERATION vs DOCUMENTS---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    documents = state_dict["documents"]
                    generation = state_dict["generation"]

                    # Data model
                    class grade(BaseModel):
                        """Binary score for relevance check."""

                        binary_score: str = Field(description="Supported score 'yes' or 'no'")

                    # LLM
                    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", streaming=True)

                    # Tool
                    grade_tool_oai = convert_to_openai_tool(grade)

                    # LLM with tool and enforce invocation
                    llm_with_tool = model.bind(
                        tools=[convert_to_openai_tool(grade_tool_oai)],
                        tool_choice={"type": "function", "function": {"name": "grade"}},
                    )

                    # Parser
                    parser_tool = PydanticToolsParser(tools=[grade])

                    # Prompt
                    prompt = PromptTemplate(
                        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
                        Here are the facts:
                        \n ------- \n
                        {documents} 
                        \n ------- \n
                        Here is the answer: {generation}
                        Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts.""",
                        input_variables=["generation", "documents"],
                    )

                    # Chain
                    chain = prompt | llm_with_tool | parser_tool

                    score = chain.invoke({"generation": generation, "documents": documents})
                    grade = score[0].binary_score

                    if grade == "yes":
                        print("---DECISION: SUPPORTED, MOVE TO FINAL GRADE---")
                        return "supported"
                    else:
                        print("---DECISION: NOT SUPPORTED, GENERATE AGAIN---")
                        return "not supported"

                def grade_generation_v_question(state):

                    """
                    Determines whether the generation addresses the question.

                    Args:
                        state (dict): The current state of the agent, including all keys.

                    Returns:
                        str: Binary decision
                    """

                    print("---GRADE GENERATION vs QUESTION---")
                    state_dict = state["keys"]
                    question = state_dict["question"]
                    documents = state_dict["documents"]
                    generation = state_dict["generation"]

                    # Data model
                    class grade(BaseModel):
                        """Binary score for relevance check."""

                        binary_score: str = Field(description="Useful score 'yes' or 'no'")

                    # LLM
                    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", streaming=True)

                    # Tool
                    grade_tool_oai = convert_to_openai_tool(grade)

                    # LLM with tool and enforce invocation
                    llm_with_tool = model.bind(
                        tools=[convert_to_openai_tool(grade_tool_oai)],
                        tool_choice={"type": "function", "function": {"name": "grade"}},
                    )

                    # Parser
                    parser_tool = PydanticToolsParser(tools=[grade])

                    # Prompt
                    prompt = PromptTemplate(
                        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
                        Here is the answer:
                        \n ------- \n
                        {generation} 
                        \n ------- \n
                        Here is the question: {question}
                        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.""",
                        input_variables=["generation", "question"],
                    )

                    # Prompt
                    chain = prompt | llm_with_tool | parser_tool

                    score = chain.invoke({"generation": generation, "question": question})
                    grade = score[0].binary_score

                    if grade == "yes":
                        print("---DECISION: USEFUL---")
                        return "useful"
                    else:
                        print("---DECISION: NOT USEFUL---")
                        return "not useful"

                # inputs = {"keys": {f"{question}": """Instructions given for offeror to following for writting response for RFP.and how will be the  the structure of response."""}}
                
                workflow = StateGraph(GraphState)

                # Define the nodes
                workflow.add_node("retrieve", retrieve)  # retrieve
                workflow.add_node("grade_documents", grade_documents)  # grade documents
                workflow.add_node("generate", generate)  # generatae
                workflow.add_node("transform_query", transform_query)  # transform_query
                workflow.add_node("prepare_for_final_grade", prepare_for_final_grade)  # passthrough

                # Build graph
                workflow.set_entry_point("retrieve")
                workflow.add_edge("retrieve", "grade_documents")
                workflow.add_conditional_edges(
                    "grade_documents",
                    decide_to_generate,
                    {
                        "transform_query": "transform_query",
                        "generate": "generate",
                    },
                )
                workflow.add_edge("transform_query", "retrieve")
                workflow.add_conditional_edges(
                    "generate",
                    grade_generation_v_documents,
                    {
                        "supported": "prepare_for_final_grade",
                        "not supported": "generate",
                    },
                )
                workflow.add_conditional_edges(
                    "prepare_for_final_grade",
                    grade_generation_v_question,
                    {
                        "useful": END,
                        "not useful": "transform_query",
                    },
                )

                # Compile
                app = workflow.compile()
                for output in app.stream(inputs):
                    for key, value in output.items():
                        # Node
                        pprint.pprint(f"Node '{key}':")
                        st.write(f"Node '{key}':")
                        # Optional: print full state at each node
                        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                    pprint.pprint("\n---\n")

                # Final generation
                st.markdown(value['keys']['generation'])
                pprint.pprint(value['keys']['generation'])
    except Exception as e:
        st.write("Upload the RFP")
        print(f"An error occurred: {str(e)}")
if __name__ == "__main__":
    main()
                    # -----------------------------------------RUN---------------------------------------------------
