import streamlit as st
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun

# Initialize Llama model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/path/to/llama/model.bin",
    callback_manager=callback_manager,
    verbose=True,
)

# Initialize DuckDuckGo search tool
search = DuckDuckGoSearchRun()

def explain_code(code):
    prompt = PromptTemplate(
        input_variables=["code"],
        template="Explain the following code:\n\n{code}\n\nExplanation:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(code)

def generate_readme(code):
    prompt = PromptTemplate(
        input_variables=["code"],
        template="Generate a README.md file for the following code:\n\n{code}\n\nREADME.md:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(code)

def generate_project_report(code):
    prompt = PromptTemplate(
        input_variables=["code"],
        template="Generate a project report for the following code:\n\n{code}\n\nProject Report:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(code)

def generate_code(prompt):
    code_prompt = PromptTemplate(
        input_variables=["prompt"],
        template="Generate Python code for the following prompt:\n\n{prompt}\n\nCode:"
    )
    chain = LLMChain(llm=llm, prompt=code_prompt)
    return chain.run(prompt)

def main():
    st.title("Coding Assistant (Llama 2)")

    option = st.selectbox(
        "Choose a function",
        ("Explain Code", "Generate README", "Generate Project Report", "Generate Code", "Search for Coding Resources")
    )

    if option in ["Explain Code", "Generate README", "Generate Project Report"]:
        code = st.text_area("Enter your code here:")
        if st.button("Submit"):
            if option == "Explain Code":
                result = explain_code(code)
            elif option == "Generate README":
                result = generate_readme(code)
            else:
                result = generate_project_report(code)
            st.write(result)

    elif option == "Generate Code":
        prompt = st.text_input("Enter your code generation prompt:")
        if st.button("Generate"):
            result = generate_code(prompt)
            st.code(result)

    else:  # Search for Coding Resources
        query = st.text_input("Enter your coding question or topic:")
        if st.button("Search"):
            results = search.run(query + " programming")
            st.write(results)

if __name__ == "__main__":
    main()