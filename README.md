# LangGraph

LangGraph is a tool designed to assist in question-answering tasks using language models and document retrieval techniques.

## Overview

LangGraph utilizes various language models and vector stores to retrieve relevant documents based on user queries. It then generates answers to these queries using state-of-the-art language models, providing a seamless question-answering experience.

## Getting Started

To get started with LangGraph, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using the following command:
git clone https://github.com/your_username/LangGraph.git

2. **Install Dependencies**: Navigate to the project directory and install the required dependencies by running:
pip install -r requirements.txt

3. **Set Up Environment Variables**: Make sure to set up the necessary environment variables, particularly `OPENAI_API_KEY`, which is required for language model access.

4. **Run the Application**: Execute the main script to run the LangGraph application:
 streamlit run langgraph.py

5. **Upload PDF Files**: Upload PDF files containing documents relevant to your queries using the provided interface.

6. **Ask Questions**: Enter your questions in the text area and click on the "Run Workflow" button to retrieve answers.

## Contributing

Contributions to LangGraph are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Description of changes'`.
4. Push to your branch: `git push origin feature-name`.
5. Submit a pull request detailing your changes.

Please ensure that your contributions adhere to the project's coding standards and include appropriate documentation where necessary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
