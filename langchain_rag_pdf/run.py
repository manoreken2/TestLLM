# setup
# conda create -n LCrag -y python=3.11
# conda activate LCrag
# pip install transformers langchain faiss-cpu peft sentence-transformers unstructured pdfminer.six langchain_huggingface langchain-community langchain-core

# run
# conda activate LCrag
# python run.py


# https://qwen3lm.com/qwen3-langchain-rag/#install


def run(args):
    # 2
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen1.5-14B", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-14B", trust_remote_code=True, device_map="auto"
    )

    # 3
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # 4
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(args.pdf)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # 5
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    retriever = vectorstore.as_retriever()

    # 6
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import pipeline

    qwen_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512
    )

    llm = HuggingFacePipeline(pipeline=qwen_pipe)

    # https://iifx.dev/en/articles/460342435/mastering-the-new-retrieval-chain-troubleshooting-and-best-practices

    from langchain_core.prompts import ChatPromptTemplate

    # 2. Define your Prompt
    system_prompt = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

    # 3. Create the "Stuff" Documents Chain
    # This replaces the internal 'combine_documents' logic of RetrievalQA
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 4. Create the final Retrieval Chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. Invoke it!
    response = rag_chain.invoke({"input": args.q})

    print("Answer is :")
    print(response["answer"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-pdf", type=str, default="saiyuuki.pdf", help="pdf file input")
    parser.add_argument(
        "-q",
        type=str,
        default="How 悟空 flies? Does he use some equipment to fly?",
        help="query string to ask",
    )

    args = parser.parse_args()

    print(f"input pdf: {args.pdf}")
    print(f'query string: "{args.q}"')

    run(args)
