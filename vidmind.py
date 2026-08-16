import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document

# Streamlit page setup
st.set_page_config(page_title="YouTube Q&A App", layout="centered")
st.title("🎥 YouTube Video Question Answering App")
st.write("Paste a **YouTube video URL**, and ask a question based on its transcript.")

# Input fields
video_url = st.text_input("🔗 Enter YouTube Video URL:")
user_question = st.text_input("❓ Ask your question:")

# Submit button
if st.button("🚀 Get Answer", use_container_width=True):
    if video_url and user_question:
        # Extract YouTube Video ID
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
        if match:
            video_id = match.group(1)

            try:
                # Get transcript
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)

                # Split transcript into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(transcript)
                documents = [Document(page_content=chunk) for chunk in chunks]

                # Vector store setup
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_store = FAISS.from_documents(documents, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                # Prompt setup
                prompt = PromptTemplate(
                    template="""
                    You are a helpful assistant.
                    Answer ONLY from the provided transcript context.
                    If the context is insufficient, just say you don't know.

                    {context}
                    Question: {question}
                    """,
                    input_variables=["context", "question"]
                )

                # LLM setup
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

                # Format context docs
                def format_docs(retrieved_docs):
                    return "\n\n".join(doc.page_content for doc in retrieved_docs)

                # Use RunnableParallel to pass both question and context
                parallel_chain = RunnableParallel({
                    "question": RunnablePassthrough(),
                    "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x)))
                })

                # Complete QA chain
                qa_chain = parallel_chain | prompt | llm | StrOutputParser()

                # Get answer
                with st.spinner("🔍 Fetching answer..."):
                    answer = qa_chain.invoke(user_question)

                st.success("✅ Answer:")
                st.write(answer)

            except TranscriptsDisabled:
                st.error("❌ No captions available for this video.")
            except Exception as e:
                st.error(f"🚨 Error: {str(e)}")
        else:
            st.warning("⚠️ Invalid YouTube URL. Please check and try again.")
    else:
        st.warning("⚠️ Please enter both the YouTube URL and your question.")
