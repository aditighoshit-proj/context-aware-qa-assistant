import streamlit as st
from qa_engine import ContextQA

st.set_page_config(page_title="Context-Aware Q&A", page_icon="📚", layout="centered")
st.title("📚 Context-Aware Q&A Assistant")
st.caption("Answers are grounded only in the supplied knowledge base. If the source doesn't contain enough evidence, the assistant abstains.")

@st.cache_resource
def load_engine():
    return ContextQA.from_directory("knowledge_base")

engine = load_engine()
with st.sidebar:
    st.header("Knowledge base")
    st.write("Loaded source files:")
    for name in engine.source_names:
        st.write(f"• `{name}`")
    st.divider()
    # st.write("**Grounding rules**")
    # st.write("- Use retrieved source passages only.")
    # st.write("- Cite the source file and section.")
    # st.write("- Say when the answer is not available.")

question = st.text_input("Ask a question", placeholder="e.g. How many annual leave days can be carried forward?")
if st.button("Get answer", type="primary", disabled=not question.strip()):
    result = engine.answer(question)
    if result["answered"]:
        st.success(result["answer"])
    else:
        st.warning(result["answer"])
    if result["sources"]:
        st.subheader("Evidence used")
        for src in result["sources"]:
            with st.expander(f'{src["file"]} — {src["section"]} (score {src["score"]:.2f})'):
                st.write(src["text"])
    else:
        st.caption("No sufficiently relevant source passage was found.")

with st.expander("Try an unanswerable question"):
    st.write("Example: What is the company's CEO's favourite colour?")
    st.caption("This fact is intentionally absent from the sample knowledge base.")
