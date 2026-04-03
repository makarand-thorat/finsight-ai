import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "FinSight AI",
    page_icon = "📊",
    layout ="wide"
)

st.title("Finsight AI")
st.caption("Production RAG evaluation platform for financial documents")

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Model",
        ["gemini"],
        index=0
    )
    run_evaluation = st.toggle("Run RAGAS evaluation", value=True)
    st.divider()
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a financial PDF",
        type=["pdf"]
    )
    if uploaded_file:
        if st.button("Ingest Document", type="primary"):
            with st.spinner("Ingesting PDF..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Ingested {data['chunks_created']} chunks")
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Could not connect to API: {str(e)}")

col1, col2 = st.columns([3, 2])

with col1:
    st.header("💬 Ask a Question")
    question = st.text_area(
        "Enter your question about the document",
        placeholder="What was the total revenue in 2024?",
        height=100
    )

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "model_name": model_name,
                        "evaluate": run_evaluation
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    st.subheader("Answer")
                    st.write(data["answer"])

                    if data["sources"]:
                        with st.expander("View sources"):
                            for i, source in enumerate(data["sources"]):
                                st.markdown(f"**Source {i+1} — Page {source['page']}**")
                                st.caption(source["content_preview"])
                                st.divider()

                    if run_evaluation and data["scores"]:
                        st.subheader("RAGAS Evaluation Scores")
                        scores = data["scores"]
                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            "Faithfulness",
                            f"{scores.get('faithfulness', 0):.3f}",
                            help="Is the answer grounded in the document?"
                        )
                        m2.metric(
                            "Answer Relevancy",
                            f"{scores.get('answer_relevancy', 0):.3f}",
                            help="Does the answer address the question?"
                        )
                        
                        m3.metric(
                            "Average",
                            f"{scores.get('average', 0):.3f}",
                            help="Average across all the metrics"
                        )

                else:
                    st.error(f"Error: {response.json()['detail']}")

            except Exception as e:
                st.error(f"Could not connect to API: {str(e)}")

with col2:
    st.header("📈 Evaluation Dashboard")

    try:
        scores_response = requests.get(f"{API_URL}/scores")
        results_response = requests.get(f"{API_URL}/results")

        if scores_response.status_code == 200:
            avg_scores = scores_response.json()["average_scores"]

            if avg_scores:
                st.subheader("Average RAGAS Scores")
                scores_df = pd.DataFrame([{
                    "Metric": "Faithfulness",
                    "Score": avg_scores.get("faithfulness", 0)
                }, {
                    "Metric": "Answer Relevancy",
                    "Score": avg_scores.get("answer_relevancy", 0)
                }])
                st.dataframe(
                    scores_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No evaluations yet — ask a question to see scores")

        if results_response.status_code == 200:
            results_data = results_response.json()
            total = results_data["total_evaluations"]
            st.metric("Total Evaluations", total)

            if results_data["results"]:
                st.subheader("Recent Evaluations")
                recent = results_data["results"][-5:][::-1]
                for r in recent:
                    with st.expander(f"Q: {r['question'][:60]}..."):
                        st.caption(f"Model: {r['model_used']} · {r['timestamp']}")
                        st.write(r["answer"])
                        cols = st.columns(2)
                        cols[0].metric("Faithfulness", r["scores"].get("faithfulness", 0))
                        cols[1].metric("Relevancy", r["scores"].get("answer_relevancy", 0))
                        

    except Exception as e:
        st.warning(f"Dashboard unavailable — make sure the API is running: {str(e)}")
