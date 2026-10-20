
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from resume_core import evaluate_pdf_resume

st.set_page_config(page_title="Resume Screening Assistant")

st.title("Resume Screening Assistant")
st.write("Compare PDF resumes with a job description.")

api_key = st.sidebar.text_input(
    "OpenAI API key",
    type="password"
)

st.caption(
    "Use fictional resumes or documents you have permission to process. "
    "Resume text and the job description are sent to OpenAI."
)

with st.form("resume_form"):
    uploaded_resumes = st.file_uploader(
        "Upload PDF resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

    job_description = st.text_area(
        "Enter the job description",
        height=220
    )

    consent = st.checkbox(
        "I have permission to process these documents using OpenAI."
    )

    submitted = st.form_submit_button("Evaluate resumes")

if submitted:
    if not api_key.strip():
        st.warning("Enter your OpenAI API key.")
    elif not uploaded_resumes:
        st.warning("Upload at least one PDF resume.")
    elif not job_description.strip():
        st.warning("Enter a job description.")
    elif not consent:
        st.warning("Confirm permission to process the documents.")
    else:
        st.session_state["resume_results"] = []
        st.session_state["resume_errors"] = []

        with tempfile.TemporaryDirectory() as temp_directory:
            for number, uploaded in enumerate(uploaded_resumes):
                filename = Path(uploaded.name).name
                pdf_path = Path(temp_directory) / f"resume_{number}.pdf"
                pdf_path.write_bytes(uploaded.getvalue())

                try:
                    with st.spinner(f"Evaluating {filename}..."):
                        review, summary = evaluate_pdf_resume(
                            pdf_path,
                            filename,
                            job_description,
                            api_key.strip()
                        )

                    st.session_state["resume_results"].append({
                        "Resume": filename,
                        "Review": review.model_dump(),
                        "Summary": summary
                    })

                except Exception:
                    st.session_state["resume_errors"].append(filename)

for filename in st.session_state.get("resume_errors", []):
    st.error(
        f"Could not evaluate {filename}. Check that the PDF contains "
        "readable text, and check your API key, balance and connection."
    )

results = st.session_state.get("resume_results", [])

if results:
    st.subheader("Comparison")

    st.dataframe(pd.DataFrame([
        {
            "Resume": item["Resume"],
            "Match score": item["Summary"]["Match score"],
            "Matching skills": len(item["Summary"]["Matching skills"]),
            "Partial matches": len(item["Summary"]["Partial matches"]),
            "Skills not found": len(item["Summary"]["Skills not found"]),
            "Recommendation": item["Summary"]["Recommendation"]
        }
        for item in results
    ]), hide_index=True)

    st.caption(
        "Score: evidence found = 1 point, partial evidence = 0.5, "
        "not found = 0. Requirements have equal weight. "
        "The score is not a hiring probability."
    )

    for item in results:
        with st.expander(item["Resume"], expanded=True):
            summary = item["Summary"]

            st.write("**Candidate summary:**", summary["Candidate summary"])
            st.write("**Strengths:**", summary["Strengths"])
            st.write("**Weaknesses or gaps:**", summary["Weaknesses or gaps"])

            st.dataframe(
                pd.DataFrame(item["Review"]["requirement_checks"]),
                hide_index=True
            )

            st.write("**Recommendation:**", summary["Recommendation"])
            st.write("**Reason:**", summary["Reason"])

    st.download_button(
        "Download evaluations",
        data=json.dumps(results, indent=2, ensure_ascii=False),
        file_name="resume_evaluations.json",
        mime="application/json"
    )

st.caption(
    "AI evaluations can be incorrect. Verify the quotes and findings "
    "against the original resumes. Missing evidence does not prove "
    "a candidate lacks a skill. A recruiter makes the final decision."
)
