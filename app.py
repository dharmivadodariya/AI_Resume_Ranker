import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import re
import plotly.graph_objects as go

load_dotenv()

# Initialize AzureChatOpenAI
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.3,
    max_tokens=500,
)

st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("🧠 Resume Matcher: Rank Resumes Against Job Description")
st.write(
    "This app uses an AI model to rank resumes based on their match with a given job description (JD). "
    "Upload a JD and multiple resumes in PDF format to get started."
)

# Skill keyword list
SKILL_KEYWORDS = ["Python", "SQL", "TensorFlow", "PyTorch", "NLP", "Machine Learning", "Data Analysis"]

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)

# Function to extract skills from text
def extract_skills(text):
    return [skill for skill in SKILL_KEYWORDS if re.search(rf"\\b{skill}\\b", text, re.IGNORECASE)]

# Function to plot skill match radar chart
def plot_skill_match(resume_skills, jd_skills):
    all_skills = list(set(jd_skills + resume_skills))
    resume_vec = [1 if s in resume_skills else 0 for s in all_skills]
    jd_vec = [1 if s in jd_skills else 0 for s in all_skills]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=resume_vec, theta=all_skills, name='Resume'))
    fig.add_trace(go.Scatterpolar(r=jd_vec, theta=all_skills, name='JD'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    return fig

# Upload JD
st.subheader("📄 Upload Job Description (JD)")
jd_file = st.file_uploader("Upload Job Description PDF", type="pdf", key="jd")

# Upload multiple resumes
st.subheader("📂 Upload Resumes (Max: 10 PDFs)")
resume_files = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True, key="resumes")

if jd_file and resume_files:
    with st.spinner("Extracting and processing..."):
        jd_text = extract_text_from_pdf(jd_file)
        jd_skills = extract_skills(jd_text)

        results = []

        for resume_file in resume_files:
            resume_text = extract_text_from_pdf(resume_file)
            resume_skills = extract_skills(resume_text)

            prompt = ChatPromptTemplate.from_template(
                """
                You are a recruiter. Based on the following job description (JD) and candidate resume,
                rate the resume's match with the JD on a scale from 0 to 100, and justify the score.

                JD:
                {jd}

                Resume:
                {resume}

                Return output in the format:
                Score: <score>/100
                Justification: <one short paragraph>
                """
            )

            formatted_prompt = prompt.format_messages(jd=jd_text, resume=resume_text)
            response = llm(formatted_prompt)

            score_line = response.content.split("\n")[0]
            try:
                score = int(score_line.split(":")[1].strip().replace("/100", ""))
            except:
                score = 0

            results.append({
                "name": resume_file.name,
                "score": score,
                "response": response.content,
                "skills": resume_skills,
                "text": resume_text
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        st.success("✅ Resumes processed and ranked!")

        # Show results
        st.subheader("🏆 Ranked Resumes")
        for idx, res in enumerate(results, 1):
            st.markdown(f"### {idx}. {res['name']} — 🟢 Score: {res['score']}/100")
            st.markdown(f"**Extracted Skills**: {', '.join(res['skills']) if res['skills'] else 'No skills detected'}")

            fig = plot_skill_match(res["skills"], jd_skills)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📝 View Justification & Feedback"):
                st.markdown(res["response"])

                if st.button(f"💡 Show GPT Resume Suggestions for {res['name']}", key=res["name"]):
                    feedback_prompt = f"""
                    You are an AI resume coach. Review the following resume and suggest improvements
                    for clarity, impact, and better alignment with this JD:

                    JD:
                    {jd_text}

                    Resume:
                    {res['text']}

                    Give feedback in bullet points.
                    """
                    feedback_response = llm([HumanMessage(content=feedback_prompt)])
                    st.markdown("#### ✍️ GPT Feedback")
                    st.markdown(feedback_response.content)
