import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from tools.patient_tool import get_patient_report
from tools.web_search_tool import web_search

load_dotenv()

# --- LLM: Groq through CrewAI's LiteLLM wrapper ---
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2048,
)

# --- Tools ---
@tool("PatientReportTool")
def patient_report_tool(patient_name: str):
    """Retrieve patient discharge report by name."""
    return get_patient_report(patient_name)

@tool("WebSearchTool")
def web_search_tool(query: str):
    """Search web when RAG cannot fully answer."""
    return web_search(query)

# --- RAG retriever over nephrology PDF ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever()

# --- Agents ---
receptionist = Agent(
    role="Receptionist Agent",
    goal="Identify the patient and retrieve discharge summary.",
    backstory="You are a friendly hospital receptionist assisting discharged nephrology patients.",
    tools=[patient_report_tool],
    llm=llm,
    verbose=True,
)

clinical = Agent(
    role="Clinical Nephrology Agent",
    goal="Provide nephrology guidance using RAG first, with web search fallback.",
    backstory="You are an AI nephrologist helping patients understand their kidney recovery.",
    tools=[web_search_tool],
    llm=llm,
    verbose=True,
)

# --- Main routing logic for Streamlit ---
def run_multi_agent_turn(user_input: str, session_state):
    """Routes user message to the correct agent and tracks patient identity."""

    # 1️⃣ Store patient name once
    if session_state.patient_name is None:
        parts = user_input.strip().split()
        if len(parts) >= 2 and all(x.isalpha() for x in parts):   # basic name check
            session_state.patient_name = user_input.strip()

    # 2️⃣ If name still unknown → receptionist must ask for it
    if session_state.patient_name is None:
        task = Task(
            description=(
                f"User said: {user_input}. You do NOT know their name yet.\n"
                "Ask for their full name and do NOT answer medical questions until the name is known."
            ),
            agent=receptionist,
            expected_output="Ask politely for patient's full name only."
        )
        crew = Crew(agents=[receptionist], tasks=[task], llm=llm, verbose=True)
        return crew.kickoff()

    # 3️⃣ Determine if the message is medical or general
    text = user_input.lower()
    medical_keywords = [
        "pain", "swelling", "medication", "tablet", "dose",
        "urine", "dialysis", "breath", "blood pressure",
        "potassium", "sodium", "protein", "phosphorus",
        "diet", "junk food", "food", "nutrition", "water intake",
        "fluid", "restriction", "exercise", "symptom", "condition",
        "kidney", "nephrology", "fatigue", "nausea", "vomiting",
        "cramps", "itching", "treatment", "follow-up", "appointment"
    ]

    if any(k in text for k in medical_keywords):
        # 👨‍⚕️ Clinical agent
        task = Task(
            description=(
                f"Message from patient {session_state.patient_name}: {user_input}\n"
                "Use nephrology PDF (RAG) first via retriever. If insufficient, use WebSearchTool.\n"
                "Add [PDF Source] when PDF was used."
            ),
            agent=clinical,
            expected_output="Clear safe nephrology advice with citations."
        )
    else:
        # 👩‍💼 Receptionist
        task = Task(
            description=(
                f"Message from patient {session_state.patient_name}: {user_input}\n"
                "Retrieve discharge summary and reply warmly with follow-up questions."
            ),
            agent=receptionist,
            expected_output="Friendly, helpful response."
        )

    # 4️⃣ Execute task
    crew = Crew(
        agents=[receptionist, clinical],
        tasks=[task],
        llm=llm,
        verbose=True
    )
    return crew.kickoff()
