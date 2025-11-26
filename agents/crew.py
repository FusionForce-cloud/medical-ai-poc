from crewai import LLM
llm = LLM(model="groq/llama-3.1-70b-versatile")
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from tools.patient_tool import get_patient_report
from tools.web_search_tool import web_search

load_dotenv()

# --- LLM: Groq through CrewAI's LLM wrapper ---
# Make sure GROQ_API_KEY is set in your .env
llm = LLM(model="groq/llama-3.3-70b-versatile")

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
    backstory="You are a friendly hospital receptionist assisting patients after discharge.",
    tools=[patient_report_tool],
    llm=llm,
    verbose=True,
)

clinical = Agent(
    role="Clinical Nephrology Agent",
    goal="Provide nephrology guidance using RAG and web search fallback.",
    backstory="You are an expert AI nephrologist helping discharged patients understand their condition.",
    tools=[web_search_tool],
    llm=llm,
    verbose=True,
)

# --- Main routing function used by Streamlit ---

def run_multi_agent_turn(user_input: str, session_state):
    """
    Route user message either to Receptionist or Clinical agent
    and handle persistent patient name recognition.
    """
    # 1️⃣ Detect and store patient name (only once)
    if session_state.patient_name is None:
        parts = user_input.strip().split()
        if len(parts) >= 2 and all(x.isalpha() for x in parts):
            session_state.patient_name = user_input.strip()

    # 2️⃣ If patient name is STILL unknown → receptionist must ask for name
    if session_state.patient_name is None:
        task = Task(
            description=(
                f"User said: {user_input}. You do NOT know their name yet.\n"
                "Ask politely for the patient's full name. Do NOT answer medical questions until name is known."
            ),
            agent=receptionist,
            expected_output="Ask for patient's name only."
        )
        crew = Crew(agents=[receptionist], tasks=[task], llm=llm, verbose=True)
        return crew.kickoff()

    # 3️⃣ If the name is known → route medical vs. non-medical messages
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
        # → route to Clinical Nephrology Agent
        task = Task(
            description=(
                f"Message from patient {session_state.patient_name}: {user_input}\n"
                "First use the RAG nephrology knowledge base (retriever). "
                "If insufficient, use WebSearchTool. Explain clearly and add [PDF Source] if PDF was used."
            ),
            agent=clinical,
            expected_output="Clear nephrology advice with citations."
        )
    else:
        # → route to Receptionist
        task = Task(
            description=(
                f"Message from patient {session_state.patient_name}: {user_input}\n"
                "Retrieve discharge summary and respond warmly with helpful follow-up questions."
            ),
            agent=receptionist,
            expected_output="Friendly, supportive response."
        )

    # 4️⃣ Run the selected agent
    crew = Crew(
        agents=[receptionist, clinical],
        tasks=[task],
        llm=llm,
        verbose=True
    )
    return crew.kickoff()
