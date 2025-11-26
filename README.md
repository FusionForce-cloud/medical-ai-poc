# 🩺 Post-Discharge Nephrology AI Assistant (RAG + Multi-Agent + Groq)

An interactive **AI-powered virtual nephrology assistant** designed to support patients **after hospital discharge**.  
The system uses **Multi-Agent AI (CrewAI)** + **RAG (PDF-based retrieval)** + **Groq LLM inference** to provide fast educational responses while ensuring patient safety.


## 🌟 **Features**
| Capability | Description |
|----------|-------------|
| 🏥 Patient Identification | Receptionist agent retrieves discharge report using stored records |
| 📚 RAG-based Medical Query Answering | Nephrology PDF used as primary medical knowledge source |
| 🌐 Web Search Fallback | External search used only when RAG cannot answer |
| 🤖 Multi-Agent Collaboration | Receptionist + Clinical Nephrologist agents work together |
| ⚡ Low latency responses | Powered by **Groq Llama-3.3 70B** |
| 💬 Chat UI | Built using **Streamlit** |

---

## 🔑 **Important Note: Limited Question Handling**
This AI responds **only when medical keywords are detected** (e.g., *pain, swelling, medication, dialysis, sodium, diet, junk food, fatigue, urine, etc.*).

If keywords are **not present**, the system assumes the conversation is administrative and interacts using the **Receptionist agent only**.

This ensures **safety + relevance** of medical responses.

---

## 🖥️ **User Interface**
To launch the chat interface locally:

```bash
streamlit run app_streamlit.py
