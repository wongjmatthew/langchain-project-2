import os
import requests
import csv
import time
from datetime import datetime
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
import docx

# ==========================================
# 1. AUTHENTICATION
# ==========================================
google_api_key = os.environ.get("GOOGLE_API_KEY")
serpapi_key = os.environ.get("SERPAPI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

# ==========================================
# 2. THE TOOLS
# ==========================================
def search_google_jobs(query: str) -> dict:
    if not query or query.lower() == "none":
        return {"readable": "I need a job title and location to search for.", "top_job_context": None}
        
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_jobs",
        "q": query,
        "hl": "en",
        "api_key": serpapi_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("jobs_results", [])
        
        if not results:
            return {"readable": f"No jobs found for '{query}' on Google Jobs right now.", "top_job_context": None}
            
        output = f"### 🔍 Top Google Jobs found for '{query}':\n\n"
        for job in results[:3]:
            title = job.get('title', 'Unknown Title')
            company = job.get('company_name', 'Unknown Company')
            link = job.get('share_link', '#')
            
            full_desc = job.get('description', 'No description provided.')
            short_desc = full_desc[:200]
            if len(full_desc) > 200:
                short_desc = short_desc.rsplit(' ', 1)[0] + "..."
                
            output += f"- **{title}** at {company}\n"
            output += f"  > *\"{short_desc}\"*\n"
            output += f"  🔗 **[Click here to view job]({link})**\n\n"
            
        best_job = results[0]
        top_job_context = f"Job Title: {best_job.get('title')}\nCompany: {best_job.get('company_name')}\nFull Description: {best_job.get('description', 'No description provided.')}"
            
        return {"readable": output, "top_job_context": top_job_context}
    except Exception as e:
        return {"readable": f"Error connecting to SerpApi: {e}", "top_job_context": None}

def read_resume(file_path) -> str:
    if not file_path:
        return "No resume uploaded."
    try:
        file_ext = file_path.lower()
        text = ""
        if file_ext.endswith('.pdf'):
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        elif file_ext.endswith('.docx'):
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_ext.endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return "Unsupported file type."
        return text.strip()
    except Exception as e:
        return f"Error reading resume: {e}"

tailor_prompt = ChatPromptTemplate.from_template("""
You are an expert ATS Resume Optimizer.
Compare the user's RESUME to the REAL JOB DESCRIPTION. 
Suggest 3 specific bullet point changes to help the user match the job description keywords.
DO NOT invent new experience. Ground your edits ONLY in the provided job description context.
RESUME:
{resume_text}
TARGET JOB DESCRIPTION:
{job_desc}
Provide your answer as a bulleted list showing "Original", "Suggested Edit", and "Reason".
""")
tailor_chain = tailor_prompt | llm

def tailor_resume(job_description: str, resume_text: str) -> str:
    if "No resume uploaded" in resume_text:
        return "⚠️ **I need you to upload your resume first!**"
    if not job_description:
        return "⚠️ **I couldn't find a valid job description in my memory.**"
        
    result = tailor_chain.invoke({"resume_text": resume_text, "job_desc": job_description})
    raw_content = result.content
    text_content = "".join([part.get("text", "") for part in raw_content if isinstance(part, dict)]) if isinstance(raw_content, list) else str(raw_content)
    return "\n\n### ✍️ Resume Tailoring Suggestions:\n\n" + text_content

cover_letter_prompt = ChatPromptTemplate.from_template("""
You are an expert Career Coach. Write a compelling, 3-paragraph cover letter for the user based on their RESUME and the TARGET JOB DESCRIPTION.
Format it professionally. Focus on how the user's specific past experience solves the employer's needs outlined in the job description.
Do NOT invent facts, degrees, or jobs the user did not have.
RESUME:
{resume_text}
TARGET JOB DESCRIPTION:
{job_desc}
""")
cover_letter_chain = cover_letter_prompt | llm

def write_cover_letter(job_description: str, resume_text: str) -> str:
    if "No resume uploaded" in resume_text:
        return "⚠️ **I need you to upload your resume first to write a cover letter!**"
    if not job_description:
        return "⚠️ **I couldn't find a valid job description in my memory.**"
        
    result = cover_letter_chain.invoke({"resume_text": resume_text, "job_desc": job_description})
    raw_content = result.content
    text_content = "".join([part.get("text", "") for part in raw_content if isinstance(part, dict)]) if isinstance(raw_content, list) else str(raw_content)
    return "\n\n### ✉️ Generated Cover Letter:\n\n" + text_content

# ==========================================
# 3. CONVERSATION LOGGING
# ==========================================
def save_to_memory(user_message: str, bot_response: str, category: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists("chat_logs.csv")
    with open("chat_logs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Category", "User Message", "Bot Response"])
        writer.writerow([timestamp, category, user_message, bot_response])

# ==========================================
# 4. MEMORY CLASSIFIER & ROUTER
# ==========================================
classifier_prompt = ChatPromptTemplate.from_template("""
You are a routing agent. Read the user's message and the recent chat history to determine their intent.
If they say "tailor it" or "write one for that job", use the history to understand what they want.
Recent Chat History:
{history}
Current User Message: {message}
Categories:
- SEARCH: Wants to find a job only.
- TAILOR: Wants to adjust their resume for a job they just searched for, or a pasted job text.
- COVER_LETTER: Wants a cover letter for a job they just searched for.
- SEARCH_AND_TAILOR: Wants to find a job AND tailor their resume automatically.
- FULL_PACKAGE: Wants to find a job, tailor the resume, AND write a cover letter.
- CHAT: General greeting.
Respond EXACTLY in this 2-line format:
CATEGORY: [category]
QUERY: [the search string if they are searching, or None]
""")
classifier_chain = classifier_prompt | llm

def handle_user_request(message: str, history: list, resume_file, session_state: dict) -> str:
    # --- Robust history parsing for Gradio 6.0 dictionaries ---
    formatted_history = ""
    if history:
        for item in history[-4:]: # Grabs last 2 turns
            if isinstance(item, dict):
                role = item.get("role", "User")
                content = str(item.get("content", ""))[:100]
                formatted_history += f"{role}: {content}...\n"
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                formatted_history += f"User: {str(item[0])[:100]}...\nBot: {str(item[1])[:100]}...\n"
                
    if not formatted_history.strip():
        formatted_history = "No history yet."
        
    # 1. Classify and Extract Parameters
    result = classifier_chain.invoke({"message": message, "history": formatted_history})
    raw_content = result.content
    full_text = "".join([part.get("text", "") for part in raw_content if isinstance(part, dict)]) if isinstance(raw_content, list) else str(raw_content)
        
    lines = full_text.strip().split('\n')
    try:
        category, query = "CHAT", "None"
        for line in lines:
            if "CATEGORY:" in line.upper(): category = line.split(":", 1)[1].strip().upper()
            if "QUERY:" in line.upper(): query = line.split(":", 1)[1].strip()
    except:
        category, query = "CHAT", "None"
        
    file_path = resume_file if resume_file else None
    resume_text = read_resume(file_path)
    
    # --- SMART MEMORY ROUTING LOGIC ---
    if "FULL_PACKAGE" in category:
        job_data = search_google_jobs(query)
        if job_data["top_job_context"]:
            session_state['last_job'] = job_data["top_job_context"] 
            tailored = tailor_resume(job_data["top_job_context"], resume_text)
            
            # API Rate Limit Protection (Wait 15 seconds)
            time.sleep(15) 
            
            cover_letter = write_cover_letter(job_data["top_job_context"], resume_text)
            response = job_data["readable"] + tailored + cover_letter
        else:
            response = job_data["readable"] 
            
    elif "SEARCH_AND_TAILOR" in category:
        job_data = search_google_jobs(query)
        if job_data["top_job_context"]:
            session_state['last_job'] = job_data["top_job_context"] 
            tailored = tailor_resume(job_data["top_job_context"], resume_text)
            response = job_data["readable"] + tailored
        else:
            response = job_data["readable"]
            
    elif "SEARCH" in category:
        job_data = search_google_jobs(query)
        if job_data["top_job_context"]:
            session_state['last_job'] = job_data["top_job_context"] 
        response = job_data["readable"]
        
    elif "TAILOR" in category:
        job_desc = message if len(message) > 100 else session_state.get('last_job')
        response = tailor_resume(job_desc, resume_text)
        
    elif "COVER_LETTER" in category:
        job_desc = message if len(message) > 100 else session_state.get('last_job')
        response = write_cover_letter(job_desc, resume_text)
        
    else:
        response = "Hi! I am your Job Autopilot. Ask me to find a job, then ask me to tailor your resume for it!"
        
    save_to_memory(message, response, category)
    return response

# ==========================================
# 5. GRADIO INTERFACE (Sleek Theme)
# ==========================================
custom_theme = gr.themes.Monochrome(
    primary_hue="neutral", secondary_hue="neutral", neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50", block_background_fill="white", block_border_width="1px",
    block_border_color="*neutral_200", button_primary_background_fill="*neutral_900",
    button_primary_background_fill_hover="*neutral_700", button_primary_text_color="white",
)

css = """
footer {display: none !important;}
.gradio-container {max-width: 900px !important; margin: auto;}
.message-wrap .user {background-color: #f1f5f9 !important; border-radius: 12px 12px 0px 12px !important;}
.message-wrap .bot {background-color: white !important; border: 1px solid #e2e8f0 !important; border-radius: 12px 12px 12px 0px !important;}
"""

with gr.Blocks() as demo:
    # A hidden state dictionary to hold the job data across turns
    session_state = gr.State({})
    
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-weight: 800; font-size: 2.5rem; letter-spacing: -0.025em; color: #0f172a;">Autopilot.</h1>
            <p style="color: #64748b; font-size: 1.1rem;">Upload your resume. Find jobs. Auto-tailor your application.</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            resume_upload = gr.File(label="📄 Upload Master Resume", file_types=[".txt", ".pdf", ".docx"], type="filepath")
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=handle_user_request,
                additional_inputs=[resume_upload, session_state], 
                examples=[
                    ["Find me an entry-level IT Audit job in Seattle.", None], 
                    ["Give me the full package (find a job, tailor resume, write cover letter).", None]
                ],
                cache_examples=False
            )
            with gr.Row():
                download_btn = gr.DownloadButton("📥 Download Chat Logs (CSV)", value="chat_logs.csv")

if __name__ == "__main__":
    demo.launch(theme=custom_theme, css=css)
