import streamlit as st
import spacy
import json
import os
import speech_recognition as sr
import PyPDF2
import Levenshtein
from gtts import gTTS
from io import BytesIO
from pydantic import BaseModel, Field
from audio_recorder_streamlit import audio_recorder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# -------------------- 1. CONFIG --------------------

st.set_page_config(page_title="Advanced AI English Tutor", page_icon="🎓", layout="wide")
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# -------------------- SESSION STATES --------------------

if "mistake_log" not in st.session_state:
    st.session_state.mistake_log = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "xp" not in st.session_state:
    st.session_state.xp = 0

if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0

if "error_stats" not in st.session_state:
    st.session_state.error_stats = {}

if "vocab_bank" not in st.session_state:
    st.session_state.vocab_bank = []

# -------------------- AI STRUCTURE --------------------

class TutorResponse(BaseModel):
    reply: str
    correction: str
    explanation: str

system_prompt = """
You are an expert English language tutor.
Tutor Personality: {personality}
Current Scenario: {scenario}
User's Proficiency Level: {level}

Engage naturally according to your personality.
Analyze grammar, spelling, and structure carefully.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
structured_llm = llm.with_structured_output(TutorResponse)
chain = prompt | structured_llm

# -------------------- HELPER FUNCTIONS --------------------

def generate_audio(text, accent):
    tld = "com" if accent == "American" else "co.uk"
    tts = gTTS(text=text, lang='en', tld=tld)
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

def analyze_sentence(text):
    doc = nlp(text)
    return [
        {"Word": token.text, "POS": token.pos_, "Role": token.dep_}
        for token in doc if token.pos_ not in ["PUNCT", "SPACE"]
    ]

def extract_vocabulary(text):
    doc = nlp(text)
    words = [
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 5
    ]
    return list(set(words))

def pronunciation_score(original, transcribed):
    distance = Levenshtein.distance(original.lower(), transcribed.lower())
    max_len = max(len(original), len(transcribed))
    if max_len == 0:
        return 100
    return round((1 - distance/max_len) * 100)

def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    audio_file = BytesIO(audio_bytes)
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
    try:
        return r.recognize_google(audio_data)
    except:
        return None

def extract_text_from_pdf(file_obj):
    pdf_reader = PyPDF2.PdfReader(file_obj)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("⚙️ Tutor Settings")
    current_level = st.selectbox("Proficiency Level", ["Beginner", "Intermediate", "Advanced"], index=1)
    current_scenario = st.selectbox(
        "Practice Scenario",
        ["Casual Conversation", "Job Interview", "Ordering at a Restaurant", "Academic Debate"]
    )

    personality = st.selectbox(
        "Tutor Personality",
        ["Friendly Coach", "Strict Examiner", "Business Mentor"]
    )

    accent = st.selectbox("Speaking Accent", ["American", "British"])

    st.divider()
    st.header("⭐ Progress")
    st.metric("XP Points", st.session_state.xp)
    level = st.session_state.xp // 100
    st.write(f"🎖 Level: {level}")

    if st.session_state.error_stats:
        st.subheader("📊 Weakness Chart")
        st.bar_chart(st.session_state.error_stats)

    st.divider()
    st.header("📚 Vocabulary Bank")
    for word in st.session_state.vocab_bank:
        st.write("•", word)

    st.divider()
    st.header("📄 Document Review")
    uploaded_file = st.file_uploader("Upload essay", type=["txt", "pdf"])

    if uploaded_file and st.button("Analyze Document"):
        if uploaded_file.name.endswith(".txt"):
            document_text = uploaded_file.read().decode("utf-8")
        else:
            document_text = extract_text_from_pdf(uploaded_file)

        hidden_prompt = f"Please review this essay in detail:\n\n{document_text}"
        msgs.add_user_message(hidden_prompt)
        st.rerun()

    if st.button("Clear Chat"):
        msgs.clear()
        st.session_state.mistake_log = []
        st.session_state.error_stats = {}
        st.session_state.vocab_bank = []
        st.session_state.xp = 0
        st.rerun()

# -------------------- MAIN UI --------------------

st.title(f"🎓 English Tutor: {current_scenario}")

col1, col2 = st.columns([2, 1])

with col1:
    for msg in msgs.messages:
        if msg.type == "ai":
            try:
                data = json.loads(msg.content)
                st.chat_message("ai").write(data.get("reply", ""))
            except:
                st.chat_message("ai").write(msg.content)
        else:
            if "Please review this essay" in msg.content:
                st.chat_message("human").write("*(Uploaded a document for review)*")
            else:
                st.chat_message("human").write(msg.content)

    input_col, mic_col = st.columns([5, 1])
    with input_col:
        user_text = st.chat_input("Type your message here...")
    with mic_col:
        audio_bytes = audio_recorder(text="", icon_size="2x", icon_name="microphone")

with col2:
    st.subheader("Live Feedback")
    feedback_placeholder = st.empty()
    st.divider()
    st.subheader("Sentence Structure")
    grammar_placeholder = st.empty()

# -------------------- INPUT PROCESSING --------------------

final_user_input = None

if user_text:
    final_user_input = user_text
elif audio_bytes and audio_bytes != st.session_state.last_audio:
    st.session_state.last_audio = audio_bytes
    with st.spinner("Transcribing..."):
        transcribed_text = transcribe_audio(audio_bytes)
        if transcribed_text:
            final_user_input = transcribed_text
            score = pronunciation_score(user_text or transcribed_text, transcribed_text)
            st.sidebar.metric("🎤 Pronunciation Score", f"{score}%")

# -------------------- AI PIPELINE --------------------

if final_user_input:

    st.chat_message("human").write(final_user_input)
    msgs.add_user_message(final_user_input)

    if len(final_user_input) < 1000:
        grammar_data = analyze_sentence(final_user_input)
        grammar_placeholder.dataframe(grammar_data, use_container_width=True)

    with st.spinner("Analyzing..."):

        import time

        max_retries = 3
        retry_delay = 25

        # Limit history to reduce tokens
        recent_history = msgs.messages[-6:-1]

        formatted_prompt = prompt.format_messages(
            input=final_user_input,
            history=recent_history,
            scenario=current_scenario,
            level=current_level,
            personality=personality
        )

        # ---------------- STREAMING RESPONSE ----------------
        full_reply = ""

        with st.chat_message("ai"):

            message_placeholder = st.empty()

            # Typing sound (looped while streaming)
            typing_audio_html = """
                <audio autoplay loop>
                    <source src="https://www.soundjay.com/mechanical/sounds/keyboard-1.mp3" type="audio/mpeg">
                </audio>
            """
            sound_placeholder = st.empty()
            sound_placeholder.markdown(typing_audio_html, unsafe_allow_html=True)

            try:
                for chunk in streaming_llm.stream(formatted_prompt):
                    if chunk.content:
                        full_reply += chunk.content
                        message_placeholder.markdown(full_reply + "▌")

                message_placeholder.markdown(full_reply)

            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

            # Stop typing sound
            sound_placeholder.empty()

            st.audio(generate_audio(full_reply, accent), format="audio/mp3")

        # ---------------- STRUCTURED ANALYSIS ----------------
        for attempt in range(max_retries):
            try:
                response_obj = structured_llm.invoke(formatted_prompt)
                break
            except Exception as e:
                error_message = str(e)

                if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
                    if attempt < max_retries - 1:
                        st.warning(f"Quota hit. Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                    else:
                        st.error("Daily Gemini free quota reached.")
                        st.stop()
                else:
                    st.error(f"Unexpected error: {e}")
                    st.stop()

        correction = response_obj.correction
        explanation = response_obj.explanation
        ai_reply = full_reply

        # ---------------- XP SYSTEM ----------------
        st.session_state.interaction_count += 1

        if correction.lower() == "perfect!":
            st.session_state.xp += 10
        else:
            st.session_state.xp += 3
            rule_key = explanation.lower()
            st.session_state.error_stats[rule_key] = \
                st.session_state.error_stats.get(rule_key, 0) + 1

        # ---------------- VOCAB EXTRACTION ----------------
        new_words = extract_vocabulary(ai_reply)

        for word in new_words:
            if word not in st.session_state.vocab_bank:
                st.session_state.vocab_bank.append(word)

        # ---------------- FEEDBACK ----------------
        if correction.lower() != "perfect!":
            feedback_placeholder.error(f"Correction: {correction}")
            feedback_placeholder.info(f"Rule: {explanation}")
        else:
            feedback_placeholder.success("Perfect grammar!")

        msgs.add_ai_message(response_obj.model_dump_json())

        # ---------------- QUIZ ----------------
        if st.session_state.interaction_count % 5 == 0:
            st.divider()
            st.subheader("🧠 Mini Quiz")

            quiz_prompt = f"""
            Create 3 multiple choice grammar questions based on:
            {st.session_state.mistake_log}
            Include answers at the end.
            """
            quiz = llm.invoke(quiz_prompt)
            st.write(quiz.content)