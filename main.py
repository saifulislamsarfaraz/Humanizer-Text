import streamlit as st
from transformer.app import AcademicTextHumanizer, NLP_GLOBAL, download_nltk_resources
from nltk.tokenize import word_tokenize

import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

def main():
    """
    Streamlit app for transforming AI-generated text into a formal academic style.
    Features include contraction expansion, academic transitions, optional passive
    voice conversion, and synonym replacement for more natural phrasing.
    """
    # --- Setup ---
    download_nltk_resources()

    st.set_page_config(
        page_title="AI Text Humanizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/saifulislamsarfaraz/Humanizer-Text/issues",
            "Report a bug": "https://github.com/saifulislamsarfaraz/Humanizer-Text/issues",
            "About": "A tool for transforming AI-generated text into a formal academic style."
        }
    )

    # --- Custom Styling ---
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            margin-top: 0.5em;
        }
        .intro {
            text-align: left;
            line-height: 1.6;
            margin-bottom: 1em;
        }
        .stats {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
            margin-top: 15px;
            font-size: 0.95em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Header Section ---
    st.markdown("<div class='main-title'>🧠 Humanize AI-Generated Text 🤖</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='intro'>
        <p><b>This app refines AI-generated text into a more academic and human-like tone by:</b></p>
        <ul>
            <li>Expanding contractions</li>
            <li>Adding academic transitions</li>
            <li><em>Optionally</em> converting sentences to passive voice</li>
            <li><em>Optionally</em> replacing common words with formal synonyms</li>
        </ul>
        <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- User Options ---
    use_passive = st.checkbox("Enable Passive Voice Transformation", value=False)
    use_synonyms = st.checkbox("Enable Synonym Replacement", value=False)

    # --- Input Section ---
    user_text = st.text_area("✍️ Enter your text below:", height=200)

    uploaded_file = st.file_uploader("📁 Or upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        try:
            file_text = uploaded_file.read().decode("utf-8", errors="ignore")
            user_text = file_text
            st.success("✅ Text file loaded successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # --- Transformation Button ---
    if st.button("🚀 Transform to Academic Style"):
        if not user_text.strip():
            st.warning("⚠️ Please enter or upload some text to transform.")
        else:
            with st.spinner("⏳ Transforming text... Please wait."):
                try:
                    # Input stats
                    input_word_count = len(word_tokenize(user_text, language="english", preserve_line=True))
                    doc_input = NLP_GLOBAL(user_text)
                    input_sentence_count = len(list(doc_input.sents))

                    # Transform text
                    humanizer = AcademicTextHumanizer(
                        p_passive=0.3,
                        p_synonym_replacement=0.3,
                        p_academic_transition=0.4,
                    )
                    transformed = humanizer.humanize_text(
                        user_text,
                        use_passive=use_passive,
                        use_synonyms=use_synonyms,
                    )

                    # Output display
                    st.subheader("📝 Transformed Text")
                    st.write(transformed)

                    # Output stats
                    output_word_count = len(word_tokenize(transformed, language="english", preserve_line=True))
                    doc_output = NLP_GLOBAL(transformed)
                    output_sentence_count = len(list(doc_output.sents))

                    st.markdown(
                        f"""
                        <div class='stats'>
                        <b>📊 Input Stats:</b> {input_word_count} words | {input_sentence_count} sentences<br>
                        <b>📊 Output Stats:</b> {output_word_count} words | {output_sentence_count} sentences
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"❌ Transformation failed: {e}")

    st.markdown("---")
    st.caption("© 2025 Humanizer-Text | Developed by Saiful")


if __name__ == "__main__":
    main()
