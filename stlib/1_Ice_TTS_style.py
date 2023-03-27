description = "Ice TTS style"


def run():

    import streamlit as st

    import spacy
    from pypinyin import lazy_pinyin, Style
    # from pypinyin_dict.phrase_pinyin_data import cc_cedict
    # cc_cedict.load()
    from io import BytesIO
    import resampy
    import soundfile as sf
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from core.stylespeech.utils.tn_zh import TextNorm
    from core.stylespeech.utils.text import text_to_sequence
    from core.stylespeech.utils.plot import plot_spectrogram
    from core import stylespeech

    import sys, os
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging


    #########################################################
    ##                  Helper Functions                   ##
    #########################################################

    @st.cache_data(show_spinner="Downloading models")
    def hf_download(hf_token):
        cache_dir = snapshot_download(
        repo_id=REPO_ID, 
        allow_patterns=["**/config.json", "**/*_latest.pth.tar"], 
        ignore_patterns=[], 
        revision=REVISION, cache_dir=CACHE_DIR, token=hf_token)
        return cache_dir

    @st.cache_resource(show_spinner="Loading model")
    def load_model(version):
        try:
            return stylespeech.load_model(
                checkpoint=os.path.join(CHECKPOINTS, version), device=DEVICE)
        except Exception as err:
            logger.info(f"Load model {version} error.")
            logger.info(err)
            return None

    @st.cache_resource(show_spinner="Loading spaCy")
    def load_spacy(version):
        spacy_zh = spacy.load(version)  # 'zh_core_web_sm'
        return spacy_zh

    @st.cache_data(show_spinner="Loading lexicon")
    def get_lexicon(column=2):
        with open(os.path.join('.', 'resources', 'icespeech-lexicon.txt'), 'r') as fp:
            items = [l.strip().split('\t') for l in fp.read().strip().split('\n')]
        lexicon = {l[0]: l[column] for l in items}
        return lexicon

    def to_pingyin(item):
        return lazy_pinyin(item, style=Style.TONE3, v_to_u=False, neutral_tone_with_five=True)

    def to_phoneme(item):
        if item in LEXICON:
            return LEXICON[item].split('+')
        elif 'u' in item:
            item = item.replace('u', 'v')
            return LEXICON[item].split('+')
        else:
            return ['-']

    normalizer = TextNorm()

    def tokenizer(text):
        return [tok.text for tok in spacy_zh(text)]

    def convert(text):
        text = normalizer(text).strip()
        tokens = tokenizer(text.replace(' ', '-'))
        pinyin = to_pingyin(tokens)
        phonemes = [p if p!='-' else 'sp' for w in pinyin for p in to_phoneme(w)]
        phonemes = ['sil'] + phonemes + ['sil']
        return ' '.join(phonemes)

    def preprocess(text):
        text = convert(text)
        phonemes = [p for p in text.split() if not p.startswith("br")]
        text_input = text_to_sequence(text)
        return text_input, phonemes

    # @st.cache_data(show_spinner="Synthesizing waveform", ttl=ST_CACHE_TTL)
    def synthesize(text_input, ref_audio, version=None):
        return stylespeech.inference(model, text_input, ref_audio, device=DEVICE)


    #########################################################
    ##                  Global Variables                   ##
    #########################################################

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    REPO_ID = "Atomicoo/Ice-TTS-stylespeech"
    # CHECKPOINT = "model/G_latest.pth"
    # CONFIG = "config.yaml"
    # SUBFOLDER = "spk-vae_v1.ice"
    CACHE_DIR = ".cache"
    REVISION = "main"
    HF_TOKEN = os.getenv("HF_TOKEN_FOR_ICEDEMO")

    # CHECKPOINTS = os.path.join('.', 'checkpoints', 'Ice-TTS-stylespeech')
    CHECKPOINTS = hf_download(hf_token=HF_TOKEN)
    MODELS = [m for m in os.listdir(CHECKPOINTS) \
            if os.path.isdir(os.path.join(CHECKPOINTS, m))]

    LEXICON = get_lexicon(column=2)

    SAMPLE_RATE = 24_000

    ST_CACHE_TTL = 0.1 * 3600


    #########################################################
    ##                  Website UI                         ##
    #########################################################

    ## Sidebar
    sidebar = st.sidebar

    with sidebar:
        st.markdown("### Model")
        st.markdown("")

        st.selectbox(
            label="Select model",
            options=MODELS,  # version options
            key="version")
        # st.write('Selected version is ', st.session_state.version)

        model = load_model(st.session_state.version)
        if model is None:
            sidebar.error("Error on loading model.")

        # 'zh_core_web_sm' or 'zh_core_web_trf'
        spacy_zh = load_spacy('zh_core_web_sm')
        if spacy_zh is None:
            sidebar.error("Error on loading spacy.")


    ## Main Area
    st.markdown("# Ice TTS style")
    st.markdown("")

    lcolumn, rcolumn = st.columns(2)

    with lcolumn:
        st.markdown("### Inputs")
        st.markdown("")

        st.text_area(label="Input syn text", key="text")

        st.file_uploader(label="Upload ref audios", type=['wav'], 
                         accept_multiple_files=True, key="sources")

    if st.session_state.text != "" and st.session_state.sources != []:

        text_input, phonemes = preprocess(st.session_state.text)

        with rcolumn:
            st.markdown("### Results")
            st.markdown("")

            st.caption("Analayzed phoneme: " + " ".join(phonemes))

            with st.spinner("Synthesizing waveform"):
                ## STYLE: general, angry, happy, sad, surprice, whisper

                logger.info(f"text: {st.session_state.text} "\
                            f"(version: {st.session_state.version})")

                ref_audio = []
                for index, audio in enumerate(st.session_state.sources, start=1):
                    # logger.info(f"Upload audio file: {audio}")

                    # basename = os.path.splitext(audio.name)[0]
                    audiodata, sr = sf.read(BytesIO(audio.getvalue()))

                    if audiodata.ndim == 2:
                        audiodata = audiodata.mean(axis=1)
                    if sr != SAMPLE_RATE:
                        audiodata = resampy.resample(audiodata, sr, SAMPLE_RATE, filter="kaiser_best", axis=0)

                    ref_audio.append(audiodata)

                ref_audio = np.concatenate(ref_audio, axis=0)
                # ref_audio = stylespeech.trim_long_silences(ref_audio, SAMPLE_RATE)  # TODO: check it later

                audio, melspec = synthesize(text_input, ref_audio, 
                                version=st.session_state.version)
                st.markdown("waveform")
                st.audio(audio, sample_rate=SAMPLE_RATE)
                st.markdown("spectrogram")
                # norm = lambda x: (x-x.min()) / (x.max()-x.min())
                # st.image(norm(np.rot90(melspec)), use_column_width=True)
                st.image(plot_spectrogram(melspec)[1], use_column_width=True)

    else:
        with lcolumn:
            st.warning("To get a steady result, you'd better upload 10~30s reference audiofiles.")



# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    run()