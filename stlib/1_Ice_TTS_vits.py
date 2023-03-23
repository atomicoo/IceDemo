description = "Ice TTS vits"


def run():

    import streamlit as st

    import spacy
    from pypinyin import lazy_pinyin, Style
    # from pypinyin_dict.phrase_pinyin_data import cc_cedict
    # cc_cedict.load()
    import torch
    from huggingface_hub import snapshot_download
    from src.vits.utils.tn_zh import TextNorm
    from src.vits.utils.text import text_to_sequence
    from src import vits

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
        allow_patterns=["**/config.yaml", "**/G_latest.pth"], 
        ignore_patterns=["*spk-vae_v1.ais*"], 
        revision=REVISION, cache_dir=CACHE_DIR, token=hf_token)
        return cache_dir

    @st.cache_resource(show_spinner="Loading model")
    def load_model(version):
        try:
            return vits.load_model(
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
            return []

    normalizer = TextNorm()

    def tokenizer(text):
        return [tok.text for tok in spacy_zh(text)]

    # def convert(text):
    #     phonemes = []
    #     for tok in spacy_zh(text):
    #         if tok.pos_ != 'PUNCT':
    #             phonemes.extend([p for w in to_pingyin(tok.text) for p in to_phoneme(w)])
    #             phonemes.append('br1')
    #         else:
    #             _ = phonemes.pop()  # pop last break label
    #             if text in ['，', '：', '；']:
    #                 phonemes.append('br3')
    #             if text in ['。', '？', '！']:
    #                 phonemes.append('br4')
    #             else:
    #                 phonemes.append('br2')
    #     phonemes.pop()  # pop last break label
    #     phonemes = ['sil'] + phonemes + ['sil']
    #     return ' '.join(phonemes)

    def convert(text):
        pinyin = to_pingyin(tokenizer(normalizer(text)))
        phonemes = [p if p!='-' else 'sp' for w in pinyin for p in to_phoneme(w)]
        phonemes = ['sil'] + phonemes + ['sil']
        return ' '.join(phonemes)

    def preprocess(text):
        text = convert(text)
        phonemes = [p for p in text.split() if not p.startswith("br")]
        text_input = text_to_sequence(text)
        return text_input, phonemes

    # @st.cache_data(show_spinner="Synthesizing waveform", ttl=ST_CACHE_TTL)
    def synthesize(text_input, z_u, lid=None, sid=None, tid=None, version=None):
        return vits.inference(model, text_input, z_u, lid=lid, sid=sid, tid=tid, device=DEVICE)


    #########################################################
    ##                  Global Variables                   ##
    #########################################################

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    REPO_ID = "Atomicoo/Ice-TTS-vits"
    # CHECKPOINT = "model/G_latest.pth"
    # CONFIG = "config.yaml"
    # SUBFOLDER = "spk-vae_v1.ice"
    CACHE_DIR = ".cache"
    REVISION = "main"
    HF_TOKEN = os.getenv("HF_TOKEN_FOR_ICEDEMO")

    # CHECKPOINTS = os.path.join('.', 'checkpoints', 'Ice-TTS-vits')
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
    st.markdown("# Ice TTS vits")
    st.markdown("")

    st.markdown("### Text")
    st.markdown("")

    st.text_input(label="Input text", key="text")

    if st.session_state.text != "":
        lcolumn, rcolumn = st.columns(2)

        ## Right Column
        with rcolumn:
            st.markdown("### Visualization")
            st.markdown("")

            st.image("./asserts/umap.png")

        ## Left Column
        with lcolumn:
            st.markdown("### Voice settings")
            st.markdown("")

            st.slider(
                label='x-axis (z_u1)', 
                min_value=-3.00, max_value=+3.00, 
                value=0.00, key="z_u1")
            # st.write('x-axis set to ', st.session_state.z_u1)

            st.slider(
                label='y-axis (z_u2)',
                min_value=-3.00, max_value=+3.00, 
                value=0.00, key="z_u2")
            # st.write('y-axis set to ', st.session_state.z_u2)

        text_input, phonemes = preprocess(st.session_state.text)
        z_u = [st.session_state.z_u1, st.session_state.z_u2]

        with lcolumn:
            st.markdown("### Results")
            st.markdown("")

            st.caption("Analayzed phoneme: " + " ".join(phonemes))

            with st.spinner("Synthesizing waveform"):
                ## STYLE: general, angry, happy, sad, surprice, whisper

                logger.info(f"text: {st.session_state.text} | z_u: {z_u} "\
                            f"(version: {st.session_state.version})")

                audio = synthesize(text_input, z_u, lid=None, sid=None, tid=None, 
                                version=st.session_state.version)
                st.markdown("general")
                st.audio(audio, sample_rate=SAMPLE_RATE)

    else:
        st.warning("Please input a Chinese text.")  # TODO: 



# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    run()