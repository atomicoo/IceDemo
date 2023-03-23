description = "Ice SRC resnet"


def run():

    import streamlit as st

    from io import BytesIO
    import resampy
    import soundfile as sf
    from scipy import spatial
    import numpy as np
    import torch
    import pandas as pd
    from huggingface_hub import snapshot_download
    from src import resnetse

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
        allow_patterns=["**/config.yaml", "**/*_latest.model"], 
        ignore_patterns=[], 
        revision=REVISION, cache_dir=CACHE_DIR, token=hf_token)
        return cache_dir

    @st.cache_resource(show_spinner="Loading model")
    def load_model(version):
        try:
            return resnetse.load_model(
                checkpoint=os.path.join(CHECKPOINTS, version), device=DEVICE)
        except Exception as err:
            logger.info(f"Load model {version} error.")
            logger.info(err)
            return None

    # @st.cache_data(show_spinner="Extracting embeddings", ttl=ST_CACHE_TTL))
    def get_embeddings(audiodata, version=None):
        if audiodata.ndim == 2:
            audiodata = audiodata.mean(axis=1)
        if sr != SAMPLE_RATE:
            audiodata = resampy.resample(audiodata, sr, SAMPLE_RATE, filter="kaiser_best", axis=0)
        audiodata, _ = resnetse.trim_long_silences(audiodata, SAMPLE_RATE)
        return resnetse.inference(model, audiodata, device=DEVICE)

    # @st.cache_data(show_spinner="Computing scores", ttl=ST_CACHE_TTL))
    def compute_scores(targets, sources):
        scores = np.zeros((len(targets), len(sources))).astype(np.float32)
        for m, target in enumerate(targets):
            for n, source in enumerate(sources):
                scores[m, n] = 1 - spatial.distance.cosine(target.mean(axis=0), source.mean(axis=0))
        return scores


    #########################################################
    ##                  Global Variables                   ##
    #########################################################

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    REPO_ID = "Atomicoo/Ice-SRC-resnet"
    # CHECKPOINT = "model/G_latest.pth"
    # CONFIG = "config.yaml"
    # SUBFOLDER = "spk-vae_v1.ice"
    CACHE_DIR = ".cache"
    REVISION = "main"
    HF_TOKEN = os.getenv("HF_TOKEN_FOR_ICEDEMO")

    # CHECKPOINTS = os.path.join('.', 'checkpoints', 'Ice-SRC-resnet')
    CHECKPOINTS = hf_download(hf_token=HF_TOKEN)
    MODELS = [m for m in os.listdir(CHECKPOINTS) \
            if os.path.isdir(os.path.join(CHECKPOINTS, m))]

    SAMPLE_RATE = 16_000

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


    ## Main Area
    st.markdown("# Ice SRC resnetse")
    st.markdown("")

    st.markdown("### Audio")
    st.markdown("")

    lcolumn, rcolumn = st.columns(2)

    with lcolumn:
        st.file_uploader(label="Upload candidate audios", type=['wav'], accept_multiple_files=True, key="sources")

    with rcolumn:
        st.file_uploader(label="Upload target audios", type=['wav'], accept_multiple_files=True, key="targets")


    if st.session_state.targets != [] and st.session_state.sources != []:

        sources, targets = {}, {}

        pbar_text = "Extracting source embeddings"
        pbar = st.progress(0, text=pbar_text)
        step = 1 / len(st.session_state.sources)
        for index, audio in enumerate(st.session_state.sources, start=1):
            # logger.info(f"Upload audio file: {audio}")

            basename = os.path.splitext(audio.name)[0]
            audiodata, sr = sf.read(BytesIO(audio.getvalue()))
            # logger.info(audiodata.shape, sr)

            sources[basename] = get_embeddings(audiodata, version=st.session_state.version)
            pbar.progress(index * step, text=pbar_text)
        pbar.empty()

        pbar_text = "Extracting target embeddings"
        pbar = st.progress(0, text=pbar_text)
        step = 1 / len(st.session_state.targets)
        for index, audio in enumerate(st.session_state.targets, start=1):
            # logger.info(f"Upload audio file: {audio}")

            basename = os.path.splitext(audio.name)[0]
            audiodata, sr = sf.read(BytesIO(audio.getvalue()))
            # logger.info(audiodata.shape, sr)

            targets[basename] = get_embeddings(audiodata, version=st.session_state.version)
            pbar.progress(index * step, text=pbar_text)
        pbar.empty()

        st.success("Embeddings extraction done.")

        st.markdown("### Results")
        st.markdown("")

        logger.info(f"Computing similarity scores ({len(targets)} x {len(sources)})")

        scores = np.zeros((len(sources), len(targets))).astype(np.float32)
        for m, t in enumerate(targets):
            for n, s in enumerate(sources):
                scores[n, m] = 1 - spatial.distance.cosine(targets[t].mean(axis=0), sources[s].mean(axis=0))

        dframe = pd.DataFrame(scores, index=sources.keys(), columns=targets.keys())
        st.dataframe(dframe.style.highlight_max(axis=0), use_container_width=True)

    if st.session_state.targets == [] and st.session_state.sources == []:
        st.warning("To get a steady result, you'd better upload a 30~60s audiofile for each speaker.")



# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    run()