import streamlit as st


## Sidebar
sidebar = st.sidebar

with sidebar:
    st.subheader("About It")
    st.info("This is a summary intended to list some display demos about AI speech, "
            "which is hosted with [Streamlit](https://streamlit.io/cloud). "
            "For more details, please see the [Github repository](https://github.com/atomicoo/IceDemo).")

    st.subheader("Contact")
    st.markdown("Maintained by Zhou (zhouzhiyang@xiaobing.ai)")


## Main Area
st.title("__Sumary__")
# st.markdown("""
# > Author: Atom (atomicoo95@gmail.com)  
# > For more details, please see [source code](https://github.com/atomicoo/IceDemo).  
# """)
# st.markdown("")

st.subheader("TTS Demo")
st.markdown("""
- Ice TTS vits: https://ice-tts-vits.streamlit.app/
""")


st.subheader("SRC Demo")
st.markdown("""
- Ice SRC resnet: https://ice-src-resnet.streamlit.app/
""")

