import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from bokeh.models.widgets import Div

# Layout
st.set_page_config(page_title='Muarrikh Yazka', page_icon='🖖', layout='wide')





st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

file_name='style.css'
with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)






# Content


with st.sidebar:
    if st.button('🏠 HOME'):
        js = "window.location.href = 'https://muarrikhyazka.github.io'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

# st.markdown(
#     f'''
#         <style>
#             .sidebar .sidebar-content {{
#                 width: 5px;
#             }}
#         </style>
#     ''',
#     unsafe_allow_html=True
# )
# with st.sidebar:
#     choose = option_menu("", ["HOME", "Photo Editing", "Project Planning", "Python e-Course", "Contact"],
#                          icons=['house', 'github', 'kanban', 'book','person lines fill'],
#                          menu_icon="app-indicator", default_index=0, orientation='vertical',
#                          styles={
#         "container": {"padding": "5!important", "background-color": "#ffffff"},
#         "icon": {"color": "#ef4da0", "font-size": "25px"}, 
#         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#ffffff"},
#         "nav-link-selected": {"background-color": "#ef4da0"},
#     }
#     )



st.title('Text Analysis of Apps Google Play Review')


st.subheader('Business Understanding')
st.write(
    """
    The crypto industry continues to progress and its development has never stopped. Contributors
    of each blockchain keep developing each segment of the industry and the whole crypto ecosystem.
    This tool is designed to allow viewers to journey into the world of crypto ecosystems of some
    of the major blockchains, and compare their performance.
    This tool is designed and structured in multiple **Pages** that are accessible using the sidebar.
    Each of these Pages addresses a different segment of the crypto industry. Within each segment
    (Macro, Transfers, Swaps, NFTs, etc.) you are able to filter your desired blockchains to
    narrow/expand the comparison. By selecting a single blockchain, you can observe a deep dive
    into that particular network.
    All values for amounts, prices, and volumes are in **U.S. dollars** and the time frequency of the
    analysis was limited to the last **30 days**.
    """
)

st.subheader('Data Understanding')
st.write(
    """
    The data for this cross-chain comparison were selected from the [**Flipside Crypto**](https://flipsidecrypto.xyz)
    data platform by using its **REST API**. These queries are currently set to **re-run every 24 hours** to cover the latest
    data and are imported as a JSON file directly to each page. The data were selected with a **1 day delay** for all
    blockchains to be in sync with one another. The codes for this tool are saved and accessible in its 
    [**GitHub Repository**](https://github.com/alitaslimi/cross-chain-monitoring).
    It is worth mentioning that a considerable portion of the data used for this tool was manually decoded from the raw
    transaction data on some of the blockchains. Besides that, the names of addresses, DEXs, collections, etc. are also
    manually labeled. As the queries are updated on a daily basis to cover the most recent data, there is a chance
    that viewers encounter inconsistent data through the app. Due to the heavy computational power required to execute
    the queries, and also the size of the raw data being too large, it was not feasible to cover data for a longer period,
    or by downloading the data and loading it from the repository itself. Therefore, the REST API was selected as the
    proper form of loading data for the time being.
    Besides the codes and queries mentioned above, the following dashboards created using Flipside Crypto were used
    as the core references in developing the current tool:
    - [Flipside World Cup: Gas Guzzlers](https://app.flipsidecrypto.com/dashboard/flipsides-world-cup-gas-guzzlers-iTcitG)
    - [Flipside World Cup: USDC Transfers](https://app.flipsidecrypto.com/dashboard/flipside-world-cup-usdc-transfers-l-dWsf)
    - [Flipside World Cup: NFT Sales](https://app.flipsidecrypto.com/dashboard/flipside-world-cup-nft-sales-lDvMLG)
    - [Flipside World Cup: Cross Chain DeFi Monitoring](https://app.flipsidecrypto.com/dashboard/flipside-world-cup-cross-chain-de-fi-monitoring-bOY5ox)
    """
)

st.subheader('Future Works')
st.write(
    """
    This tool is a work in progress and will continue to be developed moving forward. Adding other blockchains,
    more KPIs and metrics, optimizing the code in general, enhancing the UI/UX of the tool, and more importantly,
    improving the data pipeline by utilizing [**Flipside ShroomDK**](https://sdk.flipsidecrypto.xyz/shroomdk) are
    among the top priorities for the development of this app. Feel free to share your feedback, suggestions, and
    also critics with me.
    """
)

c1, c2, c3 = st.columns(3)
with c1:
    st.info('**Data Analyst: [@AliTslm](https://twitter.com/AliTslm)**', icon="💡")
with c2:
    st.info('**GitHub: [@alitaslimi](https://github.com/alitaslimi)**', icon="💻") 
with c3:
    st.info('**Data: [Flipside Crypto](https://flipsidecrypto.xyz)**', icon="🧠")