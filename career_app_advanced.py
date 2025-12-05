# career_app_advanced.py
import streamlit as st, pandas as pd, numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from io import BytesIO
from pathlib import Path

DATA_FILE="onet_merged.csv"

@st.cache_data
def load_data():
    if not Path(DATA_FILE).exists():
        st.error("Run data_builder_onet.py first.")
        return pd.DataFrame()
    df=pd.read_csv(DATA_FILE)
    df["Attributes_List"]=df["Attributes"].fillna("").apply(lambda s:[x.strip() for x in s.split("|") if x.strip()])
    return df

def build_mlb(df):
    all_attrs=sorted({a for row in df["Attributes_List"] for a in row})
    mlb=MultiLabelBinarizer().fit([all_attrs])
    X=mlb.transform(df["Attributes_List"])
    return mlb,X,all_attrs

def user_vector(selected,weights,mlb):
    vec=np.zeros(len(mlb.classes_))
    classes=list(mlb.classes_)
    for a,w in zip(selected,weights):
        if a in classes: vec[classes.index(a)]=w
    return vec.reshape(1,-1)

def recommend(selected,weights,mlb,X,df,k):
    sim=cosine_similarity(user_vector(selected,weights,mlb),X)[0]
    df2=df.copy(); df2["score"]=sim
    return df2.sort_values("score",ascending=False).head(k)

def export_pdf(selected,weights,recs):
    pdf=FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
    pdf.cell(0,8,"Career Recommendations",ln=True)
    pdf.cell(0,6,"Selected attributes:",ln=True)
    for a,w in zip(selected,weights): pdf.cell(0,6,f"- {a} ({w})",ln=True)
    pdf.ln(4); pdf.cell(0,6,"Top careers:",ln=True)
    for i,r in recs.reset_index(drop=True).iterrows():
        pdf.multi_cell(0,5,f"{i+1}. {r['Career_Title']} — {r['score']:.3f}")
    out=BytesIO(); pdf.output(out); out.seek(0); return out

def main():
    st.title("Career Recommendation System")
    df=load_data()
    if df.empty: return
    mlb,X,all_attrs=build_mlb(df)

    selected=st.multiselect("Select interests/skills:",all_attrs)
    weights=[st.slider(f"Weight: {s}",1,5,3) for s in selected]
    k=st.slider("How many recommendations?",1,20,5)

    if st.button("Recommend") and selected:
        recs=recommend(selected,weights,mlb,X,df,k)
        st.write(recs[["Career_Title","score"]])
        pdf=export_pdf(selected,weights,recs)
        st.download_button("Download PDF",pdf,"recommendations.pdf")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(50))

if __name__=="__main__":
    main()
