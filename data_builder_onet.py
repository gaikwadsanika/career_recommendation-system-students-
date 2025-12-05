# data_builder_onet.py
import io, zipfile, requests, pandas as pd
from pathlib import Path

ALL_OCCUP_URL = "https://www.onetonline.org/find/all/All_Occupations.csv?fmt=csv"
OUT_CSV = Path("onet_merged.csv")

def build_from_onet():
    text = requests.get(ALL_OCCUP_URL, timeout=60).content.decode("utf-8","replace")
    occ_df = pd.read_csv(io.StringIO(text))
    occ_df = occ_df.rename(columns=lambda c: c.strip())
    code_col = [c for c in occ_df.columns if c.lower()=="code"][0]
    title_col = [c for c in occ_df.columns if "occupation" in c.lower()][0]
    occ_df = occ_df.rename(columns={code_col:"O*NET-SOC",title_col:"Career_Title"})
    occ_df["Job_Zone"] = ""
    occ_df = occ_df[["O*NET-SOC","Career_Title","Job_Zone"]]

    rows=[]
    for _,r in occ_df.iterrows():
        title=str(r["Career_Title"])
        tokens=[t for t in title.replace("/"," ").replace("-"," ").split() if len(t)>2]
        attrs=[title]+tokens[:5]
        rows.append({
            "O*NET-SOC":r["O*NET-SOC"],
            "Career_Title":title,
            "Short_Description":"",
            "Attributes":"|".join(attrs),
            "Job_Zone":r["Job_Zone"]
        })

    pd.DataFrame(rows).to_csv(OUT_CSV,index=False)
    print("Saved",OUT_CSV)

if __name__=="__main__":
    build_from_onet()
