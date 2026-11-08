import warnings, logging, matplotlib; matplotlib.use("Agg"); warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt, numpy as np
cats=["Card testing","Account takeover","Velocity bursts","Overall\n(precision@k)"]
pt=[0.137,0.167,0.0,0.092]; sq=[0.923,0.875,0.583,0.791]
with plt.xkcd(scale=0.5,length=120,randomness=2):
    plt.rcParams["font.family"]=["Segoe Print","Comic Sans MS","sans-serif"]
    fig,ax=plt.subplots(figsize=(11,5.5),dpi=130); x=np.arange(4); w=0.36
    b1=ax.bar(x-w/2,pt,w,color="#c8c8c8",ec="#333",label="Point features only")
    b2=ax.bar(x+w/2,sq,w,color="#4a90c2",ec="#333",label="Sequence-aware features")
    for b,v in list(zip(b1,pt))+list(zip(b2,sq)): ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.0%}",ha="center",fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=11); ax.set_ylim(0,1.1); ax.set_yticks([])
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    ax.set_title("Share of fraud caught in the top-k alerts (same model, same data)",fontsize=14,pad=14)
    ax.legend(frameon=False,fontsize=11,loc="upper right")
    fig.savefig("tds-results-chart.png",bbox_inches="tight",facecolor="white")
