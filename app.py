import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Bengaluru Real Estate Intelligence", page_icon="🏘️", layout="wide")

st.markdown("""
<style>
.insight-box{background:#111118;border-left:3px solid #c8f04a;padding:.8rem 1rem;margin:.4rem 0;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def generate_data():
    np.random.seed(42); n=2000
    localities = {
        'Whitefield':{'base':85,'growth':0.12,'lat':12.9698,'lon':77.7499},
        'Koramangala':{'base':120,'growth':0.09,'lat':12.9352,'lon':77.6245},
        'HSR Layout':{'base':110,'growth':0.10,'lat':12.9116,'lon':77.6473},
        'Indiranagar':{'base':130,'growth':0.08,'lat':12.9784,'lon':77.6408},
        'Electronic City':{'base':65,'growth':0.14,'lat':12.8399,'lon':77.6770},
        'Marathahalli':{'base':80,'growth':0.11,'lat':12.9591,'lon':77.6972},
        'Sarjapur':{'base':72,'growth':0.13,'lat':12.8586,'lon':77.7872},
        'Hebbal':{'base':95,'growth':0.10,'lat':13.0350,'lon':77.5970},
        'JP Nagar':{'base':105,'growth':0.09,'lat':12.9102,'lon':77.5847},
        'Bellandur':{'base':90,'growth':0.11,'lat':12.9258,'lon':77.6774},
    }
    locs = np.random.choice(list(localities.keys()), n)
    bhk  = np.random.choice([1,2,3,4], n, p=[0.15,0.45,0.30,0.10])
    area = (bhk * np.random.uniform(550,750,n) + np.random.normal(0,80,n)).clip(400,4000).astype(int)
    years = np.random.choice(range(2018,2025), n)
    floors = np.random.randint(1,25,n)
    amenities = np.random.randint(2,10,n)
    psf = np.array([localities[l]['base']*(1+localities[l]['growth']*(2025-y))*(1+0.02*b)*(1+0.003*a)*np.random.uniform(0.88,1.12)
                    for l,y,b,a in zip(locs,years,bhk,amenities)])
    price = (psf*area/100000).round(2)
    return pd.DataFrame({'locality':locs,'bhk':bhk,'area_sqft':area,'price_lakhs':price,
        'price_per_sqft':psf.round(0).astype(int),'year_built':years,'age_years':2025-years,
        'floor':floors,'total_floors':floors+np.random.randint(0,10,n),
        'parking':np.random.choice([0,1,2],n,p=[0.2,0.6,0.2]),
        'amenities_score':amenities,
        'lat':[localities[l]['lat']+np.random.uniform(-0.02,0.02) for l in locs],
        'lon':[localities[l]['lon']+np.random.uniform(-0.02,0.02) for l in locs]}), localities

@st.cache_resource
def train_model(df):
    le = LabelEncoder(); df2=df.copy()
    df2['loc_enc']=le.fit_transform(df2['locality'])
    feats=['loc_enc','bhk','area_sqft','age_years','floor','total_floors','parking','amenities_score']
    X=df2[feats]; y=df2['price_lakhs']
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=42)
    m=GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.08,random_state=42)
    m.fit(Xt,yt); return m,le,r2_score(yv,m.predict(Xv)),mean_absolute_error(yv,m.predict(Xv)),feats

df, localities = generate_data()
model, le, r2, mae, feats = train_model(df)

st.title("🏘️ Bengaluru Real Estate Intelligence")
st.caption("Price prediction · Locality analysis · Investment insights | Prajwal Markal Puttaswamy")
st.divider()

tab1,tab2,tab3,tab4 = st.tabs(["🏷️ Price Predictor","📊 Market Analysis","🗺️ Locality Map","💡 Investment Insights"])

with tab1:
    st.subheader("Predict Property Price")
    c1,c2,c3=st.columns(3)
    with c1:
        loc=st.selectbox("Locality",sorted(df['locality'].unique()))
        bhk=st.selectbox("BHK",[1,2,3,4])
        area=st.slider("Area (sqft)",400,4000,1200,50)
    with c2:
        yr=st.slider("Year Built",2005,2024,2019)
        fl=st.slider("Floor",1,30,5)
        tfl=st.slider("Total Floors",fl,35,max(fl+2,12))
    with c3:
        pk=st.selectbox("Parking",[0,1,2])
        am=st.slider("Amenities Score",1,10,6)
        if st.button("🔮 Predict Price",use_container_width=True):
            X_in=pd.DataFrame([[le.transform([loc])[0],bhk,area,2025-yr,fl,tfl,pk,am]],columns=feats)
            pred=model.predict(X_in)[0]; psf=pred*100000/area
            comps=df[(df['locality']==loc)&(df['bhk']==bhk)]
            avg=comps['price_lakhs'].mean(); diff=(pred-avg)/avg*100
            st.metric("Predicted Price",f"₹{pred:.1f}L",f"{diff:+.1f}% vs avg")
            st.metric("Price/sqft",f"₹{psf:,.0f}")
            verdict = "🟢 Below market — good value" if diff<-5 else ("🔴 Above market — premium" if diff>5 else "🟡 Fairly priced")
            st.info(verdict)

with tab2:
    st.subheader("Market Overview")
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Avg Price",f"₹{df['price_lakhs'].mean():.0f}L")
    k2.metric("Avg ₹/sqft",f"₹{df['price_per_sqft'].mean():,.0f}")
    k3.metric("Most Premium",df.groupby('locality')['price_per_sqft'].mean().idxmax())
    k4.metric("Best Value",df.groupby('locality')['price_per_sqft'].mean().idxmin())
    loc_avg=df.groupby('locality')['price_per_sqft'].mean().sort_values(ascending=True).reset_index()
    fig=px.bar(loc_avg,x='price_per_sqft',y='locality',orientation='h',
               title="Avg ₹/sqft by Locality",color='price_per_sqft',
               color_continuous_scale=['#2a2a3a','#7c6af7','#c8f04a'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0.1)')
    st.plotly_chart(fig,use_container_width=True)

with tab3:
    st.subheader("Locality Price Heatmap")
    lg=df.groupby('locality').agg(avg_psf=('price_per_sqft','mean'),avg_price=('price_lakhs','mean'),
        count=('price_lakhs','count'),lat=('lat','mean'),lon=('lon','mean')).reset_index()
    fig=px.scatter_mapbox(lg,lat='lat',lon='lon',color='avg_psf',size='avg_psf',
        hover_name='locality',color_continuous_scale=['#2a2a3a','#7c6af7','#c8f04a'],
        size_max=40,zoom=11,mapbox_style='carto-darkmatter',title="Price per sqft Map")
    fig.update_layout(height=500)
    st.plotly_chart(fig,use_container_width=True)

with tab4:
    st.subheader("Investment Insights")
    ls=df.groupby('locality').agg(avg_psf=('price_per_sqft','mean'),listings=('price_lakhs','count'),avg_age=('age_years','mean')).reset_index()
    ls['score']=(ls['avg_psf'].rank(ascending=True)*0.4+ls['avg_age'].rank(ascending=True)*0.35+ls['listings'].rank(ascending=False)*0.25).round(1)
    ls=ls.sort_values('score',ascending=False)
    fig=px.bar(ls,x='locality',y='score',color='score',title="Investment Potential Score",
               color_continuous_scale=['#2a2a3a','#7c6af7','#c8f04a'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0.1)',coloraxis_showscale=False,xaxis_tickangle=-30)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("##### Top Investment Pick: " + ls.iloc[0]['locality'])
    st.caption("Built by Prajwal Markal Puttaswamy | [Portfolio](https://prajwalmarkalputtaswamyportfolio.netlify.app/)")
