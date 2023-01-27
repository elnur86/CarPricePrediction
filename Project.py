import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import datetime

import altair as alt



icon=Image.open('images/icon.png')
banner=Image.open('images/car.png')

st.set_page_config(layout="wide",
                   page_title="Python",
                   page_icon=icon)


#----------------------------------------------------------------------------------
df=pd.read_csv("turboAzInfoCleanedData.csv")
df.rename(columns = {'Yanacaq növü':'Yanacaq_növü', 'Sürətlər qutusu':'Sürətlər_qutusu' },inplace = True)
#----------------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Avtomobilinizin Bazar Dəyəri", "Avtomobil Bazarının Vəziyyəti", "Əlavə Məlumat"])

with tab1:
    pass

with tab2:   
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        pass
    with col2:
        st.subheader("10 ən çox satılan avtomobillər")
    with col3:
        pass
    
    source = pd.DataFrame({
    'Model': (df.Marka.value_counts()/sum(df.Marka.value_counts())*100).iloc[:10].index,
    '%': (df.Marka.value_counts()/sum(df.Marka.value_counts())*100).iloc[:10].values})
    
    chart_top_ten=alt.Chart(source).mark_bar(color="#336178").encode(
    x='Model',
    y='%')
    st.altair_chart(chart_top_ten, theme=None, use_container_width=True)

    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        pass
    with col2:
        st.subheader("10 ən çox satılan mühərrik həcmləri (l)")
    with col3:
        pass
    source = pd.DataFrame({
    'Mühərrik Həcmi (l)': (df.Mühərrik.value_counts()/sum(df.Mühərrik.value_counts())*100).iloc[:10].index,
    '%': (df.Mühərrik.value_counts()/sum(df.Mühərrik.value_counts())*100).iloc[:10].values})
    
    chart_top_ten=alt.Chart(source).mark_bar(color="#336178").encode(
    x='Mühərrik Həcmi (l)',
    y='%')
    st.altair_chart(chart_top_ten, theme=None, use_container_width=True)

    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        pass
    with col2:
        st.subheader("10 ən çox satılan rənglər")
    with col3:
        pass
    source = pd.DataFrame({
    'Rənglər': (df.Rəng.value_counts()/sum(df.Rəng.value_counts())*100).iloc[:10].index,
    '%': (df.Rəng.value_counts()/sum(df.Rəng.value_counts())*100).iloc[:10].values})
    
    chart_top_ten=alt.Chart(source).mark_bar(color="#336178").encode(
    x='Rənglər',
    y='%')
    st.altair_chart(chart_top_ten, theme=None, use_container_width=True)

    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        pass
    with col2:
        st.subheader("Satışda olan avtomobillərin yanacaq müqayisəsi")
    with col3:
        pass
    source1 = pd.DataFrame({
        "category": df.Yanacaq_növü.value_counts().index, 
        "value": df.Yanacaq_növü.value_counts().values})

    chart_old_new=alt.Chart(source1).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="value", type="quantitative"),
    color=alt.Color(field="category", type="nominal"))

    st.altair_chart(chart_old_new, theme=None, use_container_width=True)
    
with tab3:
   st.markdown('**Avtomobillərin qiymətlərini təyin etmək üçün aşağıdakı Python kitabxanalardan istifadə olunub**')
   st.markdown('_BeautifulSoup_')
   st.markdown('_Pandas_')
   st.markdown('_NumPy_')
   st.markdown('_Scikit-Learn_')
   st.markdown('_Datetime_')
   st.markdown('_Altai_')
   st.markdown('_Streamlit_')
   
   st.markdown(' ')
   st.markdown('~15000 sətr avtomobil məlumatı _scrap_ olunduqdan sonra məlumatların ilkin təmizlik prossesi həyata keçirilib.\
               Mövcud sütunların qiymətə təsirini nəzərə alaraq, az təsirli və bir-birindən aslı (motorun həcmi və at gücü) sütunlar silinib.\
               Sütunlar _encode_, sətirlər isə _train_ və _test_ olaraq 80-20 nisbətində ayrılıb.\
               _Train_ məlumatlar _DecisionTreeRegressor_ modelinə öyrənildikdən sonra, _test_ məlumatlar yoxlanılıb.')
   
   st.markdown('Öyrənmiş _model_, streamlit vasitəsi ilə yazılmış tətbiqdə istifadə edilməsi üçün _deploy_ olunub.')
   st.markdown('_Model_ növbəti günlərdə daha çox məlumat ilə yenidən öyrəniləcək. Avtomobillərin qiyməti hər il - ay dəyişdiyinə görə _modelimiz_ eyni şəkildə yenilənəcək.')
   
  

#---------------------------------------------------------------------------------------

st.sidebar.image(image=banner)

# --- Menu to select the car brand
menu=st.sidebar.selectbox("Avtomobilnizin Markası",df['Marka'].sort_values().unique())


# --- Menu to selec the car model
Selected_Car=df[df['Marka']==menu]
menu_Model=st.sidebar.selectbox("Model",Selected_Car['Model'].sort_values().unique())


# --- Year of the car
today = datetime.date.today()

year = today.year
MY=st.sidebar.slider("",min_value=1950,max_value=year)

# --- Menu to select the color
menu_color=st.sidebar.selectbox("Rəng",df['Rəng'].unique())

# --- milage
milage=st.sidebar.number_input("Yürüş (km)")

if milage<10000:
    milage=1
elif milage>10000 and milage<25000:
    milage=2
elif milage>25000 and milage<50000:
    milage=3
elif milage>50000 and milage<75000:
    milage=4
elif milage>75000 and milage<100000:
    milage=5
elif milage>100000 and milage<125000:
    milage=6
elif milage>125000 and milage<150000:
    milage=7
elif milage>150000 and milage<175000:
    milage=8
elif milage>175000 and milage<200000:
    milage=9
elif milage>200000 and milage<250000:
    milage=10
elif milage>250000 and milage<300000:
    milage=11
elif milage>300000 and milage<500000:
    milage=12
elif milage>500000 and milage<1000000:
    milage=13
else:
    milage=14


# --- Menu to select the transmission
menu_transmission=st.sidebar.selectbox("Sürətlər Qutusu",df['Sürətlər_qutusu'].unique())

# --- Menu to select the fuel
menu_fuel=st.sidebar.selectbox("Yanacaq Növü",df['Yanacaq_növü'].unique())

# --- Menu to select the engine volume

engine_size=[]
for size in np.arange(100,7000,100):
    engine_size.append(size/1000)
    
menu_engine=st.sidebar.selectbox("Mühərrik",engine_size)

#----------------------------------------------------------------------------------
PredictionDataFrame=pd.read_csv('Prediction.csv',index_col=0)

#----------------------------------------------------------------------------------
if st.sidebar.button("Qiyməti Hesabla"):
    my_file = open("columns.txt", "r",encoding="utf-8")
    
    
    for col in PredictionDataFrame.columns:
        PredictionDataFrame[col].values[0]=0
        
  
    # reading the file
    data = my_file.read()
  
    # replacing end splitting the text 
    # when newline ('\n') is seen.
    list_of_columns = data.split("\n")
    my_file.close()

    
    
    for i in range(len(PredictionDataFrame.columns)):
        if menu==PredictionDataFrame.columns[i]:
            PredictionDataFrame.iloc[0:1,i:i+1]=1
        if menu_Model==PredictionDataFrame.columns[i]:
            PredictionDataFrame.iloc[0:1,i:i+1]=1
        if PredictionDataFrame.columns[i]=='İl':
            PredictionDataFrame.iloc[0:1,i:i+1]=MY
        if menu_color==PredictionDataFrame.columns[i]:
            PredictionDataFrame.iloc[0:1,i:i+1]=1
        if PredictionDataFrame.columns[i]=='Yürüş':
            PredictionDataFrame.iloc[0:1,i:i+1]=milage
        if menu_transmission==PredictionDataFrame.columns[i]:
            PredictionDataFrame.iloc[0:1,i:i+1]=1
        if menu_fuel==PredictionDataFrame.columns[i]:
            PredictionDataFrame.iloc[0:1,i:i+1]=1
        if PredictionDataFrame.columns[i]=='Mühərrik':
            PredictionDataFrame.iloc[0:1,i:i+1]=menu_engine
    
    import pickle
    document="myModel"

    loaded_model=pickle.load(open(document,'rb'))
    y_loded_model_pred=loaded_model.predict(PredictionDataFrame)
    
#    st.dataframe(PredictionDataFrame)
    
    with tab1:
        st.header(" ")
        st.header(" ")
        st.header("Sizin avtomobilinizi")
        predicted_price=int(y_loded_model_pred[0])
        low_price=round(predicted_price*0.95,-2)
        high_price=round(predicted_price*1.05,-2)
        
        
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.header(int(low_price))
        with col2:
            st.header(" - ")
        with col3:
            st.header(int(high_price))
        with col4:
            pass
        with col5:
            pass
        with col6:
            pass
        st.header("AZN aralığında avtomobilinizi sata bilərsiniz")

milage=0

