import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import os
from sklearn.impute import SimpleImputer
import numpy as np


icon=Image.open('images/icon.png')
#logo=Image.open('images/logo.png')
banner=Image.open('images/car_banner.jpg')

st.set_page_config(layout="wide",
                   page_title="Python",
                   page_icon=icon)

st.title("Avtomobilimin qiyməti")
st.text("Avtomobilin bugünə olan bazar qiyməti")

# Sidebar Container

st.sidebar.image(image=banner)

df=pd.read_csv("turboAzInfoCleanedData.csv")



menu=st.sidebar.selectbox("",["Homepage","EDA","Modeling"])
if menu=='Homepage':
    
    #Homepage Container
    st.header('Homepage')
    #st.image(banner,use_column_width='always')
    
    dataset=st.selectbox("Select dataset", ["Loan Prediction","Water Potability"])
    st.markdown('Selected:  **{0}** Dataset'.format(dataset))
    
    if dataset=="Loan Prediction":
        st.warning("""
                   **The problem**:
        Dream Housing Finance company deals in all home loans.They have presence accross all urban, semi urban and rural
        The company wants to automate the loan eligibility process:
            
            """)
        
        st.info("""
        **Loan_ID** : Unique Loan ID.
        
        **Gender** : Male/Female
        
        **Married** : Applicant married Y/N
        
        **Dependents** : Number of dependents""")
    
    else:
        st.warning("""
                   
                   **The Problem**:
        Access to safe drinking-water is essential to health, a basic himan right and a component of 
        The data consist of the following rows:
            
            """)
        st.info("""
                
        **ph** : pH of 1. water (0 to 14).
        
        **Hardness** : Capacity of water to precipitate soap in mg/L
        
        **Solids** : Total dissolved solids in ppm""")
        
elif menu=='EDA':
    
    def outlier_treatment(datacolumn):
        sorted(datacolumn)
        Q1,Q3=np.percentile(datacolumn,[25,75])
        IQR=Q3-Q1
        lower_range=Q1-(1.5*IQR)
        upper_range=Q3+(1.5*IQR)
        return lower_range,upper_range
    
    def describeStat(df):
        st.dataframe(df)
        st.subheader("Statistical Values")
        df.describe().T
        
        st.subheader("Balance of Data")
        st.bar_chart(df.iloc[:,-1].value_counts())
        
        null_df=df.isnull().sum().to_frame().reset_index()
        null_df.columns=["Columns","Counts"]
        
        
        c_eda1,c_eda2,c_eda3=st.columns([2.5,1.5,2.5])
        
        c_eda1.subheader("Null Variables")
        c_eda1.dataframe(null_df)
        
        c_eda2.subheader("Imputation")
        cat_method=c_eda2.radio('Categorical',["Mode","Backfill","Ffill"])
        num_method=c_eda2.radio('Numerical',["Mode","Median"])
        
        #Feature Engineering
        c_eda2.subheader("Feature Engineering")
        balance_problem=c_eda2.checkbox('Under Sampling')
        outlier_problem=c_eda2.checkbox('Clean Outlier')
        
        if c_eda2.button("Data Preprocessing"):
            
            #Data Cleaning
            cat_array=df.iloc[:,:-1].select_dtypes(include="object").columns
            num_array=df.iloc[:,:-1].select_dtypes(exclude="object").columns
            
            if cat_array.size>0:
                if cat_method=='Mode':
                    imp_cat=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                    df[cat_array]=imp_cat.fit_transform(df[cat_array])
                elif cat_method=='Backfill':
                    df[cat_array].fillna(method='backfill',inplace=True)
                else:
                    df[cat_array].fillna(method='ffill', inplace=True)
                
            if num_array.size>0:
                if num_method=='Mode':
                    imp_num=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                else:
                    imp_num=SimpleImputer(missing_values=np.nan, strategy='median')
                df[num_array]=imp_num.fit_transform(df[num_array])
                
            df.dropna(axis=0, inplace=True)
            
            
            if balance_problem:
                
                from imblearn.under_sampling import RandomUnderSampler
                rus=RandomUnderSampler()
                X=df.iloc[:,:-1]
                Y=df.iloc[:,[-1]]
                X,Y=rus.fit_resample(X,Y)
                df=pd.concat([X,Y],axis=1)
                
            if outlier_problem:
                for col in num_array:
                    lowerbound,upperbound=outlier_treatment(df[col])
                    df[col]=np.clip(df[col],a_min=lowerbound,a_max=upperbound)
                    
            
            null_df=df.isnull().sum().to_frame().reset_index()
            null_df.columns=["Columns","Counts"]
            c_eda3.subheader("Null Variables")
            c_eda3.dataframe(null_df)
            st.subheader("Balance of Data")
            st.bar_chart(df.iloc[:,-1].value_counts())
            
            heatmap=px.imshow(df.corr())
            st.plotly_chart(heatmap)
            st.dataframe(df)
            
            if os.path.exists("formodel.csv"):
                os.remove("formodel.csv")
            df.to_csv("formodel.csv",index=False)
            
            
    #Homepage Container
    st.header('Exploratory Data Analysis')
    dataset=st.selectbox("Select dataset", ["Loan Prediction","Water Potability"])
    
    if dataset=="Loan Prediction":
        df=pd.read_csv("loan_pred.csv")
        describeStat(df)
        
    else:
        df=pd.read_csv("water_potability.csv")
        describeStat(df)
else:
    #Modelling Container
    
    st.header('Modelling')
    if not os.path.exists("formodel.csv"):
        st.header("Please Run Reprocessing")
    else:
        df=pd.read_csv("formodel.csv")
        st.dataframe(df)
        c_model1,c_model2=st.columns(2)
        
        c_model1.subheader("Scaling")
        scaling_method=c_model1.radio('', ["Standard","Robust","MinMax"])
        c_model2.subheader("Encoders")
        encoder_method=c_model2.radio('',["Label","One-Hot"])
        
        st.header("Train and Test Splitting")
        c_model1_1,c_model2_1=st.columns(2)
        
        random_state=c_model1_1.text_input("Random State")
        test_size=c_model2_1.text_input("Percentage")
        
        model=st.selectbox("Select Model",["Xgboost","Catboost"])
        st.markdown('Selected: **{0}** Model'.format(model))
        
        if st.button("Run Model"):
            cat_array=df.iloc[:,:-1].select_dtypes(include="object").columns
            num_array=df.iloc[:,:-1].select_dtypes(exclude="object").columns
            Y=df.iloc[:,[-1]]
            
            if num_array.size>0:
                if scaling_method=='Standard':
                    from sklearn.preprocessing import StandardScaler
                    sc=StandardScaler()
                elif scaling_method=='Robust':
                    from sklearn.preprocessing import RobustScaler
                    sc=RobustScaler()
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    sc=MinMaxScaler()
                df[num_array]=sc.fit_transform(df[num_array])
                
            if cat_array.size>0:
                if encoder_method=='Label':
                    from sklearn.preprocessing import LabelEncoder
                    lb=LabelEncoder()
                    for col in cat_array:
                        df[col]=lb.fit_transform(df[col])
                        
                else:
                    df.drop(df.iloc[:,[-1]],axis=1,inplace=True)
                    dms_df=df[car_array]
                    dms_df=pd.get_dummies(dms_df,drop_first=True)
                    df_=df.drop(cat_array,axis=1)
                    df=pd.concat([df_,dms_df,Y],axis=1)
                    
            st.dataframe(df)
    # Modeling Part
            
            X=df.iloc[:,:-1]
            Y=df.iloc[:,[-1]]
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=float(test_size),random_state=int(random_state))
            
            st.markdown('X_train size= {0}'.format(X_train.shape))
            st.markdown('X_test size= {0}'.format(X_test.shape))
            st.markdown('y_train size= {0}'.format(y_train.shape))
            st.markdown('y_test size= {0}'.format(y_test.shape))
            
            st.title('Congratulations. Your Model is working')
            
            if model=='Xgboost':
                import xgboost as xgb
                model=xgb.XGBClassifier().fit(X_train,y_train)
                
            else:
                from catboost import CatBoostClassifier
                model=CatBoostClassifier().fit(X_train,y_train)
            
            y_pred=model.predict(X_test)
            y_score=model.predict_proba(X_test)[:,1]
            
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            st.markdown('Confusion Matrix')
            st.write(confusion_matrix(y_test,y_pred))
            
            report=classification_report(y_test,y_pred,output_dict=True)
            df_report=pd.DataFrame(report).transpose()
            
            st.dataframe(df_report)
            
            accuracy=str(round(accuracy_score(y_test, y_pred),2))+"%"
            st.markdown("Accuracy Score = "+accuracy)
            
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, thresholds=roc_curve(y_test,y_score)
            fig=px.area(
                x=fpr,
                y=tpr,
                title='RocCurve',
                labels=dict(x='False Positive Rate',
                            y='True Positive Rate'),
                width=700,
                height=700)
            
            fig.add_shape(
                type='line',
                line=dict(dash='dash'),
                x0=0,
                x1=1,
                y0=0,
                y1=1)
            
            st.plotly_chart(fig)
            auc_score=f'AUC Score={auc(fpr,tpr):.4f}'
            st.markdown(auc_score)
            st.title('Thanks for using')