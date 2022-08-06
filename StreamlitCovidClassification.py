#https://github.com/streamlit/streamlit/issues/511
#pip install --upgrade protobuf
#pip install streamlit

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import pickle
import pandas as pd
from pandas import Series
import os

names = [ "SVM",  "Naive Bayes", "LDA",
        "QDA", "Decision Tree", "Random Forest",
           "K Nearest Neighbors", "Neural Networks"]

classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000)]

mode = 1
model = classifiers[0]
name = names[0]

def select_models():
    st.sidebar.markdown("# Covid-19 Classification")
    option = st.sidebar.selectbox(
         'Select a Machine Learning Model:',
         ["SVM",  "Naive Bayes", "LDA",
        "QDA", "Decision Tree", "Random Forest",
           "K Nearest Neighbors", "Neural Networks"], index=0)
    st.sidebar.write('You selected:', option)
    if option == names[0]:
        model = classifiers[0]
        mode = 1
    elif option == names[1]:
        model = classifiers[1]
        mode = 2
    elif option == names[2]:
        model = classifiers[2]
        mode = 3
    elif option == names[3]:
        model = classifiers[3]
        mode = 4
    elif option == names[4]:
        model = classifiers[4]
        mode = 5
    elif option == names[5]:
        model = classifiers[5]
        mode = 6
    elif option == names[6]:
        model = classifiers[6]
        mode = 7
    elif option == names[7]:
        model = classifiers[7]
        mode = 8
    return mode, model

#Select test dataset size
def select_test_size():
    st.sidebar.markdown("# Select Test dataset size")
    ts = float(st.sidebar.number_input('Percentage [0-100]%:', 20)/100)
    return ts

#Train models
def train_models(model, X, y, ts):
    st.sidebar.markdown("# Training the models")
    btnResult= st.sidebar.button('Train')
    if btnResult:
        s = 'Training the ' + names[mode-1] + ' model, test_size='+ str(ts)
        st.sidebar.text(s)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts, random_state = 20)
        model.fit(X_train, y_train)
        st.markdown("# Training Result:")
        st.write("Model: "+names[mode-1] )

        cm = np.array(confusion_matrix(y_test, y_predict, labels=[2,1,0]))
        confusion = pd.DataFrame(cm, index=['is_corona', 'other','is_healthy'],
                         columns=['predicted_corona','other','predicted_healthy'])
        st.write(confusion)
        st.write(classification_report(y_test, y_predict))
        st.write("")
        score = model.score(X_train, y_train)
        st.write("Tain Accuracy: " + str((int(score*10000)/100.0))+"%")
        score = model.score(X_test, y_test)
        st.write("Test Accuracy: " + str((int(score*10000)/100.0))+"%")
        # save the model to disk
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))
    return model


 


#Prediction
def prediction():
    re = ""
    st.sidebar.markdown("# Prediction with Symptons")
    # cough  fever  sore_throat  shortness_of_breath  head_ache  age_60_and_above  gender
    feature=[0,0,0,0,0,0,0]
    cough = st.sidebar.checkbox('Cough')
    if cough:
        feature[0] = 1
    fever = st.sidebar.checkbox('Fever')
    if fever:
        feature[1] = 1
    sore_throat = st.sidebar.checkbox('Sore Throat')
    if sore_throat:
        feature[2] = 1
    shortness_of_breath = st.sidebar.checkbox('Shortness of Breath')
    if shortness_of_breath:
        feature[3] = 1
    head_ache = st.sidebar.checkbox('Headache')
    if head_ache:
        feature[4] = 1
    age_60_and_above = st.sidebar.checkbox('Age 60 and above')
    if age_60_and_above:
        feature[5] = 1
    gender = st.sidebar.checkbox('Gender (Male?)')
    if gender:
        feature[6] = 1
    print(feature)
    
    # load the model from disk
    filename = 'finalized_model.sav'
    if not(os.path.exists(filename)):
        return re
    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    #print(result)
    
    test = Series(feature, index = ['cough','fever','sore_throat',
                                            'shortness_of_breath','head_ache',
                                            'age_60_and_above','gender'])
        
    btnResult2= st.sidebar.button('Prediction')
    if btnResult2:
        st.sidebar.text('Prediciton')
        pr = loaded_model.predict(test.to_frame().T)
        if (pr==0):
          re = "Negative"
        elif(pr==1):
          re = "Other"
        else:
          re = "Positive"
    return re

def main():
    """Covid-19 Classification App"""

    st.title("Covid-19 Data Classification App")
    st.text("Build with Streamlit, Scikit-Learn and TensorFlow")
    activities = ["Covid-19 Classification","About"]
    choice = st.sidebar.selectbox("Select Activty",activities)
    if choice == 'Covid-19 Classification':
        
        covid_file = st.sidebar.file_uploader("Upload Covid-19 Data",type=['csv','txt','xlsx'])

        if covid_file is not None:
            #df = pd.read_csv('corona_tested_individuals_ver_006.english.csv')
            df = pd.read_csv(covid_file)
            #count NAN
            #print(df.isna().sum().sum())
            #drop NAN values
            df = df.dropna()
            df = df.mask(df.eq('None')).dropna()
            #print(df)
            #print(df.groupby('age_60_and_above').size())
            df['age_60_and_above'] = df['age_60_and_above'].replace(to_replace ="No", value =0)
            #df['age_60_and_above'] = df['age_60_and_above'].replace(to_replace ="None", value =-1)
            df['age_60_and_above'] = df['age_60_and_above'].replace(to_replace ="Yes", value =1)
            #print(df)
            #print(df.groupby('gender').size())
            df['gender'] = df['gender'].replace(to_replace ="female", value =0)
            df['gender'] = df['gender'].replace(to_replace ="male", value =1)
            #print(df)            
            st.text("Covid-19 Data (First 10 rows)")
            st.write(df.head(10))
            st.text("Covid-19 Data (Last 10 rows)")
            st.write(df.tail(10))
            Xs = df.drop(['test_date', 'test_indication'], axis=1)
            #print(len(Xs))
            Xs = Xs.mask(Xs.eq('None')).dropna()
            X = Xs.drop(['corona_result'], axis=1)
            #st.text("X data")
            #t = [len(a) for a in X]
            #st.write(t)
            st.write("Number of Features: ")
            st.write(len(X.iloc[0]))
            st.write("Number of Samples: ")
            st.write(len(X))
            s = Xs['corona_result']
            d = dict([(y,x) for x,y in enumerate(sorted(set(s)))])
            y = [d[x] for x in s]

            #print(X)
            #print(y)
            #print(set(y))

            mode, model0 = select_models()

            #Select test dataset size
            ts = select_test_size()
            #Train models
            model0 = train_models(model0, X, y, ts)
            #Prediction
            re = prediction()
            st.markdown("# Prediction Result:")
            st.write(re)
            
    elif choice == 'About':
        st.subheader("About Covid-19 Classification App")
        st.markdown("Built with Streamlit by [LSBU](https://www.lsbu.ac.uk/)")
        st.text("Professor Perry Xiao")
        st.success("Copyright @ 2020 London South Bank University")
if __name__ == '__main__':
    main()	
