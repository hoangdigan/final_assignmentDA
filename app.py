from flask import Flask, render_template, request
import os
import webbrowser
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import scikitplot as skplt
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold


import seaborn as sns

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus


app=Flask(__name__)

def get_filename(path, filename):
    cwd = Path.cwd()
    save_name = os.path.join(cwd, path, filename)  
    return save_name  

dataset_path= "dataset"
file_name = get_filename(dataset_path, "HR_dataset.csv")
df=pd.read_csv(file_name)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/plot/')
def plot(): 
    msg1, msg2, msg3= plot_model()
    return render_template('plot.html', message_knn=msg1, message_lgr=msg2, message_dtc=msg3)

@app.route('/testing/', methods=['GET', 'POST'])
def testing():
    msg="Testing case result: "
    if request.method == "POST":
        stf = float(request.form["satisfaction"])
        evl = float(request.form["evaluatation"])
        prj = int(request.form["project"])
        hrs = int(request.form["hours"])
        yrs = int(request.form["years"])
        acc = int(request.form["accident"])
        
        # set defauft left=0
        left=0
        prm = int(request.form["promotion"])
        dpt = request.form["department"]
        slr = int(request.form["salary"])

        if slr<10000000:
            slr_text = "low"
        elif slr>= 20000000:
            slr_text ="high"
        else:
            slr_text= "medium"    

        lst = [stf, evl, prj, hrs, yrs, prm, acc, left, dpt, slr_text]
        print(lst)      
        df_length = len(df)
        df.loc[df_length] = lst

        X, y= encode_data(df)    

        # get the original dataset already encoded    
        Xb =X[:-1]
        yb = y[:-1]

        # get the item have just input already encoded   
        
        X_last = X.tail(1)
        y_last = y.tail(1)

        # split the original dataset already encoded  

        X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.3, random_state=1 )
       
        # KNN predict
        # training model
        knn= KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        
        # predit the staff have just check
        pred=knn.predict(X_last)

        msg = msg + "The ability of staff leaving of KNN is: " 
        if pred>0.5:
            msg= msg+ "LEAVE. "
        else:
            msg= msg+ "NONE LEAVE. "
        
        
        # Logistic regression prdict
        lgr= LogisticRegression(solver='lbfgs')

        # train model
        lgr.fit(X_train, y_train)


        # 6.2.2 Evalute Model
        # predict on test set

        pred = lgr.predict(X_last)

        msg = msg + "Logistic Regression is: "

        if pred>0.5:
            msg= msg + "LEAVE. "
        else:
            msg= msg + "NONE LEAVE. "
        

        # Decision Tree Classifier
        dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')

        # Train Decision Tree Classifer
        dtc = dtc.fit(X_train,y_train)

        #Predict the response for test dataset
        pred = dtc.predict(X_last)

        msg = msg + "Decision Tree Classifier is: "
        if pred>0.5:
            msg= msg + "LEAVE. "
        else:
            msg= msg + "NONE LEAVE. "
        

    return render_template('testing.html', message= msg)

def plot_model():           
    templates_path= "templates"
    static_path= "static/img"

    # 4 Exporatory Data
    # 4.1 Salary features
    # How many staff in each salary level: low, medium, high

    df["salary"].value_counts().to_frame().style.background_gradient(cmap="plasma_r")

    # display on bar abd pie chart
    labels = ["low", "medium", "high"]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"bar"}, {"type":"pie"}]])

    fig.add_trace(
        go.Bar(
            x=labels,
            y= df["salary"].value_counts(),
            marker_color = ["olive", "blue", "red"],
            showlegend=True,
            text=df["salary"].value_counts(),
            textposition='auto'
        ),
        row=1, col=1
    )

    fig.append_trace(
        go.Pie(
            labels=labels,
            values=df["salary"].value_counts().to_list(),
            hole=0.3,
            marker =dict(colors=["olive", "blue", "red"])
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=550,
        width=1000,
        title="Compare number of salary level in dataset",
        margin=dict(t=40,l=20,r=20, b=20),
        plot_bgcolor= 'rgba(0,0,0,0)'
    )
   
    file_name = get_filename(templates_path, "41_salary.html")
    fig.write_html(file_name)

    # 4.2 Explore Numeric features
    # 4.2.1 Satisfication feature

    fig=px.violin(
        df,
        y="satisfaction_level",
        box=True,
        points="all",
        labels={"Satisfaction Level":"satisfaction_level"}
    )
   
    file_name = get_filename(templates_path, "421_satisfaction.html")
    fig.write_html(file_name)

    # 4.2.2 Average Working hour monthly feature
    fig=px.violin(
        df,
        y="average_montly_hours",
        box=True,
        points="all",
        labels={"Satisfaction Level":"satisfaction_level"}
    )
   
    file_name = get_filename(templates_path, "422_workinghour.html")
    fig.write_html(file_name)

    # 4.3 Analysic affecting of demographic info to leave determination
    # 4.3.1 Salary vs Leaving
    # select new data

    df_stl = df[["salary", "left"]]

    df_stl=df_stl.groupby("salary")["left"].value_counts()
    df_stl = df_stl.unstack()
    df_stl.style.background_gradient(cmap="plasma_r")
    
    # draw sunburst chart
    fig=px.sunburst(
        data_frame=df,
        path=["salary", "left"],
        color="left",
        title ="Salary vs Leaving determination"
    )

    fig.update_traces(
        textinfo="label+percent parent"
    )

    fig.update_layout(
        margin=dict(t=40, l=0,r=0, b=0)
    )

    file_name = get_filename(templates_path, "431_salary_affect_leave.html")
    fig.write_html(file_name)    


    # 4.3.2 Promotion vs Leaving
    # draw sunburst chart

    fig=px.sunburst(
        data_frame=df,
        path=["promotion_last_5years", "left"],
        color="left",
        title ="Promotion vs Leaving determination"
    )

    fig.update_traces(
        textinfo="label+percent parent"
    )

    fig.update_layout(
        margin=dict(t=40, l=0,r=0, b=0)
    )

    
    file_name = get_filename(templates_path, "432_promotion_affect_leave.html")
    fig.write_html(file_name)  

    
    # 4.3.3 Time_spend_company vs Leaving
    # draw sunburst chart

    fig=px.sunburst(
        data_frame=df,
        path=["time_spend_company", "left"],
        color="left",
        title ="Promotion vs Leaving determination"
    )

    fig.update_traces(
        textinfo="label+percent parent"
    )

    fig.update_layout(
        margin=dict(t=40, l=0,r=0, b=0)
    )

    
    file_name = get_filename(templates_path, "433_time_affect_leave.html")
    fig.write_html(file_name)  


    # 5. Feature Engineering
    # Use function encode_data() to: 
    # Normalization data, Using OrdinalEncoder to transform categorical values
    # Label encoding & One hot Encoding
    # Split data into train & test set


    X, y = encode_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1 )
   
    # 6 Train ML_Model
    # 6.1 KNN Model

    # Select K

    error = []
    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    file_name = get_filename(static_path, "knn0.png")
    plt.savefig(file_name)

    # Implement KNN with K=10
    k=10
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred=knn.predict(X_test)
    print(round(accuracy_score(y_test, pred),3))
    message = "The accuracy score of KNN Model with K={} is: ".format(str(k)) + str(round(accuracy_score(y_test, pred),3))

    # Normalized confusion matrix for the K-NN model
   
    prediction_labels = knn.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, prediction_labels, normalize=True)
    
    file_name = get_filename(static_path, "knn1.png")
    plt.savefig(file_name)


    cm=confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # plt.show()

    file_name = get_filename(static_path, "knn2.png")
    plt.savefig(file_name)


    # 6.2 Logistic Regression Model
    # 6.2.1 Train model
    # declare an object model

    lgr= LogisticRegression(solver='lbfgs')

    # train model
    lgr.fit(X_train, y_train)


    # 6.2.2 Evalute Model
    # predict on test set

    y_pred = lgr.predict(X_test) 

    # accuracy score
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Model accuracy score: {acc}")
    message_lgr = "The accuracy score of Logistic Regression Model is: " + str(round(acc,3))

    # precision score
    pres = metrics.precision_score(y_test, y_pred)
    print(f"Model precision score: {pres}")

    # recall score
    rec = metrics.recall_score(y_test, y_pred)
    print(f"Model recall score: {rec}")

    # 6.2.3 Visualizing Confusion Matrix using Heatmap
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    file_name = get_filename(static_path, "lgr1.png")
    plt.savefig(file_name)

    # 6.2.4 ROC curse

    y_pred_proba = lgr.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)

    file_name = get_filename(static_path, "lgr2.png")
    plt.savefig(file_name)

    # 6.3 Decision Tree Model
    # 6.3.1 Train model
    # Create Decision Tree classifer object

    dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')

    # Train Decision Tree Classifer
    dtc = dtc.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = dtc.predict(X_test)

    # 6.3.2 Evalute model
    # Accuracy method
    # Model Accuracy, how often is the classifier correct?

    acc = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    message_dtc = "The accuracy score of Decision Tree Classifier Model is: " + str(round(acc,3))

    # Crosss validation method
    # declare object 

    dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')

    # define the model evaluation procedure
    cv=KFold(n_splits=3, shuffle=True, random_state=1)

    # evalue model
    result = cross_val_score(dtc, X, y, cv=cv, scoring="accuracy")

    print(f"Crosss validation Accuracy: {result.mean()}")

    # Repeat cross_validation method
    # declare object 
    dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')

    # define the model evaluation procedure
    cv= RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)

    # evaluate model
    scores = cross_val_score(dtc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    print(f"Repeat Cross validation accuracy: {scores.mean()}")

    # 6.3.3 Visualizing Decision Trees

    # Retrain model
    dtc = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')

    # Train Decision Tree Classifer
    dtc = dtc.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = dtc.predict(X_test)

    feature_cols=list(X_train.columns)
    dot_data = StringIO()
    export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    
    file_name = get_filename(static_path, "dct.png")   
    graph.write_png(file_name)
    

    return message,  message_lgr,  message_dtc

def encode_data(df):
    df_trans = df.copy()
    # 5.1 Normalization data

    standard_cols =["average_montly_hours"]
    scaler_std= StandardScaler()
    scaler_std.fit(df_trans[standard_cols])
    df_trans[standard_cols] =scaler_std.transform(df_trans[standard_cols])

    # 5.2 Using OrdinalEncoder to transform categorical values

    enc = OrdinalEncoder()
    df[["Department","salary"]] = enc.fit_transform(df[["Department","salary"]])

    # 5.3 One hot Encoding

    one_hot_enc_cols = ['satisfaction_level',
                        'last_evaluation',
                        'number_project',
                        'average_montly_hours',
                        'time_spend_company',
                        'Work_accident',
                        'promotion_last_5years',
                        'Department',
                        'salary']
    df_trans =pd.get_dummies(df_trans, columns= one_hot_enc_cols)

    df_trans.head()

    # 5.4 Split data into train & test set

    X= df_trans.drop(["left"], axis=1)
    y= df_trans["left"]

    return X, y

def open_browser():
      webbrowser.open('http://127.0.0.1:5000/')


if __name__=="__main__":
    open_browser()
    app.run(debug=True)