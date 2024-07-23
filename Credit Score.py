import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "plotly_white"

data=pd.read_csv("Datasets/credit_scoring.csv")
data.head()
data.info()
data.columns
data.describe()
credi_uti_plot=px.box(data,y="Credit Utilization Ratio",title='Credit Utilization Ratio Distribution')
print(credi_uti_plot)
loan_amt_plot=px.histogram(data,x="Loan Amount",nbins=20,title="Loan Amount Distribution")
print(loan_amt_plot)
df=data[["Type of Loan","Loan Amount"]]
loan_type_vs_amt=df.groupby("Type of Loan")["Loan Amount"].mean()
numeric_df = data[['Credit Utilization Ratio', 
                   'Payment History', 
                   'Number of Credit Accounts', 
                   'Loan Amount', 'Interest Rate', 
                   'Loan Term']]
corr_fig=px.imshow(numeric_df.corr(),title="COrrelation Heatmap")
print(corr_fig)
edu_mapping={"High School":1,"Bachelor":2,"Master":3,"PhD":4}
data["Education Level"]=data["Education Level"].map(edu_mapping)
emp_satus_mapping={'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}
data["Employment Status"]=data["Employment Status"].map(emp_satus_mapping)
credit_scores=[]
for index,row in data.iterrows():
    payment_his=row["Payment History"]
    credit_usage=row["Credit Utilization Ratio"]
    no_cc=row["Number of Credit Accounts"]
    edu_lvl=row["Education Level"]
    emp_status=row["Employment Status"]

    credit_score = (payment_his * 0.35) + (credit_usage * 0.30) + (no_cc * 0.15) + (edu_lvl * 0.10) + (emp_status * 0.10)
    
    credit_scores.append(credit_score)
    data['Credit Score'] = credit_scores
from sklearn.cluster import KMeans
x=data[["Credit Score"]]
x=data[["Credit Score"]]
kmeans=KMeans(n_clusters=4,n_init=4,random_state=23)
kmeans.fit(x)
data["Segments"]=kmeans.labels_
data.head()
data["Segments"]=data["Segments"].astype("category")
