import streamlit as st 
import pandas as pd 
import numpy as np 

import plotly.graph_objs as go 
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler 

st.set_page_config (page_title="Stock Price Prediction", layout="wide")



# load dataset 
df=pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
df['Date']=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df["Date"] 


# preparing Data

data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])


new_dataset=data[['Date','Close']].reset_index(drop=True)


new_dataset.index=new_dataset['Date']
new_dataset.drop("Date",axis=1 , inplace=True) 


train_data = new_dataset[:987]
valid_data= new_dataset[987:]


scaler =MinMaxScaler(feature_range=(0,1))
scaled_train_data = scaler.fit_transform(train_data)

x_train, y_train= [],[]

for i in range(60, len(scaled_train_data)):
  x_train.append(scaled_train_data[i-60:i,0])
  y_train.append(scaled_train_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Load model and predict
model_new = load_model("C:/Users/rohit/OneDrive/Desktop/stock_market_prediction/saved_model_new.h5")

model_new.summary()
inputs = train_data[len(train_data) - len(valid_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)



X_test=[]
for i in range(60,inputs.shape[0]):
   X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=model_new.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)


# Visuals
train= new_dataset[:987]
valid=new_dataset[987:].copy()
valid["Predictions"]=predicted_closing_price


st.subheader("Actual vs Predicted Closing Prices")
fig=go.Figure()



# Plot only Training Data 
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name="Training Data", line=dict(color='blue')))


# plot Actual prices from  validation {after training ends}
fig.add_trace(go.Scatter(x=valid.index, y=valid["Close"], name='Actual Price', line=dict(color='green')))


# plot Predicted prices from the same range 
fig.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"], name='Prediction Price', line=dict(color='red')))


fig.update_layout(title="TATA Stock Price Prediction", xaxis_title="Date",yaxis_title="Price")

st.plotly_chart(fig,use_container_width=True)


st.markdown("---")
st.write("Model:LSTM")
st.write("Note: This is a simple example using past 60 days' data to predict the next closing price. The model is trained only once and loaded from a saved file.")
st.write("For a more accurate prediction,consider using more features and a more extensive dataset.")