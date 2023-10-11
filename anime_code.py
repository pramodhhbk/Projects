import pandas as pd 
import numpy as np 
import pickle
from sentence_transformers import SentenceTransformer, util
import streamlit as st

st.title("Anime Recommendations")

model = SentenceTransformer('all-MiniLM-L6-v2')
#embeddings = pickle.load(open('/Users/pramodh/Downloads/embedding_model.sav', 'rb'))
data = pd.read_csv("/Users/pramodh/Downloads/imdb_anime.csv")
## Parsing only relevant data such as Title , Genre, Summary
data = data[['Title','Genre','Summary']]
data.dropna(inplace=True)
data.drop_duplicates('Title',inplace=True)
summaries = np.array(data.Summary)
embeddings = model.encode(summaries, show_progress_bar=True)

def recommendations(title):
  title_index = data.index[data['Title']==title].tolist()[0]
  cos_sim={}
  for i in range(0,len(embeddings)):
    if(i!=title_index):
      cos_sim[i] = util.cos_sim(embeddings[title_index],embeddings[i])
  sorted_items = sorted(cos_sim.items(), key=lambda x: x[1], reverse=True)
  top_5_items = sorted_items[:5]
  top_5_dict = dict(top_5_items)
  print("Anime Interested: ")
  print(title)
  print(" ")
  summary = data.iloc[title_index]['Summary']
  print(summary)
  print(" ")
  print("Recommended Based on Your Previous Watch:")
  print(" ")
  list_values = []
  for key, value in top_5_dict.items():
    title = data.iloc[key]['Title']
    summary = data.iloc[key]['Summary']
    list_values.append((title, summary))
  df = pd.DataFrame(list_values, columns=['Title', 'Summary'])
  return df.reset_index(drop=True)





##st.write(data)
selected_value = st.selectbox("Select a Anime from the List:", data['Title'])
##st.write(selected_value)
df = recommendations(selected_value)
st.write(f"Similar to {selected_value}:")
st.write(df[['Title','Summary']])




