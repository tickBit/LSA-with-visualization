from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import numpy as np
import spacy
import requests

nlp = spacy.load("en_core_web_lg")

# SPARQL query to get the abstract of Isaac Newton from DBPedia
endpoint_url = "http://dbpedia.org/sparql"
query = '''
    SELECT *
    WHERE {
            ?scientist  rdfs:label      "Isaac Newton"@en ;
            dbo:abstract  ?abstarct .
            FILTER ( LANG ( ?abstarct ) = 'en' )
        }
'''

r = requests.get(endpoint_url, params = {'format': 'json', 'query': query})
data = r.json()

text = data["results"]["bindings"][0]["abstarct"]["value"]
doc = nlp(text)

# Create a list of sentences from the abstract
document = []
for sent in doc.sents:
    document.append(sent.text)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
A = vectorizer.fit_transform(document).toarray()

# Create a TruncatedSVD model with 4 components
svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
matrix = svd.fit(A)

topic_results = vectorizer.transform(document)
topic_results = svd.transform(topic_results)
tarr = topic_results.argmax(axis=1)

# Print the topics
printed_topic = []
for topic in tarr:
    if topic not in printed_topic:
        print('---\n'+f'TOPIC #{topic+1}')
        printed_topic.append(topic)
        indexArr = np.where(tarr == topic)
        indexArr = indexArr[0]
        
        for i in indexArr:
            print(document[i])

# Top 4 words for each topic
plt.subplots(figsize=(7,4))
plt.suptitle('Isaac Newton', fontsize=16)
for index, topic in enumerate(matrix.components_):
    #print(f'THE TOP 4 WORDS FOR TOPIC #{index+1}')
    #print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-4:]])
    # set size of subplots
    plt.subplot(2,2,index+1)
    plt.barh(np.arange(4), topic.argsort()[-4:], height=0.5)
    plt.yticks(np.arange(4), [vectorizer.get_feature_names()[i] for i in topic.argsort()[-4:]], fontsize=10)
    plt.title(f'Topic #{index+1}')
    plt.xlabel("Importance",fontsize=8)
    plt.tight_layout()
    
plt.show()
