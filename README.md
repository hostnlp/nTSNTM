####Datasets
The raw datasets used in the paper can be downloaded via:

20NEWS:   
http://qwone.com/~jason/20Newsgroups/

Reuters:   
https://www.nltk.org/book/ch02.html

Wikitext-103:   
https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

Rcv1-v2:   
http://trec.nist.gov/data/reuters/reuters.html

We use the same preprocessing steps as described in Miao et al. (2017), Wu et al. (2020), Nan et al. (2019), and Miao et al. (2017) to obtain the vocabulary of 20NEWS, Reuters, Wikitext-103, and Rcv1-v2, respectively.


####Model
The model can be trained on Reuters by running:

    python nTSNTM.py

The best hyperparameter values on the validation set are as follows:  
prior\_alpha = 1  
prior\_beta = 10 (Reuters) /20 (20NEWS, Wikitext-103, Rcv1-v2)   
truncation\_level = 200  
n_topic2 = 30  
decay\_rate = 0.03  
learning\_rate = 5e-4  
batch\_size = 64  
n\_epoch = 100  
hidden\_size = 256



####Requirements
tensorflow==1.12.0  
numpy==1.14.5