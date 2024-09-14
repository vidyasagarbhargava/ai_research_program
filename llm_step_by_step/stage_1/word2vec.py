# Training a Embedding Model
# Step 1 - Getting our corpus and tokenized words 
# Step 2 - Creating Vocab and mapping text to integer
# Step 3 - Creating Context Pairs or Bigrams
# Step 4 - Model Architecture
# Step 5 - Traning Model and saving




# Step 1 - Getting our corpus and tokenized words 
corpus=[["this movie had a brilliant story line with great action"],
["some parts were not so great but overall pretty ok"],
["my dog went to sleep watching this piece of trash"]] 



from tqdm import tqdm
import numpy as np
import random


def get_tokenized_corpus(corpus):
    return [sentence[0].split(" ") for sentence in corpus]

tokenized_corpus = get_tokenized_corpus(corpus)
print(tokenized_corpus)



# Step 2 - Creating Vocab and mapping text to integer
vocabulary = set()
for review in tokenized_corpus:
    vocabulary.update(review)

print("LENGTH OF VOCAB:", len(vocabulary), "\nVOCAB:", vocabulary)


word2idx = {}
n_words = 0

for token in vocabulary:
    if token not in word2idx:
        word2idx[token] = n_words
        n_words += 1


# Step 3 - Creating Context Pairs or Bigrams

def get_focus_context_pairs(tokenized_corpus, window_size=2):
    focus_context_pairs = []
    for sentence in tokenized_corpus:

        for token_idx, token in enumerate(sentence):
            for w in range(-window_size, window_size+1):
                context_word_pos = token_idx + w

                if w == 0 or context_word_pos >= len(sentence) or context_word_pos < 0:
                    continue

                try:
                    focus_context_pairs.append([token, sentence[context_word_pos]])
                except:
                    continue
    
    return focus_context_pairs
                
focus_context_pairs = get_focus_context_pairs(tokenized_corpus)
print(focus_context_pairs)



# Let's map these to our indicies in preparation to one-hot
def get_focus_context_idx(focus_context_pairs):
    idx_pairs = []
    for pair in focus_context_pairs:
        idx_pairs.append([word2idx[pair[0]], word2idx[pair[1]]])
    
    return idx_pairs

idx_pairs = get_focus_context_idx(focus_context_pairs)
print(idx_pairs)

def get_one_hot(indicies, vocab_size=len(vocabulary)):
    oh_matrix = np.zeros((len(indicies), vocab_size))
    for i, idx in enumerate(indicies):
        oh_matrix[i, idx] = 1

    return torch.Tensor(oh_matrix)

# Step 4 - Model Architecture
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import random


class Word2Vec(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim_size):
        super().__init__()
        
        # Why do you think we don't have an activation function here?
        self.projection = nn.Linear(input_size, hidden_dim_size, bias=False)
        self.output = nn.Linear(hidden_dim_size, output_size)
        
    def forward(self, input_token):
        x = self.projection(input_token)
        output = self.output(x)
        return output

#word2idx = {k: v for k, v in sorted(word2idx.items(), key=lambda item: item[1])}


# Step 5 - Traning Model and saving

def train(word2vec_model, idx_pairs, state_dict_filename, early_stop=False, num_epochs=10, lr=1e-3):

    word2vec_model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(word2vec_model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):

        random.shuffle(idx_pairs)

        for focus, context in idx_pairs:
            print(focus)
            oh_inputs = get_one_hot([focus], len(vocabulary))
            target = torch.LongTensor([context])

            pred_outputs = word2vec_model(oh_inputs)

            loss = criterion(pred_outputs, target)

            loss.backward()
            optimizer.step()
            word2vec_model.zero_grad()
            
        ### These lines stop training early
            if early_stop: break
        if early_stop: break
        ###

        torch.save(word2vec_model.state_dict(), state_dict_filename)


word2vec = Word2Vec(len(vocabulary), len(vocabulary), 10)
train(word2vec, idx_pairs, "word2vec.pt")