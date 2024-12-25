#%% import libraries
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

#nltk.download("gutenberg")
#nltk.download('punkt')

from nltk.corpus import gutenberg
print(gutenberg.fileids)
# %%
# Sample Data
corpus = gutenberg.raw('austen-emma.txt')

# Preprocessing
vocab = set(word_tokenize(corpus))

word_to_ix = {word: i for i, word in enumerate(vocab)}
word_to_ix['<PAD>'] = len(word_to_ix)  # Add a token for padding
ix_to_word = {i: word for word, i in word_to_ix.items()}

# Key variables
vocab_size = len(vocab)
embedding_dim = 10
window_size = 2
# %%
# Generate CBOW Training Data
data = []
words = word_tokenize(corpus)
for idx in range(len(words)):
    context = [
        words[i] for i in range(max(0, idx - window_size), min(len(words), idx + window_size + 1))
        if i != idx
    ]
    target = words[idx]
    data.append((context, target))
# %%
# One-hot encoding for words
# def one_hot_vector(word, vocab_size):
#     vec = np.zeros(vocab_size)
#     vec[word_to_ix[word]] = 1
#     return vec


# # Initialize Weights
# W1 = np.random.rand(vocab_size, embedding_dim)  # Input to Hidden
# W2 = np.random.rand(embedding_dim, vocab_size)  # Hidden to Output

# # Training Parameters
# learning_rate = 0.01
# epochs = 1000

# # Training Loop
# for epoch in range(epochs):
#     loss = 0
#     for context, target in data:
#         # Average context word embeddings
#         context_vectors = np.array([one_hot_vector(word, vocab_size) for word in context]) # [n_context_word x vocab_size]
#         context_mean = np.mean(context_vectors, axis=0) # [vocab_size]

#         hidden = np.dot(context_mean, W1) # [embedding_dimension]
#         output = np.dot(hidden, W2) # [vocab_size]

#         predicted = np.exp(output) / np.sum(np.exp(output))  # Softmax 

#         # Calculate loss
#         target_vector = one_hot_vector(target, vocab_size)
#         loss += -np.sum(target_vector * np.log(predicted))

#         # Backpropagation
#         error = predicted - target_vector
#         dW2 = np.outer(hidden, error)
#         dW1 = np.outer(context_mean, np.dot(W2, error))

#         W2 -= learning_rate * dW2
#         W1 -= learning_rate * dW1

#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss:.4f}")

# # Display Word Embeddings
# print("\nWord Embeddings (CBOW):")
# for word, idx in word_to_ix.items():
#     print(f"{word}: {W1[idx]}")

# %%
import tensorflow as tf
import numpy as np
X = []
y = []

max_context_length = window_size * 2
for context, target in data:
    context_idx = [word_to_ix[word] for word in context]
    padded_context = pad_sequences([context_idx], maxlen=max_context_length, padding='post', value=word_to_ix['<PAD>'])
    
    X.append(padded_context)
    y.append(word_to_ix[target])

X = np.array(X)
y = np.array(y)


# 🧠 CBOW Model Without Embedding Layer
class CBOWModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.W1 = tf.Variable(tf.random.normal([vocab_size, embedding_dim]), trainable=True)
        self.W2 = tf.Variable(tf.random.normal([embedding_dim, vocab_size]), trainable=True)
    
    def call(self, inputs):
        # One-hot encode context words
        one_hot = tf.one_hot(inputs, depth=vocab_size)
        
        # Compute hidden layer (Average Embedding)
        hidden = tf.reduce_mean(tf.matmul(one_hot, self.W1), axis=1)
        
        # Compute output layer
        logits = tf.matmul(hidden, self.W2)
        output = tf.nn.softmax(logits, axis=1)
        return logits, output

# ✅ Training with Explicit GPU Usage
with tf.device('/GPU:0'):  # Explicitly use GPU
    model = CBOWModel(vocab_size, embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    EPOCHS = 100
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        for i in range(len(X)):
            context_words = X[i]
            target_word = y[i]
            
            with tf.GradientTape() as tape:
                logits, predictions = model(tf.expand_dims(context_words, axis=0))
                loss = loss_fn(tf.expand_dims(target_word, axis=0), logits)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


# 🔍 Inspect Word Representations (Embeddings)
print("\nWord Embeddings (Learned from W1):")
for word, idx in word_to_idx.items():
    print(f"{word}: {model.W1.numpy()[idx]}")
# %%
