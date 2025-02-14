#%% import libraries
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% Download Books and Packages (only if requireD)
# https://www.gutenberg.org - Free Ebook Collection

nltk.download("gutenberg")
nltk.download('punkt')
nltk.download('stopwords')

#%%
from nltk.corpus import gutenberg
print(gutenberg.fileids())


# Load data Emma by Jane Austen into corpus
corpus = gutenberg.raw('austen-emma.txt')

#%% Generate vocab

from collections import Counter
from nltk.corpus import stopwords
import string

stopwords = set(stopwords.words("english"))
important_words = {"i", "he", "she", "we", "they", "you"}  # Keep these
filtered_stopwords = stopwords - important_words  # Remove only unwanted wordsåç

tokens = word_tokenize(corpus)
tokens = [word for word in tokens if word not in string.punctuation]
tokens = [word.lower() for word in tokens if word not in filtered_stopwords]

c = Counter(tokens)

print(c.most_common(10))

#%% Building Vocab  
vocab = set(tokens) # setting a unique list of words

word_to_ix = {'<PAD>': 0, '<OOV>': 1}  # Reserve first two indices for special tokens
word_to_ix.update({word: i+2 for i, word in enumerate(vocab)})  # Shift words by 2

ix_to_word = {i: word for word, i in word_to_ix.items()}

print(ix_to_word)

#%% Key variable
vocab_size = len(ix_to_word)
embedding_dim = 10
window_size = 5 # window_size is the size to cater for each side of the target.
å
print(vocab_size, embedding_dim, window_size)
# %% Generate CBOW Training Data
data = []

for idx in range(len(tokens)):
    # capture context words (words around the current idx)
    context = [
        word_to_ix.get(tokens[i], word_to_ix["<OOV>"])  # Handle OOV words
        for i in range(max(0, idx - window_size), min(len(tokens), idx + window_size + 1))
        if i != idx
    ]
    target = word_to_ix.get(tokens[idx])
    
    # padding
    while len(context) < 2 * window_size:
        context.append(word_to_ix["<PAD>"])

    data.append((context, target))

# %% Train CBOW
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam

# Convert to NumPy arrays for TensorFlow
X_train = np.array([context for context, _ in data]) # batch_size x 2 * Window_size
y_train = np.array([target for _, target in data])

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)

# CBOW Model Definition (Using Keras Functional API)
input_layer = Input(shape=(2 * window_size,))  # Shape: (batch_size, 4)
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2 * window_size)(input_layer)
mean_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(embedding_layer)  # Averaging context embeddings
output_layer = Dense(vocab_size, activation="softmax")(mean_embedding)  # Predicts the target word

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

# Model Summary
model.summary()

#%% Model Training
epochs = 20
model.fit(X_train, y_train, batch_size=128, epochs=epochs, verbose=1)

# Extract Embeddings
word_vectors = model.layers[1].get_weights()[0]  # Extract learned embeddings


#%%
# 🔹 Convert Word Indices to One-Hot Encoding for Context Words
def to_one_hot(indices, vocab_size):
    batch_size, context_size = indices.shape
    one_hot = np.zeros((batch_size, context_size, vocab_size))
    for i in range(batch_size):
        for j in range(context_size):
            one_hot[i, j, indices[i, j]] = 1
    return one_hot


X_train_one_hot = to_one_hot(X_train, vocab_size)

# 🔹 Define CBOW Model Using Dense Instead of Embedding Layer
input_layer = Input(shape=(2 * window_size, vocab_size))  # One-hot input

# Dense layer simulating embedding lookup (weights = vocab_size x embedding_dim)
dense_embedding = Dense(embedding_dim, use_bias=False)(input_layer)  # (batch_size, context_size, embedding_dim)

# Mean Pooling (Averaging word vectors)
mean_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(dense_embedding)

# Output Layer (Softmax over Vocabulary)
output_layer = Dense(vocab_size, activation="softmax")(mean_embedding)

# Define Model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

# Model Summary
model.summary()

# Train the Model
epochs = 10
model.fit(X_train_one_hot, y_train, batch_size=128, epochs=epochs, verbose=1)

# Extract Embeddings (Weights of First Dense Layer)
word_vectors = model.layers[1].get_weights()[0]  # Extract embedding weights

# Print sample word embeddings
sample_words = ["emma", "love", "friend"]
for word in sample_words:
    if word in word_to_ix:
        print(f"Embedding for '{word}': {word_vectors[word_to_ix[word]]}")
# %% Visualise
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 🔹 Extract Embedding Weights from First Dense Layer (Simulated Embedding Layer)
word_vectors = model.layers[1].get_weights()[0]  # Shape: (vocab_size, embedding_dim)

# 🔹 Reduce Dimensions Using t-SNE
tsne = TSNE(n_components=2, random_state=42)
word_vectors_2d = tsne.fit_transform(word_vectors)

# 🔹 Select a Sample of Words to Plot
num_words = 100  # Limit the number of words to avoid clutter
sample_words = list(word_to_ix.keys())[:num_words]  # Pick the first N words
sample_words = ["king", "queen", "woman", "man", "emma", "harry", "cat", "dog", "fish"]
sample_indices = [word_to_ix[word] for word in sample_words]

# 🔹 Get Their Corresponding 2D Points
word_vectors_2d_sample = word_vectors_2d[sample_indices]

# 🔹 Plot Word Embeddings
plt.figure(figsize=(12, 8))
plt.scatter(word_vectors_2d_sample[:, 0], word_vectors_2d_sample[:, 1], marker='o', color='blue')

# 🔹 Annotate Words
for i, word in enumerate(sample_words):
    plt.annotate(word, xy=(word_vectors_2d_sample[i, 0], word_vectors_2d_sample[i, 1]), fontsize=12)

plt.title("Word Embeddings Visualized Using t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
# %%
word_vectors_2d_sample
# %%
# =============================================
# 📌 CBOW Word Embeddings Training in TensorFlow
# =============================================

# ✅ 1. Import Required Libraries
import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg, stopwords
from collections import Counter
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =============================================
# ✅ 2. Download Required NLTK Datasets
# =============================================
nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("stopwords")

#%%
# =============================================
# ✅ 3. Load & Preprocess Text Data 
# =============================================
def load_gutenberg_text():
    """Loads all books from the NLTK Gutenberg dataset and concatenates them into a single text corpus."""
    corpus = ""
    for file_id in gutenberg.fileids():
        corpus += gutenberg.raw(file_id) + "\n\n"  # Add spacing between books
    return corpus


def preprocess_text(corpus):
    """Tokenizes text, removes punctuation and stopwords, and converts to lowercase."""
    stop_words = set(stopwords.words("english"))
    important_words = {"i", "he", "she", "we", "they", "you"}  # Keep personal pronouns
    filtered_stopwords = stop_words - important_words  # Remove other stopwords

    tokens = word_tokenize(corpus)  # Tokenize text
    tokens = [word.lower() for word in tokens if word.isalnum() and word not in filtered_stopwords]  # Clean tokens
    return tokens

# Load and preprocess text
corpus = load_gutenberg_text()
tokens = preprocess_text(corpus)

# Print the 10 most common words
word_counts = Counter(tokens)
print("Most common words:", word_counts.most_common(10))

#%%
# =============================================
# ✅ 4. Build Vocabulary & Word Index Mappings
# =============================================
def build_vocab(tokens):
    """Creates word-to-index and index-to-word mappings."""
    vocab = set(tokens)  # Get unique words
    word_to_ix = {'<PAD>': 0, '<OOV>': 1}  # Special tokens for padding and unknown words
    word_to_ix.update({word: i+2 for i, word in enumerate(vocab)})  # Assign index to each word
    ix_to_word = {i: word for word, i in word_to_ix.items()}  # Reverse mapping
    return word_to_ix, ix_to_word

word_to_ix, ix_to_word = build_vocab(tokens)
vocab_size = len(word_to_ix)  # Total words in vocabulary

# Print vocabulary size
print(f"Vocabulary Size: {vocab_size}")

#%%
# =============================================
# ✅ 5. Generate Training Data for CBOW
# =============================================
def generate_cbow_data(tokens, word_to_ix, window_size=5):
    """Generates (context, target) pairs for CBOW training."""
    data = []
    for idx in range(len(tokens)):
        context = [
            word_to_ix.get(tokens[i], word_to_ix["<OOV>"])
            for i in range(max(0, idx - window_size), min(len(tokens), idx + window_size + 1))
            if i != idx
        ]

        target = word_to_ix.get(tokens[idx], word_to_ix["<OOV>"])  # Target word

        # Ensure fixed-size context (pad if necessary)
        while len(context) < 2 * window_size:
            context.append(word_to_ix["<PAD>"])

        data.append((context, target))
    
    return data

# Define parameters
window_size = 5
embedding_dim = 10

# Generate CBOW training data
data = generate_cbow_data(tokens, word_to_ix, window_size)

#%%
# =============================================
# ✅ 6. Convert Training Data to NumPy Arrays
# =============================================
X_train = np.array([context for context, _ in data])  # Context words
y_train = np.array([target for _, target in data])  # Target word

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)

print("Training data shape:", X_train.shape, y_train.shape)

#%%

# =============================================
# ✅ 7. Define CBOW Model (Using One-Hot Encoding + Dense)
# =============================================
def build_cbow_model(vocab_size, embedding_dim, window_size):
    """Builds a CBOW model using a Dense layer (simulating embeddings)."""
    input_layer = Input(shape=(2 * window_size, vocab_size))  # One-hot encoded input

    # Dense layer simulating embedding lookup
    dense_embedding = Dense(embedding_dim, use_bias=False)(input_layer)  # (batch_size, context_size, embedding_dim)

    # Mean Pooling (Averaging word vectors)
    mean_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(dense_embedding)

    # Output Layer (Softmax over Vocabulary)
    output_layer = Dense(vocab_size, activation="softmax")(mean_embedding)

    # Compile Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
    
    return model

# One-hot encode input data
def to_one_hot(indices, vocab_size):
    """Converts word indices to one-hot encoded representation."""
    batch_size, context_size = indices.shape
    one_hot = np.zeros((batch_size, context_size, vocab_size))
    for i in range(batch_size):
        for j in range(context_size):
            one_hot[i, j, indices[i, j]] = 1
    return one_hot

X_train_one_hot = to_one_hot(X_train, vocab_size)

# Build and summarize model
model = build_cbow_model(vocab_size, embedding_dim, window_size)
model.summary()

# =============================================
# ✅ 8. Train CBOW Model
# =============================================
epochs = 10
model.fit(X_train_one_hot, y_train, batch_size=128, epochs=epochs, verbose=1)

# =============================================
# ✅ 9. Extract and Visualize Word Embeddings
# =============================================
word_vectors = model.layers[1].get_weights()[0]  # Extract learned embeddings

# Function to visualize embeddings using t-SNE
def visualize_embeddings(word_vectors, word_to_ix, sample_words=None, num_words=100):
    """Visualizes word embeddings using t-SNE."""
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_2d = tsne.fit_transform(word_vectors)

    # Select words to plot
    if sample_words:
        sample_indices = [word_to_ix[word] for word in sample_words if word in word_to_ix]
    else:
        sample_words = list(word_to_ix.keys())[:num_words]
        sample_indices = [word_to_ix[word] for word in sample_words]

    word_vectors_2d_sample = word_vectors_2d[sample_indices]

    # Plot embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(word_vectors_2d_sample[:, 0], word_vectors_2d_sample[:, 1], marker='o', color='blue')

    for i, word in enumerate(sample_words):
        plt.annotate(word, xy=(word_vectors_2d_sample[i, 0], word_vectors_2d_sample[i, 1]), fontsize=12)

    plt.title("Word Embeddings Visualized Using t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

# Sample words for visualization
sample_words = ["king", "queen", "man", "woman", "emma", "love", "friend", "house", "family"]
visualize_embeddings(word_vectors, word_to_ix, sample_words)
# %%
