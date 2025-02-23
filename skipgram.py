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
from collections import Counter, defaultdict
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
nltk.download("punkt_tab")

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
def build_vocab(tokens, max_vocab_size=5000, min_freq=5):
    """Creates word-to-index and index-to-word mappings with OOV handling."""
    word_counts = Counter(tokens)
    
    # Filter words by minimum frequency
    filtered_words = {word: count for word, count in word_counts.items() if count >= min_freq}
    
    # Sort words by frequency and keep only the most common
    most_common_words = sorted(filtered_words, key=filtered_words.get, reverse=True)[:max_vocab_size]
    
    # Special tokens
    word_to_ix = {'<PAD>': 0, '<OOV>': 1}
    
    # Assign indices to words
    word_to_ix.update({word: i + 2 for i, word in enumerate(most_common_words)})
    ix_to_word = {i: word for word, i in word_to_ix.items()}

    # ✅ Use defaultdict to handle OOV words
    word_to_ix = defaultdict(lambda: 1, word_to_ix)  # Default to index 1 (OOV)

    return word_to_ix, ix_to_word

# Example usage
max_vocab_size = 5000  # Ensure vocabulary size is defined
word_to_ix, ix_to_word = build_vocab(tokens, max_vocab_size = max_vocab_size, min_freq=5)
vocab_size = len(word_to_ix)
print(f"Vocabulary Size: {vocab_size}")

#%%
# =============================================
# ✅ 5. Generate Training Data for Skip-gram
# =============================================
import numpy as np
import tensorflow as tf

def skipgram_data_generator(tokens, word_to_ix, window_size=5):
    """Generates (target, context) pairs for Skip-gram training (yields raw indices)."""
    for idx in range(len(tokens)):
        target = word_to_ix.get(tokens[idx], word_to_ix["<OOV>"])  # Target word as index

        # Generate context words within the window
        for i in range(max(0, idx - window_size), min(len(tokens), idx + window_size + 1)):
            if i != idx:  # Ensure target word is not included in context
                context = word_to_ix.get(tokens[i], word_to_ix["<OOV>"])  # Context word as index
                
                yield np.array(target, dtype=np.int32), np.array(context, dtype=np.int32)  # Yield target-context pair

# Define parameters
window_size = 5
vocab_size = len(word_to_ix)

# ✅ Keep dataset as word indices, apply one-hot encoding inside the model
dataset = tf.data.Dataset.from_generator(
    lambda: skipgram_data_generator(tokens, word_to_ix, window_size),
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),  # Target word index
        tf.TensorSpec(shape=(), dtype=tf.int32)   # Context word index
    )
)

#%%
# =============================================
# ✅ 6. Build CBOW Model
# =============================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_skipgram_model(vocab_size, embedding_dim):
    """
    Builds a Skip-gram model where a single word (target) is used to predict its context words.
    """

    # 🔹 Input: Word indices (target word)
    input_layer = Input(shape=(), dtype=tf.int32)  # (batch_size,)

    # 🔹 Embedding layer: Converts word index to dense vector representation
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1)
    target_embedding = embedding_layer(input_layer)  # (batch_size, embedding_dim)

    # 🔹 Output layer (Softmax over vocabulary): Predict context word
    output_layer = Dense(vocab_size, activation="softmax")(target_embedding)  # (batch_size, vocab_size)

    # ✅ Define Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

    return model

# ✅ Build and summarize the model
skipgram_model = build_skipgram_model(vocab_size, embedding_dim)
skipgram_model.summary()

#%% 
# ============================================= 
# ✅ 8. Train Skip-gram Model 
# ============================================= 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Build Skip-gram model
model = build_skipgram_model(vocab_size, embedding_dim)

# Checkpoint to save best model
checkpoint = ModelCheckpoint(
    filepath="skipgram_best_model.h5",
    monitor="loss",  # Monitor training loss
    save_best_only=True,
    mode="min",
    verbose=1
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="loss",      # Monitor training loss (no val_loss in unsupervised training)
    patience=5,          # Stop if loss doesn't improve for 5 epochs
    min_delta=1e-4,      # Ignore very small changes
    mode="min",
    verbose=1
)

# Training dataset (replace with actual tf.data.Dataset)
batch_size = 128
epochs = 50  # More epochs needed for embeddings

# ✅ Skip-gram dataset (already yields (target, context) pairs)
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Train model
model.fit(
    dataset,
    epochs=epochs,
    verbose=1,
    callbacks=[checkpoint, early_stopping]
)


#%% 
# ============================================= 
# ✅ 9. Extract and Visualize Word Embeddings 
# ============================================= 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Extract learned embeddings
word_vectors = model.layers[1].get_weights()[0]  # Embedding layer is now at index 1

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
sample_words = ["king", "queen", "man", "woman", "house", "throne","family", "cat", "dog"]
visualize_embeddings(word_vectors, word_to_ix, sample_words)
# %%
