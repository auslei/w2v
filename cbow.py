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
# ✅ 5. Generate Training Data for CBOW
# =============================================

def cbow_data_generator(tokens, word_to_ix, window_size=5):
    """Generates (context, target) pairs for CBOW training (yields raw indices)."""
    for idx in range(len(tokens)):
        
        context = [
            word_to_ix.get(tokens[i], word_to_ix["<OOV>"])
            for i in range(max(0, idx - window_size), min(len(tokens), idx + window_size + 1))
            if i != idx
        ]  # Context words as word indices

        target = word_to_ix.get(tokens[idx], word_to_ix["<OOV>"])  # Target word as index

        # Ensure fixed-size context (pad if necessary)
        while len(context) < 2 * window_size:
            context.append(word_to_ix["<PAD>"])
        
        vocab_size = len(word_to_ix)  # Ensure vocabulary size is defined
        target_one_hot = tf.one_hot(tf.cast(target, tf.int32), depth=vocab_size) 

        yield np.array(context, dtype=np.int32), target_one_hot.numpy()  # Yield context and target

# Define parameters
window_size = 5
embedding_dim = 10

# ✅ Keep dataset as word indices, apply one-hot encoding inside the model
dataset = tf.data.Dataset.from_generator(
    lambda: cbow_data_generator(tokens, word_to_ix, window_size),
    output_signature=(
        tf.TensorSpec(shape=(2 * window_size,), dtype=tf.int32),  # Context words (indices)
        tf.TensorSpec(shape=(vocab_size), dtype=tf.int32)  # Target (index)
    )
)


#%%
# =============================================
# ✅ 6. Build CBOW Model
# =============================================
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_cbow_model(vocab_size, embedding_dim, window_size):
    """Builds a CBOW model where one-hot encoding is done inside the model for both input and output."""

    input_layer = Input(shape=(2 * window_size,))  # Input: word indices (NOT one-hot)

    # Convert word indices to one-hot dynamically (for input)
    one_hot_input = Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), depth=vocab_size))(input_layer)

    # Dense layer simulating embedding lookup (trainable weight matrix)
    dense_embedding = Dense(embedding_dim, use_bias=False)(one_hot_input)  # (batch, context_size, embedding_dim)

    # Mean Pooling (Averaging context word vectors)
    mean_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(dense_embedding)

    # Output Layer (Softmax for predicting target word)
    output_layer = Dense(vocab_size, activation="softmax")(mean_embedding)

    # Compile Model (Now Uses Standard `categorical_crossentropy`)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

    return model


def build_cbow_model_embedding_layer(vocab_size, embedding_dim, window_size):
    """Builds a CBOW model using an Embedding layer instead of one-hot encoding."""
    
    # 🔹 Input: word indices (no need for one-hot encoding)
    input_layer = Input(shape=(2 * window_size,), dtype=tf.int32)  # (batch_size, context_size)

    # 🔹 Embedding layer replaces one-hot + dense embedding lookup
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2 * window_size)
    embedded_context = embedding_layer(input_layer)  # (batch_size, context_size, embedding_dim)

    # 🔹 Mean Pooling (Averaging context word vectors)
    mean_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=1))(embedded_context)  # (batch_size, embedding_dim)

    # 🔹 Output Layer (Softmax for predicting target word)
    output_layer = Dense(vocab_size, activation="softmax")(mean_embedding)  # (batch_size, vocab_size)

    # ✅ Define Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

    return model

# ✅ No need to precompute one-hot encoding!
# Use word indices directly
model = build_cbow_model(vocab_size, embedding_dim, window_size)
model.summary()
#%%
# =============================================
# ✅ 8. Train CBOW Model
# =============================================
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Build CBOW model
model = build_cbow_model(vocab_size, embedding_dim, window_size)

# Checkpoint to save best model
checkpoint = ModelCheckpoint(
    filepath="cbow_best_model.h5",
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
word_vectors = model.layers[2].get_weights()[0]  # Extract learned embeddings

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
