import marimo

__generated_with = "0.9.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Political Ideologies Analysis

        This project provides a detailed analysis of political ideologies using data from the Huggingface Political Ideologies dataset. The code leverages various data science libraries and visualization tools to map, analyze, and visualize political ideology text data.
        Project Structure

        This analysis is based on huggingface dataset repository. <br>
        You can visit right [here](https://huggingface.co/datasets/JyotiNayak/political_ideologies)
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt

    from gensim.models import Word2Vec
    from sklearn.manifold import TSNE
    from umap import UMAP

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

    import re
    import string

    from gensim.models import FastText
    from wordcloud import WordCloud
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.porter import PorterStemmer
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from sklearn.model_selection import train_test_split

    import nltk

    mo.md("""
    ## 1. Import all libraries needed

    The initial cells import the necessary libraries for data handling, visualization, and word embedding.
    """)
    return (
        Bidirectional,
        Dense,
        EarlyStopping,
        Embedding,
        FastText,
        LSTM,
        PorterStemmer,
        ReduceLROnPlateau,
        Sequential,
        TSNE,
        Tokenizer,
        UMAP,
        Word2Vec,
        WordCloud,
        WordNetLemmatizer,
        alt,
        mo,
        nltk,
        np,
        pad_sequences,
        pd,
        plt,
        re,
        sns,
        stopwords,
        string,
        tf,
        train_test_split,
        word_tokenize,
    )


@app.cell
def __(nltk):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    return


@app.cell
def __(mo):
    mo.md(
        """
        Here are the mapped of label and issue type columns.

        ```yaml
        Label Mapping: {'conservative': 0, 'liberal': 1 }
        Issue Type Mapping: {
            'economic': 0, 'environmental': 1,
            'family/gender': 2, 'geo-political and foreign policy': 3,
            'political': 4, 'racial justice and immigration': 5,
            'religious': 6, 'social, health and education': 7
        }
        ```
        """
    )
    return


@app.cell
def __(mo, pd):
    df = pd.concat(
        [pd.read_parquet(f'{name}.parquet') for name in ['train', 'val', 'test']], 
        axis=0,
    )

    df = df.drop('__index_level_0__', axis=1)

    mo.md("""
    ## 2. Dataset Loading

    The dataset files (`train.parquet`, `val.parquet`, and `test.parquet`) are loaded, concatenated, and cleaned to form a single DataFrame (df). Columns are mapped to readable labels for ease of understanding.
    """)
    return (df,)


@app.cell(hide_code=True)
def __():
    label_mapping = {
        'conservative': 0, 
        'liberal': 1
    }

    issue_type_mapping = {
        'economic': 0,
        'environmental': 1,
        'family/gender': 2,
        'geo-political and foreign policy': 3,
        'political': 4,
        'racial justice and immigration': 5,
        'religious': 6,
        'social, health and education': 7
    }
    return issue_type_mapping, label_mapping


@app.cell
def __(issue_type_mapping, label_mapping):
    label_mapping_reversed = {v: k for k, v in label_mapping.items()}
    issue_type_mapping_reversed = {v: k for k, v in issue_type_mapping.items()}

    print(label_mapping_reversed)
    print(issue_type_mapping_reversed)
    return issue_type_mapping_reversed, label_mapping_reversed


@app.cell
def __(df, issue_type_mapping_reversed, label_mapping_reversed, mo):
    df['label_text'] = df['label'].replace(label_mapping_reversed)
    df['issue_type_text'] = df['issue_type'].replace(issue_type_mapping_reversed)

    labels_grouped = df['label_text'].value_counts().rename_axis('label_text').reset_index(name='counts')
    issue_types_grouped = (
        df["issue_type_text"]
        .value_counts()
        .rename_axis("issue_type_text")
        .reset_index(name="counts")
    )

    mo.md("""
    ## 3. Mapping Labels and Issue Types

    Two dictionaries map labels (conservative and liberal) and issue types (e.g., economic, environmental, etc.) to numerical values for machine learning purposes. Reversed mappings are created to convert numerical labels back into their text form.
    """)
    return issue_types_grouped, labels_grouped


@app.cell
def __(df):
    df.iloc[:, :6].head(7)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## 4. Visualizing Data Distributions

        Bar plots visualize the proportions of conservative vs. liberal ideologies and the count of different issue types. These provide an overview of the dataset composition.
        """
    )
    return


@app.cell
def __(alt, labels_grouped, mo):
    mo.ui.altair_chart(
        alt.Chart(labels_grouped).mark_bar(
            fill='#4C78A8',
            cursor='pointer',
        ).encode(
            x=alt.X('label_text', axis=alt.Axis(labelAngle=0)),
            y='counts:Q'
        )
    )
    return


@app.cell
def __(alt, issue_types_grouped, mo):
    mo.ui.altair_chart(
        alt.Chart(issue_types_grouped)
        .mark_bar(
            fill="#4C78A8",
            cursor="pointer",
        )
        .encode(
            x=alt.X(
                "issue_type_text:O",
                axis=alt.Axis(
                    labelAngle=-10, labelAlign="center", labelPadding=10
                ),
            ),
            y="counts:Q",
        )
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Text Preprocessing""")
    return


@app.cell
def __(WordCloud, df):
    all_text = ''.join(df['statement'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    return all_text, wordcloud


@app.cell
def __(plt, wordcloud):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis='off'
    plt.show()
    return


@app.cell
def __(WordNetLemmatizer, stopwords):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words


@app.cell
def __(lemmatizer, re, stop_words, word_tokenize):
    # Function for preprocessing text
    def preprocess_text(text):
        # 1. Lowercase the text
        text = text.lower()

        # 2. Remove punctuation and non-alphabetical characters
        text = re.sub(r'[^a-z\s]', '', text)

        # 3. Tokenize the text
        tokens = word_tokenize(text)

        # 4. Remove stopwords and lemmatize each token
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        return processed_tokens
    return (preprocess_text,)


@app.cell
def __(df, preprocess_text):
    # Terapkan fungsi preprocessing pada kolom 'statement'
    df['processed_statement'] = df['statement'].apply(preprocess_text)
    processed_statement = df['processed_statement']
    return (processed_statement,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 5. Word Embeddings""")
    return


@app.cell
def __(np):
    def get_doc_embedding(tokens, embeddings_model):
        vectors = [embeddings_model.wv[word] for word in tokens if word in embeddings_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(embeddings_model.vector_size)
    return (get_doc_embedding,)


@app.cell
def __(FastText, Word2Vec, processed_statement):
    embedding_models = {
      'fasttext': FastText(sentences=processed_statement, vector_size=100, window=3, min_count=1, seed=0),
      'word2vec': Word2Vec(sentences=processed_statement, vector_size=100, window=3, min_count=1, seed=0)
    }
    return (embedding_models,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### 5.1 Word Embedding using FastText""")
    return


@app.cell
def __(df, embedding_models, get_doc_embedding):
    for name, embedding_model in embedding_models.items():
        df['embeddings_' + name] = df['processed_statement'].apply(lambda x: get_doc_embedding(x, embedding_model))
    return embedding_model, name


@app.cell
def __(df, np):
    embeddings_fasttext = df['embeddings_fasttext']
    fasttext_embeddings_matrix = np.vstack(df['embeddings_fasttext'].values)
    return embeddings_fasttext, fasttext_embeddings_matrix


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## 6. Dimensionality Reduction using UMAP

        Embeddings are projected into a 2D space using UMAP for visualization. The embeddings are colored by issue type, showing clusters of similar statements.
        """
    )
    return


@app.cell
def __(UMAP, alt, df, mo, np):
    def word_embedding_2d(embedding_model, embedding_model_name):
        embeddings_matrix = np.vstack(df[f'embeddings_{embedding_model_name}'].values)
        
        umap = UMAP(n_components=2, random_state=42)
        umap_results = umap.fit_transform(embeddings_matrix)

        df[f'{embedding_model_name}_x'] = umap_results[:, 0]
        df[f'{embedding_model_name}_y'] = umap_results[:, 1]

        brush = alt.selection_interval()
        size = 350

        points1 = alt.Chart(df, height=size, width=size).mark_point().encode(
            x=f'{embedding_model_name}_x:Q',
            y=f'{embedding_model_name}_y:Q',
            color=alt.condition(brush, 'label_text', alt.value('grey')),
            tooltip=[f'{embedding_model_name}_x:Q', f'{embedding_model_name}_y:Q', 'statement:N', 'label_text:N'] 
        ).add_params(brush).properties(title='By Political Ideologies')
        
        scatter_chart1 = mo.ui.altair_chart(points1)
        
        points2 = alt.Chart(df, height=size, width=size).mark_point().encode(
            x=f'{embedding_model_name}_x:Q',
            y=f'{embedding_model_name}_y:Q',
            color=alt.condition(brush, 'issue_type_text', alt.value('grey')),
            tooltip=['x:Q', 'y:Q', 'statement:N', 'issue_type:N'] 
        ).add_params(brush).properties(title='By Issue Types')
        
        scatter_chart2 = mo.ui.altair_chart(points2)
        
        combined_chart = (scatter_chart1 | scatter_chart2)
        return combined_chart
    return (word_embedding_2d,)


@app.cell
def __(embedding_models, word_embedding_2d):
    fasttext_plot = word_embedding_2d(embedding_models['fasttext'], 'fasttext')
    return (fasttext_plot,)


@app.cell
def __(fasttext_plot, mo):
    fasttext_table = fasttext_plot.value[['statement', 'label_text', 'issue_type_text']]
    mo.vstack([
        fasttext_plot,
        fasttext_table
    ])
    return (fasttext_table,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## 7. Interactive Visualizations

        Interactive scatter plots in Altair show ideology and issue types in 2D space. A brush selection tool allows users to explore specific points and view tooltip information.

        ### Combined Scatter Plot

        Combines the two scatter plots into a side-by-side visualization for direct comparison of ideologies vs. issue types.
        Running the Code

        Run the code using the marimo.App instance. This notebook can also be run as a standalone Python script:
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Data Insights

        - Ideology Distribution: Visualizes proportions of conservative and liberal ideologies.
        - Issue Types: Bar plot reveals the diversity and frequency of issue types in the dataset.
        - Word Embeddings: Using UMAP for 2D projections helps identify clusters in political statements.
        - Interactive Exploration: Offers detailed, interactive views on ideology vs. issue type distribution.

        This code provides a thorough analysis pipeline, from data loading to interactive visualizations, enabling an in-depth exploration of political ideologies.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Building Model""")
    return


@app.cell
def __(df, embeddings_fasttext, np, train_test_split):
    X = np.array(embeddings_fasttext.tolist())
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    y = df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X, X_test, X_train, X_val, y, y_test, y_train, y_val


@app.cell
def __(X_train, df):
    all_tokens = [token for tokens in df['processed_statement'] for token in tokens]
    vocab_size = len(set(all_tokens))
    vocab_size
    input_dim = X_train.shape[1]  # Dimensi dari embedding yang digunakan (misalnya 50 atau 100)
    sent_length = X_train.shape[1]  # Ukuran dimensi per embedding

    input_dim, sent_length
    return all_tokens, input_dim, sent_length, vocab_size


@app.cell
def __(Bidirectional, Dense, LSTM, Sequential, input_dim, sent_length):
    clf_model = Sequential()
    clf_model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=(sent_length, input_dim))))  # LSTM bidirectional
    clf_model.add(Bidirectional(LSTM(16, activation='relu')))  # LSTM bidirectional
    clf_model.add(Dense(2, activation='softmax'))  # Output layer dengan softmax untuk klasifikasi biner

    clf_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    clf_model.summary()
    return (clf_model,)


@app.cell
def __(X_train, X_val, clf_model, y_train, y_val):
    model_history = clf_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=2)
    return (model_history,)


@app.cell
def __():
    # clf_model.save('models/model_dump.keras')
    # joblib.dump(model_history, 'history/history_model_dump.pkl')
    return


@app.cell
def __(joblib, tf):
    loaded_model = tf.keras.models.load_model('models/model_2.keras')
    model_history_loaded = joblib.load('history/history_model_2.pkl')

    # loaded_model = clf_model
    # model_history_loaded = model_history
    return loaded_model, model_history_loaded


@app.cell
def __(model_history_loaded, plt):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(model_history_loaded.history['accuracy'], label='Training Accuracy')
    plt.plot(model_history_loaded.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(model_history_loaded.history['loss'], label='Training Loss')
    plt.plot(model_history_loaded.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def __(X_test, loaded_model, np):
    y_pred = loaded_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    return (y_pred,)


@app.cell
def __():
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    return accuracy_score, classification_report, joblib


@app.cell
def __(accuracy_score, y_pred, y_test):
    accuracy_score(y_test, y_pred)
    return


@app.cell
def __(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


if __name__ == "__main__":
    app.run()
