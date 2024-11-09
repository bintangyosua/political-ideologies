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


@app.cell
def __(form, mo, try_predict):
    text_classified = 'Please write something'
    if (form.value):
        text_classified = try_predict(form.value)
    mo.vstack([form, mo.md(f"Your Opinion Classified as: **{text_classified}**")])
    return (text_classified,)


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
    df = pd.read_parquet('train.parquet')
    df_val = pd.read_parquet('val.parquet')
    df_test =  pd.read_parquet('test.parquet')

    df = df.drop('__index_level_0__', axis=1)

    mo.md("""
    ## 2. Dataset Loading

    The dataset files (`train.parquet`, `val.parquet`, and `test.parquet`) are loaded, concatenated, and cleaned to form a single DataFrame (df). Columns are mapped to readable labels for ease of understanding.
    """)
    return df, df_test, df_val


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
    mo.md(
        r"""
        ## 5. Text Preprocessing

        Texts preprocessed to remove any ineffective words.
        """
    )
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
    plt.plot()
    plt.gca()
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
def __(df, df_test, df_val, preprocess_text):
    # Terapkan fungsi preprocessing pada kolom 'statement'
    df['processed_statement'] = df['statement'].apply(preprocess_text)
    df_val['processed_statement'] = df_val['statement'].apply(preprocess_text)
    df_test['processed_statement'] = df_test['statement'].apply(preprocess_text)
    processed_statement = df['processed_statement']
    return (processed_statement,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 6. Word Embeddings""")
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
    mo.md(r"""### 6.1 Word Embedding using FastText and Word2Vec""")
    return


@app.cell
def __(df, df_test, df_val, embedding_models, get_doc_embedding):
    for name, embedding_model in embedding_models.items():
        df['embeddings_' + name] = df['processed_statement'].apply(lambda x: get_doc_embedding(x, embedding_model))
        df_val['embeddings_' + name] = df_val['processed_statement'].apply(lambda x: get_doc_embedding(x, embedding_model))
        df_test['embeddings_' + name] = df_test['processed_statement'].apply(lambda x: get_doc_embedding(x, embedding_model))
    return embedding_model, name


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        #### Dimensionality Reduction using UMAP

        Embeddings are projected into a 2D space using UMAP for visualization. The embeddings are colored by issue type, showing clusters of similar statements.

        Interactive scatter plots in Altair show ideology and issue types in 2D space. A brush selection tool allows users to explore specific points and view tooltip information.

        #### Combined Scatter Plot

        Combines the two scatter plots into a side-by-side visualization for direct comparison of ideologies vs. issue types.
        Running the Code

        Run the code using the marimo.App instance. This notebook can also be run as a standalone Python script:
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
            tooltip=[f'{embedding_model_name}_x:Q', f'{embedding_model_name}_y:Q', 'statement:N', 'issue_type:N'] 
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
    fasttext_chart = mo.vstack([
        fasttext_plot,
        fasttext_table
    ])
    return fasttext_chart, fasttext_table


@app.cell
def __(embedding_models, word_embedding_2d):
    word2vec_plot = word_embedding_2d(embedding_models['word2vec'], 'word2vec')
    return (word2vec_plot,)


@app.cell
def __(fasttext_plot, mo, word2vec_plot):
    word2vec_table = fasttext_plot.value[['statement', 'label_text', 'issue_type_text']]
    word2vec_chart = mo.vstack([
        word2vec_plot,
        word2vec_table
    ])
    return word2vec_chart, word2vec_table


@app.cell
def __(fasttext_chart, mo, word2vec_chart):
    mo.ui.tabs({
        'FastText': fasttext_chart,
        'Word2Vec': word2vec_chart
    })
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
def __(df, df_test, df_val, np):
    X_train = np.array(df['embeddings_fasttext'].tolist())
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    y_train = df['label'].values

    X_val = np.array(df_val['embeddings_fasttext'].tolist())
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    y_val = df_val['label'].values

    X_test = np.array(df_test['embeddings_fasttext'].tolist())
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_test = df_test['label'].values
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def __():
    # all_tokens = [token for tokens in df['processed_statement'] for token in tokens]
    # vocab_size = len(set(all_tokens))
    # vocab_size
    # input_dim = X_train.shape[1]  # Dimensi dari embedding yang digunakan (misalnya 50 atau 100)
    # sent_length = X_train.shape[1]  # Ukuran dimensi per embedding

    # input_dim, sent_length
    return


@app.cell
def __():
    # clf_model = Sequential()
    # clf_model.add(Bidirectional(tf.keras.layers.GRU(64, 
    #                                  activation='relu', 
    #                                  # return_sequences=True, 
    #                                  input_shape=(sent_length, input_dim),
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))))  # L2 regularization
    # clf_model.add(tf.keras.layers.Dropout(0.5))
    # clf_model.add(Dense(2, 
    #                     activation='softmax', 
    #                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))  # L2 regularization in the Dense layer

    # clf_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # clf_model.summary()
    return


@app.cell
def __():
    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-10)

    # model_history = clf_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=2, callbacks=[lr_scheduler])
    return


@app.cell
def __():
    # clf_model.save('models/model_8781.keras')
    # joblib.dump(model_history, 'history/history_model_8781.pkl')
    return


@app.cell
def __(joblib, tf):
    loaded_model = tf.keras.models.load_model('models/model_8781.keras')
    model_history_loaded = joblib.load('history/history_model_8781.pkl')

    # loaded_model = clf_model
    # model_history_loaded = model_history
    return loaded_model, model_history_loaded


@app.cell
def __(model_history_loaded, pd):
    history_data = {
        'epoch': range(1, len(model_history_loaded.history['accuracy']) + 1),
        'accuracy': model_history_loaded.history['accuracy'],
        'val_accuracy': model_history_loaded.history['val_accuracy'],
        'loss': model_history_loaded.history['loss'],
        'val_loss': model_history_loaded.history['val_loss']
    }

    history_df = pd.DataFrame(history_data)
    return history_data, history_df


@app.cell
def __(alt, history_df, mo):
    accuracy_chart = alt.Chart(history_df).transform_fold(
        ['accuracy', 'val_accuracy'],
        as_=['type', 'accuracy']
    ).mark_line().encode(
        x='epoch:Q',
        y='accuracy:Q',
        color='type:N',
        tooltip=['epoch', 'accuracy']
    ).properties(title='Training and Validation Accuracy')

    loss_chart = alt.Chart(history_df).transform_fold(
        ['loss', 'val_loss'],
        as_=['type', 'loss']
    ).mark_line().encode(
        x='epoch:Q',
        y='loss:Q',
        color='type:N',
        tooltip=['epoch', 'loss']
    ).properties(title='Training and Validation Loss')

    mo.hstack([accuracy_chart | loss_chart])
    return accuracy_chart, loss_chart


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
def __(accuracy_score, mo, y_pred, y_test):
    mo.md(f"Accuracy score: **{round(accuracy_score(y_test, y_pred) * 100, 2)}**%")
    return


@app.cell
def __(classification_report, mo, y_pred, y_test):
    with mo.redirect_stdout():
        print(classification_report(y_test, y_pred))
    return


@app.cell
def __(embedding_models, get_doc_embedding, loaded_model, preprocess_text):
    def try_predict(text):
      tokenized = preprocess_text(text)
      embedded = get_doc_embedding(tokenized, embedding_models['fasttext'])
      embedded = embedded.reshape(1, 1, -1)
      prediction = loaded_model.predict(embedded)
      predicted_class = prediction.argmax(axis=-1)
      predicted_class = "Progressive" if predicted_class == 1 else "Conservative"
      return predicted_class
    return (try_predict,)


@app.cell
def __():
    def validate(value):
        if len(value.split()) < 15:
            return 'Please enter more than 15 words.'
    return (validate,)


@app.cell
def __(mo, validate):
    form = mo.ui.text_area(placeholder="...").form(validate=validate)
    return (form,)


if __name__ == "__main__":
    app.run()
