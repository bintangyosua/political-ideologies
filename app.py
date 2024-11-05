import marimo

__generated_with = "0.9.14"
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

    mo.md("""
    ## 1. Import all libraries needed

    The initial cells import the necessary libraries for data handling, visualization, and word embedding.
    """)
    return TSNE, Word2Vec, alt, mo, np, pd, plt, sns


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(issue_type_mapping, label_mapping):
    label_mapping_reversed = {v: k for k, v in label_mapping.items()}
    issue_type_mapping_reversed = {v: k for k, v in issue_type_mapping.items()}

    print(label_mapping_reversed)
    print(issue_type_mapping_reversed)
    return issue_type_mapping_reversed, label_mapping_reversed


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
        """
        ## 5. Word Embedding with Word2Vec

        Using Word2Vec, word embeddings are created from text statements in the dataset. The model trains on tokenized sentences, generating a 100-dimensional embedding for each word. Statements are averaged to form document-level embeddings.
        """
    )
    return


@app.cell(hide_code=True)
def __(Word2Vec, df):
    df['tokens'] = df['statement'].apply(lambda x: x.lower().split())
    word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, seed=0)
    return (word2vec_model,)


@app.cell(hide_code=True)
def __(np, word2vec_model):
    def get_doc_embedding(tokens):
        vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)
    return (get_doc_embedding,)


@app.cell(hide_code=True)
def __(df, get_doc_embedding, np):
    df['embedding'] = df['tokens'].apply(get_doc_embedding)
    embeddings_matrix = np.vstack(df['embedding'].values)
    return (embeddings_matrix,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## 6. Dimensionality Reduction with TSNE

        Embeddings are projected into a 2D space using TSNE for visualization. The embeddings are colored by issue type, showing clusters of similar statements.
        """
    )
    return


@app.cell(hide_code=True)
def __(TSNE, alt, df, embeddings_matrix, plt, sns):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings_matrix)
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]

    # Brush for selection
    brush = alt.selection_interval()
    size = 350

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='issue_type_text', palette='Set1', s=100)
    plt.title("2D Visualization of Text Data by Ideology (Word2Vec Embeddings)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title='Ideology')
    plt.show()
    return brush, size, tsne, tsne_results


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
def __(alt, brush, df, mo, size):
    points1 = alt.Chart(df, height=size, width=size).mark_point().encode(
        x='x:Q',
        y='y:Q',
        color=alt.condition(brush, 'label_text', alt.value('grey')),
        tooltip=['x:Q', 'y:Q', 'statement:N', 'label_text:N'] 
    ).add_params(brush).properties(title='By Political Ideologies')

    scatter_chart1 = mo.ui.altair_chart(points1)

    points2 = alt.Chart(df, height=size, width=size).mark_point().encode(
        x='x:Q',
        y='y:Q',
        color=alt.condition(brush, 'issue_type_text', alt.value('grey')),
        tooltip=['x:Q', 'y:Q', 'statement:N', 'issue_type:N'] 
    ).add_params(brush).properties(title='By Issue Types')

    scatter_chart2 = mo.ui.altair_chart(points2)

    combined_chart = (scatter_chart1 | scatter_chart2)
    combined_chart
    return combined_chart, points1, points2, scatter_chart1, scatter_chart2


@app.cell(hide_code=True)
def __(combined_chart):
    combined_chart.value[['statement', 'label_text', 'issue_type_text']]
    return


@app.cell(hide_code=True)
def __(combined_chart):
    combined_chart.value['statement']
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Data Insights

        - Ideology Distribution: Visualizes proportions of conservative and liberal ideologies.
        - Issue Types: Bar plot reveals the diversity and frequency of issue types in the dataset.
        - Word Embeddings: Using TSNE for 2D projections helps identify clusters in political statements.
        - Interactive Exploration: Offers detailed, interactive views on ideology vs. issue type distribution.

        This code provides a thorough analysis pipeline, from data loading to interactive visualizations, enabling an in-depth exploration of political ideologies.
        """
    )
    return


if __name__ == "__main__":
    app.run()
