import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Political Ideologies Analysis

        This analysis is based on huggingface dataset repository. <br>
        You can visit right [here](https://huggingface.co/datasets/JyotiNayak/political_ideologies)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt

    from gensim.models import Word2Vec
    from sklearn.manifold import TSNE

    import pygwalker as pyg

    mo.md('## Import all libraries needed')
    return TSNE, Word2Vec, alt, mo, np, pd, plt, pyg, sns


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
def __(pd):
    df = pd.concat(
        [pd.read_parquet(f'{name}.parquet') for name in ['train', 'val', 'test']], 
        axis=0,
    )

    df = df.drop('__index_level_0__', axis=1)
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
    df['label_text'] = df['label'].map(label_mapping_reversed)
    df['issue_type_text'] = df['issue_type'].map(issue_type_mapping_reversed)

    mo.md('Here are the mapped label and issue type dataframe.')
    return


@app.cell
def __(df):
    df.head(7)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## Barplot of Libral Vs. Conservative Ideologies Proportion""")
    return


@app.cell
def __(alt, df, mo):
    labels_grouped = df['label_text'].value_counts().rename_axis('label_text').reset_index(name='counts')
    mo.ui.altair_chart(
        alt.Chart(labels_grouped).mark_bar(
            fill='#4C78A8',
            cursor='pointer',
        ).encode(
            x=alt.X('label_text', axis=alt.Axis(labelAngle=0)),
            y='counts:Q'
        )
    )
    return (labels_grouped,)


@app.cell
def __(mo):
    mo.md("""## Barplot of Number of Issue Types""")
    return


@app.cell
def __(alt, df, mo):
    issue_types_grouped = (
        df["issue_type_text"]
        .value_counts()
        .rename_axis("issue_type_text")
        .reset_index(name="counts")
    )
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
    return (issue_types_grouped,)


@app.cell
def __(mo):
    mo.md("""# Word Embedding using Word2Vec""")
    return


@app.cell
def __(Word2Vec, df):
    df['tokens'] = df['statement'].apply(lambda x: x.lower().split())
    word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, seed=0)
    return (word2vec_model,)


@app.cell
def __(np, word2vec_model):
    def get_doc_embedding(tokens):
        vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)
    return (get_doc_embedding,)


@app.cell
def __(df, get_doc_embedding, np):
    df['embedding'] = df['tokens'].apply(get_doc_embedding)
    embeddings_matrix = np.vstack(df['embedding'].values)
    return (embeddings_matrix,)


@app.cell
def __(TSNE, df, embeddings_matrix):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings_matrix)
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]
    return tsne, tsne_results


@app.cell
def __(df, plt, sns):
    # Step 5: Plot the embeddings with ideology as the color
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='issue_type_text', palette='Set1', s=100)
    plt.title("2D Visualization of Text Data by Ideology (Word2Vec Embeddings)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title='Ideology')
    plt.show()
    return


@app.cell
def __(df):
    df.columns
    return


@app.cell
def __(alt):
    # Brush for selection
    brush = alt.selection_interval()
    return (brush,)


@app.cell
def __(alt, brush, df, mo):
    points1 = alt.Chart(df, height=300, width=300).mark_point().encode(
        x='x:Q',
        y='y:Q',
        color=alt.condition(brush, 'label_text', alt.value('grey')),
        tooltip=['x:Q', 'y:Q', 'statement:N', 'label_text:N'] 
    ).add_params(brush).properties(name='By Political Ideology')

    scatter_chart1 = mo.ui.altair_chart(points1)
    return points1, scatter_chart1


@app.cell
def __(alt, brush, df, mo):
    points2 = alt.Chart(df, height=300, width=300).mark_point().encode(
        x='x:Q',
        y='y:Q',
        color=alt.condition(brush, 'issue_type_text', alt.value('grey')),
        tooltip=['x:Q', 'y:Q', 'statement:N', 'issue_type:N'] 
    ).add_params(brush).properties(title='By Issue Types')

    scatter_chart2 = mo.ui.altair_chart(points2)
    return points2, scatter_chart2


@app.cell
def __(scatter_chart1, scatter_chart2):
    combined_chart = (scatter_chart1 | scatter_chart2)
    combined_chart
    return (combined_chart,)


@app.cell(hide_code=True)
def __(combined_chart):
    combined_chart.value[['statement', 'label_text', 'issue_type_text']]
    return


@app.cell
def __(combined_chart):
    combined_chart.value['statement']
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
