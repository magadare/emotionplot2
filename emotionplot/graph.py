import plotly.graph_objects as go

def plot_stacked_emotions(df, group_size=5, exclude_neutral=True):
    """
    Plots a stacked bar chart of emotion scores from a DataFrame using Plotly.

    Parameters:
    - emotions_df (pd.DataFrame): A DataFrame containing emotion scores per chunk.
    - group_size (int): Number of chunks to group together for aggregation.
    - exclude_neutral (bool): Whether to exclude the "neutral" emotion from the plot.

    Returns:
    - None (shows an interactive plot)
    """
    # Select emotions to plot
    emotions_to_plot = [
        col for col in df.columns
     #   if not (exclude_neutral and col.lower() == "neutral")
    ]

    # Group emotion scores
    grouped = df[emotions_to_plot].groupby(df.index // group_size).sum()

    template_to_select = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    template_selected = "plotly_white"

    fig = go.Figure()

    default_visible = ["anger", "joy", "disapproval", "fear", "surprise", "curiosity", "sadness"]  # Emotions to show by default


    # Your custom emotion order (must match column names in emotions_df)
    custom_order = [
        "anger", "joy", "disapproval", "fear", "surprise", "curiosity", "sadness"
        'love',
        'gratitude',
        'pride',
        'relief',
        'amusement',
        'admiration',
        'approval',
        'excitement',
        'optimism',
        'caring',
        'desire',
        'realization',
        'confusion',
        'nervousness',
        'embarrassment',
        'annoyance',
        'disappointment',
        'remorse',
        'disgust',
        'grief',
        'neutral'
    ]


# Loop through custom order
for emotion in custom_order:
    if emotion in emotions_df.columns:
        fig.add_trace(
            go.Scatter(
                x=emotions_df.index,
                y=emotions_df[emotion],
                mode='lines',
                name=emotion,
                customdata=df["chunk"],
                hovertemplate=(
                    "<b>Chunk Index:</b> %{x}<br>" +
                    "<b>Emotion:</b> %{fullData.name}<br>" +
                    "<b>Emotion Score:</b> %{y:.2f}<br>" +
                    "<b>Text:</b> %{customdata}<extra></extra>"
                ),
                visible=True if emotion in default_visible else "legendonly"
            )
        )

    # Configure layout
    fig.update_layout(
        barmode='stack',
        title="Stacked Emotion Scores per Chunk",
        xaxis=dict(
            title="Chunk Index",
            rangeslider=dict(visible=True),
            type="linear"
        ),
        yaxis=dict(
            title="Emotion Score"
        ),
        height=600,
        legend_title="Emotion",
        dragmode="pan",
        template=template_selected
    )

    fig.show(config={"scrollZoom": True})
