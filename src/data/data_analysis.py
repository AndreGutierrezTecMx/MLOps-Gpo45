import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalysis:
    """A class for get visualizations from cleaned dataframe."""
    def __init__(self, dataframe):
        self.df = dataframe

    def preview(self, n=5):
        """Muestra las primeras n filas del DataFrame."""
        return self.df.head(n)
    
    # Funciones para obtener columnas específicas a través de columnas binarias y conteo de artículos y compartidos por canal
    def get_channel_counts_and_shares(df):
        channel_cols = [col for col in df.columns if col.startswith('data_channel_is_')]

        # Conteo de artículos por canal
        counts = df[channel_cols].sum()
        no_channel_flag = (df[channel_cols].sum(axis=1) == 0).astype(int)
        counts['no_channel'] = no_channel_flag.sum()

        # Total de compartidos por canal
        shares = df[channel_cols].multiply(df['shares'], axis=0).sum()
        shares['no_channel'] = df[no_channel_flag == 1]['shares'].sum()

        # Combinar en un solo DataFrame
        combined = pd.DataFrame({
            'Channel': list(counts.index),
            'Count': counts.values,
            'Total Shares': shares.values
        })

        return combined.sort_values(by='Count', ascending=False)

    # Función para obtener conteo de artículos por día de la semana
    def get_weekday_counts(df):
        weekday_cols = [col for col in df.columns if col.startswith('weekday_is_')]
        counts = df[weekday_cols].sum()
        counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_out = counts.reset_index()
        df_out.columns = ['Day', 'Count']
        return df_out
    
    # Función para preparar dataframes resumen
    def prepare_summary_dataframes(df):
        channel_df = get_channel_counts_and_shares(df)
        weekday_df = get_weekday_counts(df)
        channel_shares_df = channel_df[['Channel', 'Total Shares']].copy()
        return channel_df[['Channel', 'Count']], weekday_df, channel_shares_df

    # Función para obtener los artículos más compartidos
    def get_top_shared_articles(self, top_n=20):
        self.top_articles_df = self.df.sort_values(by='shares', ascending=False).head(top_n)
        return self.top_articles_df[['url', 'shares']]
    
    # Funciones de visualización

    # Gráfico de barras genérico
    def plot_bar_chart(
        df,
        x_col,
        y_col,
        title,
        xlabel=None,
        ylabel=None,
        rotate_xticks=True,
        annotate=True,
        palette='viridis'
    ):
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=x_col, y=y_col, data=df, palette=palette, hue=x_col, legend=False)
        plt.title(title)
        plt.xlabel(xlabel or x_col)
        plt.ylabel(ylabel or y_col)
        if rotate_xticks:
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if annotate:
            for p in ax.patches:
                height = p.get_height()
                if not pd.isna(height):
                    ax.annotate(f'{height:,.0f}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='center',
                                xytext=(0, 5), textcoords='offset points')

        plt.show()

    # Diagrama de dispersión genérico
    def scatter_plot(self, x_col, y_col='shares', title=None, xlabel=None, ylabel='Compartidos'):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[x_col], self.df[y_col], alpha=0.5)
        plt.title(title or f'Dispersión de {x_col} vs. {y_col}')
        plt.xlabel(xlabel or x_col)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    # Histograma genérico
    def histogram(self, column, bins=50, kde=True, title=None, xlabel=None, ylabel='Frecuencia'):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], bins=bins, kde=kde)
        plt.title(title or f'Distribución de {column}')
        plt.xlabel(xlabel or column)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    # Uso de la clase DataAnalysis

    # Suponiendo que df_clean es el DataFrame limpio ya cargado y preprocesado
    DataAnalysis = DataAnalysis(df_clean)

    # Vista previa de los datos
    DataAnalysis.data_channel_cols(df_clean)
    print(DataAnalysis.preview())

    # Gráficos de barras
    channel_df, weekday_df, channel_shares_df = DataAnalysis.prepare_summary_dataframes(df_clean)
    plot_bar_chart(channel_df, 'Channel', 'Count', 'Artículos por canal')
    plot_bar_chart(channel_shares_df, 'Channel', 'Total Shares', 'Compartidos por canal')
    plot_bar_chart(weekday_df, 'Day', 'Count', 'Artículos por día de la semana')

    # Artículos más compartidos
    top_articles = DataAnalysis.get_top_shared_articles(top_n=20)
    print(top_articles)
                    
    # Diagramas de dispersión

    #Imágenenes vs Compartidos
    DataAnalysis.scatter_plot('num_imgs', title='Dispersión de Imágenes vs. Compartidos', xlabel='Número de Imágenes')

    #Videos vs Compartidos
    DataAnalysis.scatter_plot('num_videos', title='Dispersión de Videos vs. Compartidos', xlabel='Número de Videos')

    #Referencias vs Compartidos
    DataAnalysis.scatter_plot('num_hrefs', title='Dispersión de Referencias vs. Compartidos', xlabel='Número de Referencias')

    #Tokens vs Compartidos
    DataAnalysis.scatter_plot('n_tokens_title', title='Dispersión de Tokens de Título vs. Compartidos', xlabel='Tokens de Título')

    #Tokens de contenido vs Compartidos
    DataAnalysis.scatter_plot('n_tokens_content', title='Dispersión de Tokens de Contenido vs. Compartidos', xlabel='Tokens de Contenido')

    # Histogramas
    DataAnalysis.histogram('shares', title='Distribución de Compartidos', xlabel='Compartidos')



