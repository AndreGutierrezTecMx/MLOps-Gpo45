import os
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalysis:
    """Clase para obtener visualizaciones y resúmenes de un DataFrame limpio."""

    def __init__(self, dataframe):
        self.df = dataframe

    def export_plot(self, title: str, prefix: str = "plot"):
        """Guarda y cierra la figura actual con nombre basado en el título."""
        os.makedirs("output", exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{safe_title}_{timestamp}.png"
        filepath = os.path.join("output", filename)
        plt.savefig(filepath)
        plt.close()
        print(f"✅ Gráfico guardado: {filepath}")

    def get_channel_counts_and_shares(self):
        channel_cols = [col for col in self.df.columns if col.startswith("data_channel_is")]
        self.df[channel_cols] = self.df[channel_cols].apply(pd.to_numeric, errors="coerce")
        self.df["shares"] = pd.to_numeric(self.df["shares"], errors="coerce")

        counts = {col: self.df[self.df[col] == 1].shape[0] for col in channel_cols}
        no_channel_flag = (self.df[channel_cols].sum(axis=1) == 0)
        counts["no_channel"] = no_channel_flag.sum()
        counts = pd.Series(counts)

        shares = {col: self.df[self.df[col] == 1]["shares"].sum() for col in channel_cols}
        shares["no_channel"] = self.df[no_channel_flag]["shares"].sum()
        shares = pd.Series(shares)

        combined = pd.DataFrame({
            "channel": counts.index,
            "count": counts.values,
            "total_shares": shares.values
        })

        return combined.sort_values(by="count", ascending=False)

    def get_weekday_counts(self):
        weekday_cols = [col for col in self.df.columns if col.startswith("weekday_is")]
        counts = {col: self.df[self.df[col] == 1].shape[0] for col in weekday_cols}
        day_mapping = {
            "weekday_is_monday": "Monday",
            "weekday_is_tuesday": "Tuesday",
            "weekday_is_wednesday": "Wednesday",
            "weekday_is_thursday": "Thursday",
            "weekday_is_friday": "Friday",
            "weekday_is_saturday": "Saturday",
            "weekday_is_sunday": "Sunday"
        }
        counts = pd.Series(counts).rename(index=day_mapping)
        df_out = counts.reset_index()
        df_out.columns = ["day", "count"]
        return df_out

    def prepare_summary_dataframes(self):
        self.channel_df = self.get_channel_counts_and_shares()
        self.weekday_df = self.get_weekday_counts()
        self.channel_shares_df = self.channel_df[["channel", "total_shares"]].copy()

    def print_top_shared_articles(self, top_n=20):
        self.top_articles_df = self.df.sort_values(by="shares", ascending=False).head(top_n)
        print(self.top_articles_df[["url", "shares"]])

    def plot_bar_chart(self, df, x_col, y_col, title, xlabel=None, ylabel=None, rotate_xticks=True, annotate=True, palette='viridis'):
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

        self.export_plot(title, prefix="bar")

    def scatter_plot(self, x_col, y_col="shares", title=None, xlabel=None, ylabel="Compartidos"):
        x = pd.to_numeric(self.df[x_col], errors='coerce')
        y = pd.to_numeric(self.df[y_col], errors='coerce')

        title = str(title or f"Dispersión de {x_col} vs {y_col}")
        xlabel = str(xlabel or x_col)
        ylabel = str(ylabel or "Compartidos")

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        self.export_plot(title, prefix="scatter")

    # Histogramas
    def histogram(self, column, bins=50, kde=True, title=None, xlabel=None, ylabel="Frecuencia"):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], bins=bins, kde=kde)
        plt.title(title or f'Distribución de {column}')
        plt.xlabel(xlabel or column)
        plt.ylabel(ylabel)
        plt.grid(True)
        self.export_plot(title or column, prefix="hist")

    # Obtener gráficos con datos específicos del df
    def plot_bar_charts(self):
        self.prepare_summary_dataframes()
        self.plot_bar_chart(self.channel_df, "channel", "count", "Artículos por canal")
        self.plot_bar_chart(self.channel_shares_df, "channel", "total_shares", "Compartidos por canal")
        self.plot_bar_chart(self.weekday_df, "day", "count", "Artículos por día de la semana")

    def print_scatter_plot(self):
        self.scatter_plot("num_imgs", title="Imágenes vs Compartidos", xlabel="Número de Imágenes")
        self.scatter_plot("num_videos", title="Videos vs Compartidos", xlabel="Número de Videos")
        self.scatter_plot("num_hrefs", title="Referencias vs Compartidos", xlabel="Número de Referencias")
        self.scatter_plot("n_tokens_title", title="Tokens de Título vs Compartidos", xlabel="Tokens de Título")
        self.scatter_plot("n_tokens_content", title="Tokens de Contenido vs Compartidos", xlabel="Tokens de Contenido")

    def print_histograms(self):
        self.histogram("shares", title="Distribución de Compartidos", xlabel="Compartidos")