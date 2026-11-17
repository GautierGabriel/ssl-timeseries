import os
import json
import logging
import pandas as pd
import numpy as np
import umap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class TrainingVisualizer:

    @staticmethod
    def save_loss_curve(json_path: str, output_dir: str) -> None:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        train_loss = data.get("train_loss", [])
        val_loss = data.get("val_loss", [])

        if not train_loss:
            logger.warning(f"No training loss found in '{json_path}'")
            return

        epochs = list(range(1, len(train_loss) + 1))
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss, mode="lines+markers", name="Train Loss",
            line=dict(color="#1f77b4", width=2), marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=val_loss, mode="lines+markers", name="Validation Loss",
            line=dict(color="#9467bd", width=2), marker=dict(size=4)
        ))

        fig.update_layout(
            title={"text": "Model Training Dynamics", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            xaxis_title="Epochs", yaxis_title="Loss", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)", dtick=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")

        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, "loss_curve.html")
        fig.write_html(html_path)

        try:
            image_path = os.path.join(output_dir, "loss_curve.png")
            fig.write_image(image_path, scale=2)
        except Exception as e:
            logger.warning(f"Could not generate PNG loss curve: {e}")


class LatentSpaceVisualizer:
    @staticmethod
    def plot_umap_clusters(
            embeddings_np: np.ndarray = None,
            cluster_labels: np.ndarray = None,
            metadata: list = None,
            title: str = "Espace Latent UMAP",
            emb_2d: np.ndarray = None
    ) -> go.Figure:

        if emb_2d is None:
            reducer = umap.UMAP(n_components=2, random_state=42)
            emb_2d = reducer.fit_transform(embeddings_np)

        df_dict = {
            "UMAP 1": emb_2d[:, 0],
            "UMAP 2": emb_2d[:, 1],
            "Cluster": [str(c) for c in cluster_labels] if cluster_labels is not None else ["N/A"] * len(emb_2d)
        }

        hover_cols = ["Cluster"]
        columns_to_plot = ["Cluster"]

        if metadata and len(metadata) > 0:
            flat_metadata = []
            for m in metadata:
                trial_meta = m.get("trial_metadata", {})

                flat_m = {
                    "Trial Name": trial_meta.get("name"),
                    "Channel": m.get("channel_id"),
                    "Date": m.get("timestamp"),
                    "Client": trial_meta.get("client", {}).get("name") if isinstance(trial_meta.get("client"),
                                                                                     dict) else None,
                    "Project": trial_meta.get("project", {}).get("name") if isinstance(trial_meta.get("project"),
                                                                                       dict) else None,
                    "Plant Type": trial_meta.get("plantTypeDto", {}).get("name") if isinstance(
                        trial_meta.get("plantTypeDto"), dict) else None,
                    "Stressor": trial_meta.get("stressorTypeDto", {}).get("name") if isinstance(
                        trial_meta.get("stressorTypeDto"), dict) else None,
                    "Substrate": trial_meta.get("substrateTypeDto", {}).get("name") if isinstance(
                        trial_meta.get("substrateTypeDto"), dict) else None,
                    "Chamber": trial_meta.get("chamberTypeDto", {}).get("name") if isinstance(
                        trial_meta.get("chamberTypeDto"), dict) else None,
                    "Resilient Crops": trial_meta.get("isResilientCrops"),
                    "EPCT": trial_meta.get("isEpct")
                }

                clean_m = {k: (str(v) if v is not None else "N/A") for k, v in flat_m.items()}
                flat_metadata.append(clean_m)
            valid_keys = [k for k in flat_metadata[0].keys() if any(m[k] != "N/A" for m in flat_metadata)]

            for key in valid_keys:
                df_dict[key] = [m[key] for m in flat_metadata]
                hover_cols.append(key)
                columns_to_plot.append(key)

            if "Trial Name" in valid_keys:
                intersection_col_name = "Trial x Cluster"
                df_dict[intersection_col_name] = [
                    f"{trial} - C:{cluster}"
                    for trial, cluster in zip(df_dict["Trial Name"], df_dict["Cluster"])
                ]
                hover_cols.append(intersection_col_name)
                columns_to_plot.append(intersection_col_name)

        df = pd.DataFrame(df_dict)

        figs_data = []
        trace_counts = []

        for col in columns_to_plot:
            temp_fig = px.scatter(df, x="UMAP 1", y="UMAP 2", color=col, hover_data=hover_cols)
            figs_data.extend(temp_fig.data)
            trace_counts.append(len(temp_fig.data))

        fig = go.Figure(data=figs_data)

        buttons = []
        current_trace_idx = 0

        for i, col in enumerate(columns_to_plot):
            visibility = [False] * len(figs_data)

            for j in range(trace_counts[i]):
                visibility[current_trace_idx + j] = True

            current_trace_idx += trace_counts[i]

            button = dict(
                label=f"Colorer par : {col}",
                method="update",
                args=[{"visible": visibility},
                      {"title": f"{title} (Couleur : {col})"}],
            )
            buttons.append(button)

        for i, trace in enumerate(fig.data):
            trace.visible = (i < trace_counts[0])

        layout_updates = {"title": f"{title} (Couleur : {columns_to_plot[0]})"}

        if len(columns_to_plot) > 1:
            layout_updates["updatemenus"] = [dict(
                active=0,
                buttons=buttons,
                x=1.0,
                y=1.1,
                xanchor="right",
                yanchor="bottom"
            )]

        fig.update_layout(**layout_updates)
        fig.update_traces(marker=dict(size=5, opacity=0.8))

        return fig

class DownstreamTaskVisualizer:

    @staticmethod
    def plot_stress_probability_dynamics(
            df_trial: pd.DataFrame,
            trial_name: str,
            stress_start_time: pd.Timestamp = None,
            output_html_path: str = None
    ) -> go.Figure:
        """
        Génère une courbe temporelle des probabilités moyennes avec une zone de stress.
        """
        fig = go.Figure()

        # Courbe du groupe Contrôle
        if "prob_control" in df_trial.columns:
            fig.add_trace(go.Scatter(
                x=df_trial.index, y=df_trial["prob_control"],
                mode='lines', name='Groupe Contrôle',
                line=dict(color='royalblue', width=2.5)
            ))

        # Courbe du groupe Stressé
        if "prob_stress" in df_trial.columns:
            fig.add_trace(go.Scatter(
                x=df_trial.index, y=df_trial["prob_stress"],
                mode='lines', name='Groupe Stressé',
                line=dict(color='crimson', width=2.5)
            ))

        # Zone ombrée définissant le début du stress (si applicable)
        if stress_start_time is not None:
            fig.add_vrect(
                x0=stress_start_time, x1=df_trial.index.max(),
                fillcolor="red", opacity=0.1, layer="below",
                line_width=1.5, line_dash="dash", line_color="red",
                annotation_text=" Début du Stress", annotation_position="top left",
                annotation_font_color="red"
            )

        fig.update_layout(
            title=f"Dynamique d'Apprentissage du Stress - Essai : {trial_name}",
            xaxis_title="Date",
            yaxis_title="Probabilité Moyenne (Classe 'Stress')",
            yaxis=dict(range=[-0.05, 1.05]),  # Marge pour bien voir les extrêmes
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if output_html_path:
            fig.write_html(output_html_path)

        return fig

    @staticmethod
    def plot_optimization_heatmap(grid_data: dict, output_html_path: str = None, window_label: str = "1h") -> go.Figure:
        f1_matrix = np.array(grid_data.get("f1_matrix", []))
        tp_matrix = np.array(grid_data.get("tp_matrix", []))
        tn_matrix = np.array(grid_data.get("tn_matrix", []))
        fp_matrix = np.array(grid_data.get("fp_matrix", []))
        fn_matrix = np.array(grid_data.get("fn_matrix", []))

        thresholds = grid_data.get("thresholds", [])
        percentiles = grid_data.get("percentiles", [])
        best = grid_data.get("best_params", {})

        customdata = np.stack((tp_matrix, tn_matrix, fp_matrix, fn_matrix), axis=-1)

        fig = go.Figure(data=go.Heatmap(
            z=f1_matrix,
            x=thresholds,
            y=percentiles,
            colorscale='Viridis',
            colorbar=dict(title='F1-Score'),
            customdata=customdata,
            hovertemplate=(
                'Threshold: %{x:.3f}<br>'
                'Percentile: %{y}%<br>'
                'F1-Score: %{z:.3f}<br>'
                '---<br>'
                'TP: %{customdata[0]} | FN: %{customdata[3]}<br>'
                'FP: %{customdata[2]} | TN: %{customdata[1]}<extra></extra>'
            )
        ))

        title = f"Optimization Grid: F1-Score (Threshold vs Percentile)"
        if best:
            title += f"<br><sup>Optimum: P={best.get('percentile')}%, T={best.get('threshold'):.3f} (F1={best.get('f1_score'):.3f} | TP:{best.get('tp')} FP:{best.get('fp')})</sup>"

        fig.update_layout(
            title=title,
            xaxis_title='Threshold',
            yaxis_title=f'Percentile ({window_label} Macro-Window)',
            width=900,
            height=600,
            template="plotly_white"
        )

        if output_html_path:
            fig.write_html(output_html_path)

        return fig
    
    @staticmethod
    def plot_decision_thresholds(curves_data: dict, output_html_path: str = None) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=curves_data["thresholds"], y=curves_data["f1_scores"],
            mode='lines', name='F1-Score', line=dict(color='green', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=curves_data["thresholds"], y=curves_data["false_negatives"],
            mode='lines', name='FN', line=dict(color='crimson', width=2, dash='dash'),
            yaxis='y2'
        ))

        fig.add_trace(go.Scatter(
            x=curves_data["thresholds"], y=curves_data["false_positives"],
            mode='lines', name='FP', line=dict(color='royalblue', width=2, dash='dot'),
            yaxis='y2'
        ))

        fig.update_layout(
            title=f"Threshold analysis (Total Positives: {curves_data.get('total_positives', 'N/A')})",
            xaxis=dict(title="Threshold", tickformat=".2f"),
            yaxis=dict(title="Score (F1)", side="left", range=[0, 1]),
            yaxis2=dict(title="Errors (FN / FP)", side="right", overlaying="y", showgrid=False),
            legend=dict(x=0.02, y=0.02, bgcolor="rgba(255, 255, 255, 0.8)"),
            hovermode="x unified"
        )

        if output_html_path:
            fig.write_html(output_html_path)
        return fig

    @staticmethod
    def plot_roc_curve(curves_data: dict, roc_auc: float, output_html_path: str = None) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=curves_data["fpr"], y=curves_data["tpr"],
            mode='lines', name=f'ROC Curve (AUC = {roc_auc:.4f})',
            line=dict(color='darkorange', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Random choice (AUC = 0.50)',
            line=dict(color='navy', width=2, dash='dash')
        ))

        fig.update_layout(
            title="ROC curve", xaxis=dict(title="FPR", range=[0, 1]),
            yaxis=dict(title="TPR", range=[0, 1]), width=700, height=600,
            legend=dict(x=0.55, y=0.05, bgcolor="rgba(255, 255, 255, 0.8)")
        )

        if output_html_path:
            fig.write_html(output_html_path)
        return fig

    def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list,
                              title: str = "Matrice de Confusion",
                              output_html_path: str = None) -> go.Figure:
        """
        Génère une matrice de confusion interactive.
        """

        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Classe Prédite", y="Classe Réelle", color="Nombre"),
            x=class_names,
            y=class_names,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto"
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis=dict(side="bottom"),
            width=600,
            height=600,
            coloraxis_showscale=False
        )

        if output_html_path:
            fig.write_html(output_html_path)

        return fig


class AnomalyVisualizer:

    @staticmethod
    def plot_signal_and_anomaly(df_signal: pd.DataFrame, df_ml: pd.DataFrame,
                                plant_id: str, output_html_path: str = None) -> go.Figure:

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df_signal.index,
                y=df_signal["mV"],
                name=f"Signal {plant_id} (mV)",
                mode='lines',
                line=dict(color='rgba(44, 160, 44, 0.5)'),
            ),
            secondary_y=False
        )

        if "timestamp" in df_ml.columns and "score" in df_ml.columns:
            df_ml['timestamp'] = pd.to_datetime(df_ml['timestamp'])

            fig.add_trace(
                go.Scatter(
                    x=df_ml['timestamp'],
                    y=df_ml['score'],
                    name="Distance",
                    mode='lines',
                    line=dict(color='crimson', width=2, dash='dot'),
                ),
                secondary_y=True
            )

        fig.update_layout(
            title=f"Distance - Plant: {plant_id}",
            xaxis_title="Date",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.update_yaxes(title_text="Potential (mV)", secondary_y=False)
        fig.update_yaxes(title_text="Distance", secondary_y=True, showgrid=False)

        if output_html_path:
            fig.write_html(output_html_path)

        return fig

    @staticmethod
    def plot_signal_with_heatmap(df_signal: pd.DataFrame, df_ml: pd.DataFrame,
                                  plant_id: str, output_html_path: str = None) -> go.Figure:

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.05,
            subplot_titles=(f"Signal brut - {plant_id}", "Score d'anomalie")
        )

        fig.add_trace(
            go.Scatter(
                x=df_signal.index,
                y=df_signal["mV"],
                name=f"Signal {plant_id} (mV)",
                mode='lines',
                line=dict(color='rgba(44, 160, 44, 0.7)', width=1),
            ),
            row=1, col=1
        )

        df_ml = df_ml.copy()
        df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"])
        df_ml = df_ml.sort_values("timestamp")

        fig.add_trace(
            go.Heatmap(
                x=df_ml["timestamp"],
                y=[str(plant_id)],
                z=[df_ml["score"].tolist()],
                colorscale="YlOrRd_r",
                showscale=True,
                colorbar=dict(title="Score", len=0.3, y=0.12)
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f"Rapport signal + anomalie - Plante {plant_id}",
            hovermode="x unified",
            height=600,
            showlegend=True
        )
        fig.update_yaxes(title_text="Potentiel (mV)", row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        if output_html_path:
            fig.write_html(output_html_path)

        return fig
