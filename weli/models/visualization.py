"""
Visualisation des architectures de modèles Weli.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def _get_layer_color(layer_type: str) -> str:
    """
    Retourne une couleur en fonction du type de couche.
    
    Args:
        layer_type: Type de la couche
        
    Returns:
        Couleur HEX
    """
    color_map = {
        'Dense': '#4CAF50',           # Vert
        'Conv2D': '#2196F3',          # Bleu
        'Conv1D': '#2196F3',          # Bleu
        'Conv3D': '#2196F3',          # Bleu
        'MaxPool2D': '#FF9800',       # Orange
        'AveragePool2D': '#FF9800',   # Orange
        'Flatten': '#9C27B0',         # Violet
        'Dropout': '#F44336',         # Rouge
        'BatchNorm': '#795548',       # Marron
        'ReLU': '#FF5722',            # Rouge foncé
        'Sigmoid': '#E91E63',         # Rose
        'Tanh': '#9C27B0',            # Violet
        'Softmax': '#673AB7',         # Violet foncé
        'LSTM': '#009688',            # Turquoise
        'GRU': '#009688',             # Turquoise
        'RNN': '#009688',             # Turquoise
        'Embedding': '#00BCD4',       # Cyan
    }
    
    # Chercher des correspondances partielles
    for key, color in color_map.items():
        if key in layer_type:
            return color
    
    # Couleur par défaut
    return '#607D8B'  # Gris bleu

def _format_shape(shape: Tuple) -> str:
    """
    Formate une shape pour l'affichage.
    
    Args:
        shape: Tuple représentant la shape
        
    Returns:
        String formatée
    """
    if shape is None:
        return "?"
    
    if isinstance(shape, tuple):
        # Supprimer le batch_size si présent
        if len(shape) > 1:
            return str(shape)
        elif len(shape) == 1:
            return f"({shape[0]},)"
        else:
            return "()"
    
    return str(shape)

def plot_model_architecture(model: Any, 
                           figsize: Tuple[int, int] = (14, 10),
                           show_shapes: bool = True,
                           show_layer_names: bool = True,
                           show_parameters: bool = True,
                           rankdir: str = 'TB',
                           filename: Optional[str] = None,
                           dpi: int = 100) -> None:
    """
    Affiche une visualisation graphique de l'architecture du modèle.
    
    Args:
        model: Modèle Weli à visualiser
        figsize: Taille de la figure (largeur, hauteur)
        show_shapes: Afficher les shapes d'entrée/sortie
        show_layer_names: Afficher les noms des couches
        show_parameters: Afficher le nombre de paramètres
        rankdir: Direction du graphe ('TB' = haut-bas, 'LR' = gauche-droite)
        filename: Si spécifié, sauvegarde la figure
        dpi: Résolution pour la sauvegarde
    """
    if not hasattr(model, 'layers') or not model.layers:
        print("Model has no layers to visualize")
        return
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(f"Architecture du modèle: {model.name}", fontsize=16, fontweight='bold', pad=20)
    
    # Configurer l'affichage en fonction de la direction
    is_vertical = (rankdir == 'TB')
    
    # Déterminer l'espacement
    n_layers = len(model.layers)
    if is_vertical:
        layer_spacing = 0.8 / max(n_layers, 1)
        start_y = 0.85
    else:
        layer_spacing = 0.8 / max(n_layers, 1)
        start_x = 0.1
    
    # Collecter les informations sur les paramètres
    total_params = 0
    layer_params = []
    
    for layer in model.layers:
        layer_param_count = 0
        if hasattr(layer, 'parameters'):
            for param in layer.parameters.values():
                layer_param_count += param.size
        layer_params.append(layer_param_count)
        total_params += layer_param_count
    
    # Dessiner chaque couche
    for i, (layer, param_count) in enumerate(zip(model.layers, layer_params)):
        if is_vertical:
            # Layout vertical
            x_center = 0.5
            y_center = start_y - i * layer_spacing
            
            # Dimensions du rectangle
            rect_width = 0.4
            rect_height = layer_spacing * 0.7
            
            # Position du rectangle
            rect_x = x_center - rect_width / 2
            rect_y = y_center - rect_height / 2
        else:
            # Layout horizontal
            y_center = 0.5
            x_center = start_x + i * layer_spacing
            
            # Dimensions du rectangle
            rect_width = layer_spacing * 0.7
            rect_height = 0.4
            
            # Position du rectangle
            rect_x = x_center - rect_width / 2
            rect_y = y_center - rect_height / 2
        
        # Obtenir la couleur de la couche
        layer_type = layer.__class__.__name__
        layer_color = _get_layer_color(layer_type)
        
        # Dessiner le rectangle de la couche
        rect = patches.Rectangle(
            (rect_x, rect_y), rect_width, rect_height,
            linewidth=2, edgecolor='black',
            facecolor=layer_color, alpha=0.8,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Préparer le texte de la couche
        texts = []
        
        # Nom de la couche
        if show_layer_names:
            texts.append(f"{layer.name}")
        
        # Type de couche
        texts.append(f"({layer_type})")
        
        # Shape de sortie si disponible
        if show_shapes and hasattr(layer, 'output_shape'):
            output_shape = _format_shape(layer.output_shape)
            texts.append(f"Shape: {output_shape}")
        
        # Nombre de paramètres
        if show_parameters and param_count > 0:
            texts.append(f"Params: {param_count:,}")
        
        # Positionner le texte
        text_y_offset = 0.08 if is_vertical else 0.05
        for j, text_line in enumerate(texts):
            text_y = y_center - (len(texts) - 1) * text_y_offset / 2 + j * text_y_offset
            
            ax.text(
                x_center, text_y, text_line,
                ha='center', va='center',
                fontsize=9 if len(text_line) < 30 else 8,
                fontweight='bold' if j == 0 else 'normal',
                color='white' if j == 0 else 'lightyellow',
                zorder=3
            )
        
        # Dessiner les flèches de connexion
        if i > 0:
            if is_vertical:
                # Flèche verticale
                arrow_start = (x_center, y_center + rect_height / 2 + 0.02)
                arrow_end = (x_center, y_center + rect_height / 2 + layer_spacing * 0.3)
                
                ax.annotate(
                    '', xy=arrow_start, xytext=arrow_end,
                    arrowprops=dict(
                        arrowstyle='->',
                        color='gray',
                        linewidth=1.5,
                        connectionstyle='arc3,rad=0',
                        alpha=0.7
                    ),
                    zorder=1
                )
            else:
                # Flèche horizontale
                arrow_start = (x_center - rect_width / 2 - 0.02, y_center)
                arrow_end = (x_center - rect_width / 2 - layer_spacing * 0.3, y_center)
                
                ax.annotate(
                    '', xy=arrow_start, xytext=arrow_end,
                    arrowprops=dict(
                        arrowstyle='->',
                        color='gray',
                        linewidth=1.5,
                        connectionstyle='arc3,rad=0',
                        alpha=0.7
                    ),
                    zorder=1
                )
    
    # Ajouter une légende
    legend_elements = []
    unique_layer_types = set(layer.__class__.__name__ for layer in model.layers)
    
    for layer_type in sorted(unique_layer_types):
        color = _get_layer_color(layer_type)
        legend_elements.append(
            patches.Patch(facecolor=color, edgecolor='black', label=layer_type)
        )
    
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(5, len(legend_elements)),
            fontsize=9
        )
    
    # Informations en bas du graphe
    info_text = f"Total layers: {n_layers}"
    if show_parameters:
        info_text += f" | Total parameters: {total_params:,}"
    
    if hasattr(model, '_input_shape') and model._input_shape:
        input_shape = _format_shape(model._input_shape)
        info_text += f" | Input shape: {input_shape}"
    
    if hasattr(model, '_output_shape') and model._output_shape:
        output_shape = _format_shape(model._output_shape)
        info_text += f" | Output shape: {output_shape}"
    
    ax.text(
        0.5, 0.02, info_text,
        ha='center', va='center',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
        transform=ax.transAxes
    )
    
    # Configuration des axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"✓ Diagramme sauvegardé: {filename}")
    
    plt.show()

def plot_training_history(history: Dict[str, List[float]], 
                         figsize: Tuple[int, int] = (12, 8),
                         filename: Optional[str] = None) -> None:
    """
    Affiche les courbes d'apprentissage du modèle.
    
    Args:
        history: Historique d'entraînement (dict avec 'train_loss', 'val_loss', etc.)
        figsize: Taille de la figure
        filename: Si spécifié, sauvegarde la figure
    """
    if not history:
        print("No training history to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Plot des losses
    ax1 = axes[0]
    
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot des accuracy
    ax2 = axes[1]
    
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"✓ Training history plot saved: {filename}")
    
    plt.show()

def plot_layer_activations(model: Any, 
                          input_data: np.ndarray,
                          layer_indices: Optional[List[int]] = None,
                          figsize: Tuple[int, int] = (15, 10),
                          cmap: str = 'viridis',
                          filename: Optional[str] = None) -> None:
    """
    Visualise les activations des différentes couches du modèle.
    
    Args:
        model: Modèle Weli
        input_data: Données d'input pour générer les activations
        layer_indices: Indices des couches à visualiser (si None, toutes)
        figsize: Taille de la figure
        cmap: Colormap pour les heatmaps
        filename: Si spécifié, sauvegarde la figure
    """
    if not hasattr(model, 'layers'):
        print("Model has no layers")
        return
    
    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))
    
    # Filtrer les couches valides
    valid_indices = [i for i in layer_indices if 0 <= i < len(model.layers)]
    n_layers = len(valid_indices)
    
    if n_layers == 0:
        print("No valid layers to visualize")
        return
    
    # Calculer la disposition de la grille
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Si une seule ligne/colonne, convertir axes en array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f"Layer Activations for Input Shape: {input_data.shape}", 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Propagation avant pour obtenir les activations
    activations = []
    current_output = input_data
    
    for i, layer in enumerate(model.layers):
        current_output = layer.forward(current_output)
        if i in valid_indices:
            activations.append((i, layer.name, current_output.copy()))
    
    # Plot chaque activation
    for idx, (ax, (layer_idx, layer_name, activation)) in enumerate(zip(axes.flat, activations)):
        # Aplatir si nécessaire pour la visualisation
        if activation.ndim > 2:
            # Pour les activations 3D+, prendre la moyenne sur certains axes
            if activation.ndim == 4:  # Conv2D output: (batch, height, width, channels)
                # Prendre la moyenne sur les canaux pour un batch
                viz_data = activation[0].mean(axis=-1)
            elif activation.ndim == 3:  # RNN output: (batch, timesteps, features)
                # Prendre la moyenne sur les features
                viz_data = activation[0].mean(axis=-1)
            else:
                viz_data = activation.flatten()
        else:
            # Pour les activations 2D, prendre le premier échantillon
            viz_data = activation[0] if activation.shape[0] > 1 else activation.flatten()
        
        # Créer le heatmap ou line plot
        if viz_data.ndim == 2:
            # Heatmap 2D
            im = ax.imshow(viz_data, cmap=cmap, aspect='auto')
            plt.colorbar(im, ax=ax)
        elif viz_data.ndim == 1:
            # Line plot 1D
            ax.plot(viz_data, linewidth=2)
            ax.fill_between(range(len(viz_data)), 0, viz_data, alpha=0.3)
            ax.set_ylim(viz_data.min() * 1.1, viz_data.max() * 1.1)
        
        ax.set_title(f"Layer {layer_idx}: {layer_name}\nShape: {activation.shape}", 
                    fontsize=10, pad=10)
        ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for idx in range(len(activations), n_rows * n_cols):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"✓ Layer activations plot saved: {filename}")
    
    plt.show()

def generate_model_summary_table(model: Any) -> str:
    """
    Génère un tableau texte résumé du modèle.
    
    Args:
        model: Modèle Weli
        
    Returns:
        String formaté du résumé
    """
    if not hasattr(model, 'layers'):
        return "Model has no layers"
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"MODEL SUMMARY: {model.name}")
    lines.append("=" * 80)
    lines.append(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<15} {'Connected to'}")
    lines.append("-" * 80)
    
    total_params = 0
    
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        layer_name = f"{layer.name} ({layer_type})"
        
        # Output shape
        output_shape = getattr(layer, 'output_shape', '?')
        output_shape_str = str(output_shape) if output_shape else '?'
        
        # Param count
        param_count = 0
        if hasattr(layer, 'parameters'):
            for param in layer.parameters.values():
                param_count += param.size
        total_params += param_count
        
        # Connected to (pour functional models)
        connected_to = "previous"
        if hasattr(layer, 'inputs') and layer.inputs:
            input_names = []
            for inp in layer.inputs:
                if hasattr(inp, 'name'):
                    input_names.append(inp.name)
                else:
                    input_names.append(str(inp))
            connected_to = ", ".join(input_names)
        
        lines.append(f"{layer_name:<30} {output_shape_str:<20} {param_count:<15,} {connected_to}")
    
    lines.append("=" * 80)
    
    # Informations générales
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {total_params:,}")
    lines.append(f"Non-trainable params: 0")
    
    if hasattr(model, '_input_shape') and model._input_shape:
        lines.append(f"Input shape: {model._input_shape}")
    
    if hasattr(model, '_output_shape') and model._output_shape:
        lines.append(f"Output shape: {model._output_shape}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

def visualize_weights_distribution(model: Any, 
                                 figsize: Tuple[int, int] = (12, 8),
                                 bins: int = 50,
                                 filename: Optional[str] = None) -> None:
    """
    Visualise la distribution des poids du modèle.
    
    Args:
        model: Modèle Weli
        figsize: Taille de la figure
        bins: Nombre de bins pour les histogrammes
        filename: Si spécifié, sauvegarde la figure
    """
    if not hasattr(model, 'layers'):
        print("Model has no layers")
        return
    
    # Collecter tous les poids
    all_weights = []
    weight_labels = []
    
    for layer in model.layers:
        if hasattr(layer, 'parameters'):
            for param_name, param in layer.parameters.items():
                if 'W' in param_name or 'weight' in param_name.lower():
                    all_weights.append(param.flatten())
                    weight_labels.append(f"{layer.name}.{param_name}")
    
    if not all_weights:
        print("No weights found in the model")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Weight Distributions - Model: {model.name}", 
                fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Histogramme combiné
    ax1 = axes[0, 0]
    for weights, label in zip(all_weights, weight_labels):
        ax1.hist(weights, bins=bins, alpha=0.5, label=label, density=True)
    
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Density')
    ax1.set_title('All Weights Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # 2. Statistiques par couche
    ax2 = axes[0, 1]
    
    stats_data = []
    for weights, label in zip(all_weights, weight_labels):
        stats = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights)
        }
        stats_data.append((label, stats))
    
    # Bar plot des std
    labels = [data[0] for data in stats_data]
    stds = [data[1]['std'] for data in stats_data]
    
    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, stds, alpha=0.7, color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Standard Deviation')
    ax2.set_title('Weight Standard Deviation per Layer')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Box plot
    ax3 = axes[1, 0]
    box_data = all_weights[:10]  # Limiter à 10 couches pour la lisibilité
    box_labels = weight_labels[:10]
    
    bp = ax3.boxplot(box_data, vert=False, patch_artist=True)
    
    # Colorer les boxplots
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_yticks(range(1, len(box_labels) + 1))
    ax3.set_yticklabels(box_labels, fontsize=8)
    ax3.set_xlabel('Weight Value')
    ax3.set_title('Weight Distribution (Box Plot)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Résumé statistique
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculer les statistiques globales
    all_weights_flat = np.concatenate(all_weights)
    global_stats = {
        'Total Parameters': sum(w.size for w in all_weights),
        'Global Mean': f"{np.mean(all_weights_flat):.6f}",
        'Global Std': f"{np.std(all_weights_flat):.6f}",
        'Global Min': f"{np.min(all_weights_flat):.6f}",
        'Global Max': f"{np.max(all_weights_flat):.6f}",
        'L2 Norm': f"{np.linalg.norm(all_weights_flat):.6f}"
    }
    
    # Afficher le texte
    stats_text = "Global Statistics:\n" + "\n".join(
        [f"{key}: {value}" for key, value in global_stats.items()]
    )
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, 
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"✓ Weight distribution plot saved: {filename}")
    
    plt.show()

def export_architecture_to_html(model: Any, filename: str = "model_architecture.html") -> None:
    """
    Exporte l'architecture du modèle en HTML interactif.
    
    Args:
        model: Modèle Weli
        filename: Nom du fichier HTML
    """
    if not hasattr(model, 'layers'):
        print("Model has no layers")
        return
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Architecture: {model_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            .layer {{
                margin: 20px 0;
                padding: 15px;
                border-left: 5px solid {color};
                background-color: #f9f9f9;
                transition: all 0.3s;
                cursor: pointer;
            }}
            .layer:hover {{
                transform: translateX(10px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .layer-header {{
                font-weight: bold;
                font-size: 16px;
                color: #333;
            }}
            .layer-details {{
                margin-top: 10px;
                padding: 10px;
                background-color: white;
                border-radius: 5px;
                display: none;
            }}
            .stats {{
                background-color: #e8f5e8;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .color-legend {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
            }}
            .color-item {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .color-box {{
                width: 20px;
                height: 20px;
                border-radius: 3px;
            }}
        </style>
        <script>
            function toggleDetails(id) {{
                var element = document.getElementById(id);
                if (element.style.display === "none") {{
                    element.style.display = "block";
                }} else {{
                    element.style.display = "none";
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>📊 Model Architecture: {model_name}</h1>
            
            <div class="stats">
                <h3>📈 Model Statistics</h3>
                <p><strong>Total Layers:</strong> {total_layers}</p>
                <p><strong>Total Parameters:</strong> {total_params:,}</p>
                <p><strong>Input Shape:</strong> {input_shape}</p>
                <p><strong>Output Shape:</strong> {output_shape}</p>
            </div>
            
            <h3>🎨 Layer Color Legend</h3>
            <div class="color-legend">
                {color_legend}
            </div>
            
            <h3>🏗️ Layers</h3>
            {layers_html}
        </div>
    </body>
    </html>
    """
    
    # Collecter les informations
    total_params = 0
    layers_info = []
    color_legend_html = ""
    color_map = {}
    
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        layer_color = _get_layer_color(layer_type)
        
        # Ajouter à la légende des couleurs
        if layer_type not in color_map:
            color_map[layer_type] = layer_color
            color_legend_html += f"""
            <div class="color-item">
                <div class="color-box" style="background-color: {layer_color};"></div>
                <span>{layer_type}</span>
            </div>
            """
        
        # Informations de la couche
        output_shape = getattr(layer, 'output_shape', '?')
        
        param_count = 0
        if hasattr(layer, 'parameters'):
            for param in layer.parameters.values():
                param_count += param.size
        total_params += param_count
        
        # Détails de la couche
        layer_details = f"""
        <div class="layer-details" id="details-{i}">
            <p><strong>Type:</strong> {layer_type}</p>
            <p><strong>Output Shape:</strong> {output_shape}</p>
            <p><strong>Parameters:</strong> {param_count:,}</p>
            <p><strong>Trainable:</strong> {layer.trainable if hasattr(layer, 'trainable') else 'Yes'}</p>
        </div>
        """
        
        layer_html = f"""
        <div class="layer" style="border-left-color: {layer_color};" onclick="toggleDetails('details-{i}')">
            <div class="layer-header">
                {i+1}. {layer.name} ({layer_type})
            </div>
            {layer_details}
        </div>
        """
        
        layers_info.append(layer_html)
    
    # Remplir le template
    html_content = html_template.format(
        model_name=model.name,
        total_layers=len(model.layers),
        total_params=total_params,
        input_shape=getattr(model, '_input_shape', '?'),
        output_shape=getattr(model, '_output_shape', '?'),
        color_legend=color_legend_html,
        layers_html="\n".join(layers_info)
    )
    
    # Écrire le fichier HTML
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML architecture exported: {filename}")
    print(f"  Open {filename} in your browser to view the interactive model architecture.")