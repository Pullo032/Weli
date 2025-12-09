"""
Visualisation des architectures de modèles.
"""
from typing import Dict, Any
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_model_architecture(model: 'Model', 
                           figsize: tuple = (12, 8),
                           show_shapes: bool = True,
                           show_dtype: bool = False,
                           show_layer_names: bool = True,
                           rankdir: str = 'TB',
                           filename: str = None):
    """
    Affiche une visualisation de l'architecture du modèle.
    
    Args:
        model: Modèle à visualiser
        figsize: Taille de la figure
        show_shapes: Afficher les shapes
        show_dtype: Afficher les types de données
        show_layer_names: Afficher les noms des couches
        rankdir: Direction du graphe ('TB' = top-bottom, 'LR' = left-right)
        filename: Si spécifié, sauvegarde la figure
    """
    print(f"Plotting model architecture: {model.name}")
    print("(Note: Full visualization requires graphviz)")
    
    # Version simplifiée avec matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Model: {model.name}")
    
    # Positionner les couches
    n_layers = len(model.layers)
    layer_height = 0.8 / n_layers if n_layers > 0 else 0.1
    
    for i, layer in enumerate(model.layers):
        y_pos = 0.9 - i * 0.1
        
        # Rectangle pour la couche
        rect = patches.Rectangle((0.1, y_pos - layer_height/2), 
                                0.8, layer_height,
                                linewidth=2, edgecolor='blue',
                                facecolor='lightblue', alpha=0.7)
        ax.add_patch(rect)
        
        # Texte
        layer_name = f"{layer.name}\n({layer.__class__.__name__})"
        if show_shapes and hasattr(layer, 'output_shape'):
            layer_name += f"\n{layer.output_shape}"
        
        ax.text(0.5, y_pos, layer_name,
                ha='center', va='center',
                fontsize=9, fontweight='bold')
        
        # Flèches entre les couches
        if i > 0:
            ax.arrow(0.5, y_pos + layer_height/2 + 0.05,
                    0, -0.07,
                    head_width=0.02, head_length=0.02,
                    fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()

def generate_model_diagram_dot(model: 'Model', 
                              filename: str = "model_diagram.dot"):
    """
    Génère un fichier DOT pour Graphviz.
    
    Args:
        model: Modèle
        filename: Nom du fichier DOT
    """
    dot_lines = [
        "digraph G {",
        f'  label="Model: {model.name}";',
        "  rankdir=TB;",
        "  node [shape=box, style=filled, fillcolor=lightblue];",
        ""
    ]
    
    # Nœuds pour chaque couche
    for i, layer in enumerate(model.layers):
        node_name = f"layer_{i}"
        label = f"{layer.name}\\n({layer.__class__.__name__})"
        
        if hasattr(layer, 'output_shape'):
            label += f"\\n{layer.output_shape}"
        
        dot_lines.append(f'  {node_name} [label="{label}"];')
    
    # Connexions entre les couches
    for i in range(len(model.layers) - 1):
        dot_lines.append(f"  layer_{i} -> layer_{i+1};")
    
    dot_lines.append("}")
    
    # Écrire le fichier
    with open(filename, 'w') as f:
        f.write("\n".join(dot_lines))
    
    print(f"DOT file generated: {filename}")
    print("To generate image: dot -Tpng model_diagram.dot -o model.png")

def print_model_ascii(model: 'Model'):
    """
    Affiche une représentation ASCII de l'architecture.
    """
    print(f"\n{'='*60}")
    print(f"MODEL: {model.name}")
    print(f"{'='*60}")
    
    max_name_length = max(len(layer.name) for layer in model.layers) if model.layers else 0
    max_type_length = max(len(layer.__class__.__name__) for layer in model.layers) if model.layers else 0
    
    for i, layer in enumerate(model.layers):
        # Formatter les informations
        layer_info = f"{layer.name:<{max_name_length}} "
        layer_info += f"({layer.__class__.__name__:<{max_type_length}})"
        
        if hasattr(layer, 'output_shape'):
            layer_info += f" → {layer.output_shape}"
        
        # Ligne avec flèche
        arrow = "↓" if i < len(model.layers) - 1 else "✓"
        print(f"  {layer_info}")
        if i < len(model.layers) - 1:
            print(f"  {' ' * (max_name_length + max_type_length + 4)} {arrow}")
    
    print(f"{'='*60}")
    params = sum(p.size for p in model.get_parameters().values())
    print(f"Total parameters: {params:,}")
    print(f"{'='*60}")