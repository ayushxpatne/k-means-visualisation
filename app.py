from flask import Flask, render_template, request, jsonify, session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import io
import base64
from matplotlib.patches import FancyArrowPatch, Ellipse

app = Flask(__name__)
app.secret_key = 'kmeans_secret_key_2024'

def get_euclidean_distance(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def generate_points(n, x_max=20, y_max=20):
    """Generate random points"""
    x_arr = [random.randint(0, x_max) for _ in range(n)]
    y_arr = [random.randint(0, y_max) for _ in range(n)]
    return x_arr, y_arr

def initialize_centroids(k, x_max=20, y_max=20):
    """Initialize random centroids"""
    x_c = [random.uniform(0, x_max) for _ in range(k)]
    y_c = [random.uniform(0, y_max) for _ in range(k)]
    return x_c, y_c

def assign_centroids(x_arr, y_arr, centroids_data, k):
    """Assign points to nearest centroid"""
    for j in range(k):
        centroids_data[f"centroid_{j}_points"] = []
    
    for i in range(len(x_arr)):
        x1, y1 = x_arr[i], y_arr[i]
        
        min_dist = float('inf')
        nearest_centroid = 0
        
        for j in range(k):
            x2, y2 = centroids_data[f"centroid_{j}_xy"]
            dist = get_euclidean_distance(x1, x2, y1, y2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_centroid = j
        
        centroids_data[f"centroid_{nearest_centroid}_points"].append([x1, y1])

def recalculate_centroids(centroids_data, k):
    """Recalculate centroid positions based on assigned points"""
    for j in range(k):
        points = centroids_data[f"centroid_{j}_points"]
        if len(points) > 0:
            sum_x = sum(p[0] for p in points)
            sum_y = sum(p[1] for p in points)
            
            new_x = sum_x / len(points)
            new_y = sum_y / len(points)
            
            centroids_data[f"centroid_{j}_xy"] = [new_x, new_y]

def run_kmeans(x_arr, y_arr, k, rounds):
    """Run K-means algorithm and store history"""
    centroids_data = {}
    
    # Initialize centroids
    x_c, y_c = initialize_centroids(k)
    for j in range(k):
        centroids_data[f"centroid_{j}_xy"] = [x_c[j], y_c[j]]
        centroids_data[f"centroid_{j}_points"] = []
    
    # Round -1: Initial State (Points and Centroids, NO assignment)
    history = [{
        'centroids': [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)],
        'assignments': [[] for j in range(k)], # Empty assignment list
        'all_points': list(zip(x_arr, y_arr)), # Full list of all points
        'prev_centroids': [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)],
    }]
    
    # Round 0: First Assignment Step (Points are now colored)
    assign_centroids(x_arr, y_arr, centroids_data, k)
    
    history.append({
        'centroids': [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)],
        'assignments': [[list(p) for p in centroids_data[f"centroid_{j}_points"]] for j in range(k)],
        'all_points': [],
        'prev_centroids': [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)], # Previous is the initial random position
    })
    
    # Run K-means for Rounds 1 to 'rounds'
    for round_num in range(rounds):
        
        prev_centroids_state = [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)]
        
        # K-Means Step 1: Recalculate Centroid Positions 
        recalculate_centroids(centroids_data, k)
        
        # K-Means Step 2: Assign Points
        assign_centroids(x_arr, y_arr, centroids_data, k)
        
        history.append({
            'centroids': [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)],
            'assignments': [[list(p) for p in centroids_data[f"centroid_{j}_points"]] for j in range(k)],
            'all_points': [],
            'prev_centroids': prev_centroids_state,
        })
    
    return history

def create_plot_base64(k, history_index, history):
    """Create visualization for a specific history index"""
    round_num = history_index - 1
    
    colors = ['#ef4444', '#ec4899', '#22c55e', '#3b82f6', '#f97316', 
              '#a855f7', '#78716c', '#64748b', '#84cc16', '#06b6d4']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'K-Means Clustering - Round {round_num}', 
                 fontsize=14, fontfamily='monospace', pad=20)
    
    current_state = history[history_index]
    
    # Plot centroid trails (Round 0 onwards)
    if round_num >= 0:
        for j in range(k):
            # Trail starts from Round 0 (history index 1)
            trail_x = [history[r]['centroids'][j][0] for r in range(1, min(history_index + 1, len(history)))]
            trail_y = [history[r]['centroids'][j][1] for r in range(1, min(history_index + 1, len(history)))]
            
            for i in range(len(trail_x) - 1):
                alpha = (i + 1) / len(trail_x) * 0.5
                ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                       color=colors[j], alpha=alpha, linewidth=2, linestyle='--')
                
                if i < len(trail_x) - 1:
                    arrow = FancyArrowPatch((trail_x[i], trail_y[i]), 
                                           (trail_x[i+1], trail_y[i+1]),
                                           arrowstyle='->', mutation_scale=15, 
                                           color=colors[j], alpha=alpha, linewidth=1.5)
                    ax.add_patch(arrow)
    
    
    # ------------------------------------------------------------------
    # *** NEW FEATURE: Cluster Glow (Ellipse) ***
    # ------------------------------------------------------------------
    if round_num >= 0:
        for j in range(k):
            points = np.array(current_state['assignments'][j])
            
            if len(points) >= 2: # Need at least 2 points to calculate variance
                cx, cy = current_state['centroids'][j]
                
                # Calculate standard deviation for spread (or covariance for orientation)
                std_x = np.std(points[:, 0]) * 1.5 # 1.5x scaling for visibility
                std_y = np.std(points[:, 1]) * 1.5 
                
                # Use a larger minimum size to ensure small clusters are visible
                width = max(std_x * 2, 2.5) 
                height = max(std_y * 2, 2.5)
                
                # Create and add the translucent ellipse
                ellipse = Ellipse((cx, cy), width, height, 
                                  angle=0, # Simplification: assume no rotation
                                  alpha=0.15, # Faint glow
                                  facecolor=colors[j],
                                  edgecolor=colors[j],
                                  linewidth=1,
                                  zorder=0) # Place far back
                ax.add_patch(ellipse)

    # Plot points
    if round_num == -1:
        # Round -1: Plot all points in a neutral color (unassigned)
        points = current_state['all_points']
        points_x = [p[0] for p in points]
        points_y = [p[1] for p in points]
        ax.scatter(points_x, points_y, c='#4b5563', s=80, 
                  alpha=0.6, edgecolors='black', linewidth=1)
    else:
        # Round 0 onwards: Plot points assigned to each centroid
        for j in range(k):
            if len(current_state['assignments'][j]) > 0:
                points = current_state['assignments'][j]
                points_x = [p[0] for p in points]
                points_y = [p[1] for p in points]
                ax.scatter(points_x, points_y, c=colors[j], s=80, 
                          alpha=0.6, edgecolors='black', linewidth=1, zorder=1) # Above the glow

    # Plot ghost of previous centroid (Ghost only appears for Round 1 onwards)
    if round_num >= 1:
        prev_state = history[history_index]['prev_centroids']
        for j in range(k):
            px, py = prev_state[j]
            ax.scatter(px, py, c=colors[j], s=400, marker='X', 
                      edgecolors='black', linewidth=2, zorder=9, alpha=0.3)
            ax.scatter(px, py, c='white', s=100, marker='X', zorder=10, alpha=0.3)
    
    # Plot current centroids (always plotted)
    for j in range(k):
        cx, cy = current_state['centroids'][j]
        ax.scatter(cx, cy, c=colors[j], s=400, marker='X', 
                  edgecolors='black', linewidth=2, zorder=11)
        ax.scatter(cx, cy, c='white', s=100, marker='X', zorder=12)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[j], markersize=10, 
                                 label=f'Cluster {j+1}') for j in range(k)]
    
    if round_num == -1:
         legend_elements.insert(0, plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='#4b5563', markersize=10, 
                                 label='Unassigned Points'))
    
    ax.legend(handles=legend_elements, loc='upper right', 
             framealpha=0.9, fontsize=10)
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

@app.route('/')
def index():
    return render_template('kmeans_index.html')

@app.route('/generate', methods=['POST'])
def generate():
    n = int(request.form.get('n', 50))
    k = int(request.form.get('k', 3))
    rounds = int(request.form.get('rounds', 10))
    
    x_arr, y_arr = generate_points(n)
    history = run_kmeans(x_arr, y_arr, k, rounds)
    
    total_states = rounds + 2 
    all_plots_base64 = []
    
    for r in range(total_states): 
        img_base64 = create_plot_base64(k, r, history)
        all_plots_base64.append(img_base64)
    
    return jsonify({
        'success': True,
        'n': n,
        'k': k,
        'rounds': rounds,
        'plots': all_plots_base64
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)