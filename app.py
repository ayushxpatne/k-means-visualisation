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

# --- Vectorized Core Functions ---

def get_data_arrays(x_arr, y_arr, centroids_data, k):
    """Converts Python lists/dicts to optimized NumPy arrays."""
    # Points array (N, 2)
    points = np.array(list(zip(x_arr, y_arr)), dtype=np.float64)
    
    # Centroids array (K, 2)
    centroids = np.array([centroids_data[f"centroid_{j}_xy"] for j in range(k)], dtype=np.float64)
    
    return points, centroids

def assign_centroids_vectorized(points, centroids, k):
    """
    Assigns points to the nearest centroid using NumPy broadcasting for speed.
    Returns: assignment_indices (N,), distances (N, K)
    """
    # 1. Calculate the difference between every point and every centroid. (N, K, 2)
    # The newaxis magic: (N, 1, 2) - (1, K, 2) -> (N, K, 2)
    diff = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    
    # 2. Square the differences and sum the dimensions (x^2 + y^2). (N, K)
    sq_distances = np.sum(diff**2, axis=2)
    
    # 3. Find the index of the minimum distance for each point. (N,)
    assignment_indices = np.argmin(sq_distances, axis=1)
    
    return assignment_indices

def run_kmeans(x_arr, y_arr, k, rounds):
    """Run K-means algorithm and store history"""
    centroids_data = {}
    
    # Initialize centroids
    x_c, y_c = initialize_centroids(k)
    for j in range(k):
        centroids_data[f"centroid_{j}_xy"] = [x_c[j], y_c[j]]
        centroids_data[f"centroid_{j}_points"] = []
    
    # Get initial NumPy data structure
    points, current_centroids = get_data_arrays(x_arr, y_arr, centroids_data, k)
    
    # Round -1: Initial State (Points and Centroids, NO assignment)
    history = [{
        'centroids': [list(c) for c in current_centroids],
        'assignments': [[] for j in range(k)], 
        'all_points': list(zip(x_arr, y_arr)), 
        'prev_centroids': [list(c) for c in current_centroids],
    }]
    
    # K-Means Loop
    for round_num in range(rounds + 1):
        prev_centroids_state = [list(c) for c in current_centroids]
        
        if round_num > 0:
            # Step 1: Recalculate Centroid Positions (only from Round 1 onwards)
            # Use the assignments from the previous step to find new means
            new_centroids = np.copy(current_centroids)
            for j in range(k):
                # Select all points assigned to cluster j
                cluster_points = points[assignment_indices == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
            current_centroids = new_centroids
        
        # Step 2: Assign Points (Run for Round 0 and all subsequent rounds)
        assignment_indices = assign_centroids_vectorized(points, current_centroids, k)
        
        # Organize assignments for history/plotting
        assignments_list = []
        for j in range(k):
            cluster_points_list = points[assignment_indices == j].tolist()
            assignments_list.append(cluster_points_list)
        
        history.append({
            'centroids': [list(c) for c in current_centroids],
            'assignments': assignments_list,
            'all_points': [],
            'prev_centroids': prev_centroids_state,
        })
        
    # The history list is now [Round -1, Round 0, Round 1, ..., Round R]
    return history

# --- Visualization Functions (Unchanged, but robust) ---

def initialize_centroids(k, x_max=20, y_max=20):
    """Initialize random centroids (unchanged)"""
    x_c = [random.uniform(0, x_max) for _ in range(k)]
    y_c = [random.uniform(0, y_max) for _ in range(k)]
    return x_c, y_c

def generate_points(n, x_max=20, y_max=20):
    """Generate random points (unchanged)"""
    x_arr = [random.randint(0, x_max) for _ in range(n)]
    y_arr = [random.randint(0, y_max) for _ in range(n)]
    return x_arr, y_arr

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
    
    
    # Cluster Glow (Ellipse)
    if round_num >= 0:
        for j in range(k):
            points = np.array(current_state['assignments'][j])
            
            if len(points) >= 2: 
                cx, cy = current_state['centroids'][j]
                
                # Calculate standard deviation for spread
                std_x = np.std(points[:, 0]) * 1.5 
                std_y = np.std(points[:, 1]) * 1.5 
                
                width = max(std_x * 2, 2.5) 
                height = max(std_y * 2, 2.5)
                
                ellipse = Ellipse((cx, cy), width, height, 
                                  angle=0,
                                  alpha=0.15,
                                  facecolor=colors[j],
                                  edgecolor=colors[j],
                                  linewidth=1,
                                  zorder=0) 
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
                          alpha=0.6, edgecolors='black', linewidth=1, zorder=1)

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

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('kmeans_index.html')

@app.route('/generate', methods=['POST'])
def generate():
    n = int(request.form.get('n', 50))
    k = int(request.form.get('k', 3))
    rounds = int(request.form.get('rounds', 10))
    
    x_arr, y_arr = generate_points(n)
    # The run_kmeans function is now significantly faster due to vectorization
    history = run_kmeans(x_arr, y_arr, k, rounds)
    
    total_states = rounds + 2 
    all_plots_base64 = []
    
    # This loop remains the bottleneck (Matplotlib rendering), but is unavoidable
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