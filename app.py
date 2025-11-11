from flask import Flask, render_template, request, jsonify, session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import io
import base64
from matplotlib.patches import FancyArrowPatch

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
    # Clear previous assignments
    for j in range(k):
        centroids_data[f"centroid_{j}_points"] = []
    
    # Assign each point to nearest centroid
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
    
    # Store history of centroid positions - initial state
    history = [{
        'centroids': [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)],
        'assignments': [[] for j in range(k)]
    }]
    
    # Run K-means for specified rounds
    for round_num in range(rounds):
        assign_centroids(x_arr, y_arr, centroids_data, k)
        recalculate_centroids(centroids_data, k)
        
        # Store this round's state
        history.append({
            'centroids': [list(centroids_data[f"centroid_{j}_xy"]) for j in range(k)],
            'assignments': [[list(p) for p in centroids_data[f"centroid_{j}_points"]] for j in range(k)]
        })
    
    return history

def create_plot(x_arr, y_arr, k, round_num, history):
    """Create visualization for a specific round"""
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
    
    current_state = history[round_num]
    
    # Plot centroid trails with arrows
    for j in range(k):
        trail_x = [history[r]['centroids'][j][0] for r in range(min(round_num + 1, len(history)))]
        trail_y = [history[r]['centroids'][j][1] for r in range(min(round_num + 1, len(history)))]
        
        # Draw trail with decreasing alpha
        for i in range(len(trail_x) - 1):
            alpha = (i + 1) / len(trail_x) * 0.5
            ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                   color=colors[j], alpha=alpha, linewidth=2, linestyle='--')
            
            # Add arrow at the end of each segment
            if i < len(trail_x) - 2:
                arrow = FancyArrowPatch((trail_x[i], trail_y[i]), 
                                       (trail_x[i+1], trail_y[i+1]),
                                       arrowstyle='->', mutation_scale=15, 
                                       color=colors[j], alpha=alpha, linewidth=1.5)
                ax.add_patch(arrow)
    
    # Plot points assigned to each centroid
    for j in range(k):
        if len(current_state['assignments'][j]) > 0:
            points = current_state['assignments'][j]
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            ax.scatter(points_x, points_y, c=colors[j], s=80, 
                      alpha=0.6, edgecolors='black', linewidth=1)
    
    # Plot current centroids
    for j in range(k):
        cx, cy = current_state['centroids'][j]
        ax.scatter(cx, cy, c=colors[j], s=400, marker='X', 
                  edgecolors='black', linewidth=2, zorder=10)
        ax.scatter(cx, cy, c='white', s=100, marker='X', zorder=11)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[j], markersize=10, 
                                 label=f'Cluster {j+1}') for j in range(k)]
    ax.legend(handles=legend_elements, loc='upper right', 
             framealpha=0.9, fontsize=10)
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    return render_template('kmeans_index.html')

@app.route('/generate', methods=['POST'])
def generate():
    n = int(request.form.get('n', 50))
    k = int(request.form.get('k', 3))
    rounds = int(request.form.get('rounds', 10))
    
    # Generate points
    x_arr, y_arr = generate_points(n)
    
    # Run K-means
    history = run_kmeans(x_arr, y_arr, k, rounds)
    
    # Store in session
    session['x_arr'] = x_arr
    session['y_arr'] = y_arr
    session['k'] = k
    session['rounds'] = rounds
    session['history'] = history
    
    return jsonify({
        'success': True,
        'n': n,
        'k': k,
        'rounds': rounds
    })

@app.route('/get_plot/<int:round_num>')
def get_plot(round_num):
    x_arr = session.get('x_arr')
    y_arr = session.get('y_arr')
    k = session.get('k')
    history = session.get('history')
    
    if not all([x_arr, y_arr, k, history]):
        return jsonify({'error': 'No data available'}), 400
    
    # Ensure round_num is within bounds
    round_num = min(round_num, len(history) - 1)
    
    img_base64 = create_plot(x_arr, y_arr, k, round_num, history)
    
    return jsonify({
        'image': img_base64,
        'round': round_num,
        'total_rounds': len(history) - 1
    })

if __name__ == '__main__':
    app.run(debug=True)