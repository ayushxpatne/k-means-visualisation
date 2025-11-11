from flask import Flask, render_template, request, jsonify
import random
import numpy as np

app = Flask(__name__)
app.secret_key = 'kmeans_secret_key_2024'

# --- Vectorized Core Functions ---

def get_data_arrays(x_arr, y_arr, centroids_data, k):
    """Converts Python lists/dicts to optimized NumPy arrays."""
    points = np.array(list(zip(x_arr, y_arr)), dtype=np.float64)
    centroids = np.array([centroids_data[f"centroid_{j}_xy"] for j in range(k)], dtype=np.float64)
    return points, centroids

def assign_centroids_vectorized(points, centroids, k):
    """Assigns points to the nearest centroid using NumPy broadcasting."""
    diff = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    sq_distances = np.sum(diff**2, axis=2)
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
    
    points, current_centroids = get_data_arrays(x_arr, y_arr, centroids_data, k)
    
    # Round -1: Initial State
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
            new_centroids = np.copy(current_centroids)
            for j in range(k):
                cluster_points = points[assignment_indices == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
            current_centroids = new_centroids
        
        assignment_indices = assign_centroids_vectorized(points, current_centroids, k)
        
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
        
    return history

def initialize_centroids(k, x_max=20, y_max=20):
    """Initialize random centroids"""
    x_c = [random.uniform(0, x_max) for _ in range(k)]
    y_c = [random.uniform(0, y_max) for _ in range(k)]
    return x_c, y_c

def generate_points(n, x_max=20, y_max=20):
    """Generate random points"""
    x_arr = [random.randint(0, x_max) for _ in range(n)]
    y_arr = [random.randint(0, y_max) for _ in range(n)]
    return x_arr, y_arr

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
    history = run_kmeans(x_arr, y_arr, k, rounds)
    
    # Return raw data instead of images - much faster and smaller!
    return jsonify({
        'success': True,
        'n': n,
        'k': k,
        'rounds': rounds,
        'history': history  # Just send the raw clustering data
    })

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)