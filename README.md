## K-Means Clustering Visualizer 📊

The live demo can be accessed here: [https://k-means-visualisation.onrender.com](https://k-means-visualisation.onrender.com)
but it will be slow, the best way is to clone the repo and run locally. if you liked the project consider starring it :)

-----

## Project Overview

This is an interactive web-based tool built with **Flask** and **Matplotlib** that visualizes the step-by-step process of the K-Means clustering algorithm. It allows users to control parameters like the number of points (**n**), the number of clusters (**k**), and the number of iterations (**rounds**). All plot generation is handled server-side, and the resulting images are loaded upfront to ensure a smooth, client-side animated transition between rounds.

-----

## Key Features

  * **Round -1 (Initial State):** Clearly displays the randomly generated data points (unassigned in grey) and the initial random centroid positions before the clustering process begins.
  * **Visual Cluster Glow:** A faint, colored ellipse is drawn around each cluster, scaled by the standard deviation of its points. This subtle glow helps visually link data points to their respective centroids, especially in complex distributions.
  * **Interactive Controls:** Sliders on the front end allow real-time adjustment of `n`, `k`, and `rounds`.
  * **Centroid Movement Trail:** Animated lines and ghost markers show the historical path of each centroid over iterations, demonstrating convergence.
  * **Performance Optimization:** The core K-Means assignment and recalculation logic is **fully vectorized using NumPy**, ensuring the mathematical computation phase is extremely fast, even on low-CPU servers.

-----

## Local Setup and Installation

Follow these steps to get the project running locally in a high-performance environment.

### 1\. Clone the Repository

```bash
git clone https://github.com/ayushxpatne/k-means-visualisation.git
cd k-means-visualisation
```

### 2\. Create and Activate Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
.\venv\Scripts\activate
```

### 3\. Install Dependencies

Install the required Python packages:

```bash
pip install flask matplotlib numpy
```

### 4\. Run the Application

Start the Flask server:

```bash
python app.py
```

### 5\. Access the Visualizer

Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

-----

## Technologies Used

  * **Backend:** Python 3, Flask
  * **Computation:** NumPy (for high-speed vectorized K-Means)
  * **Visualization:** Matplotlib (for plot generation)
  * **Frontend:** HTML, CSS, JavaScript (for interactive controls and animation)

-----

## Transparency Note

I have used AI for the frontend template design and basic structure. However, the core K-Means logic was explored and built by me. If you would like to see my original exploration of the algorithm's code and logic, please check out the `kmeans.ipynb` Python notebook in the repository.
