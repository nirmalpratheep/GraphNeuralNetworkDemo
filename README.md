# 🧠 Graph Neural Network Interactive Demo

An interactive web application that demonstrates how Graph Neural Networks (GNNs) work through visual representations and detailed mathematical calculations. This educational tool helps users understand the core concepts of message passing, node embeddings, and different GNN architectures.

## 🌟 Features

### 🎯 **Interactive GNN Models**
- **GCN (Graph Convolutional Network)**: `updated = 0.5 × self + 0.5 × pooled_neighbors`
- **GIN (Graph Isomorphism Network)**: `updated = 1.2 × self + 0.8 × pooled_neighbors`
- **GraphSage**: `updated = tanh(self + pooled_neighbors)`
- **GAN (Graph Attention Network)**: Gating mechanism with sigmoid activation

### 📊 **Pooling Strategies**
- **Mean Pooling**: Average of neighbor values
- **Max Pooling**: Maximum neighbor value
- **Attention Pooling**: Weighted average using softmax attention

### 🎨 **Visual Features**
- **Interactive Graph Visualization**: Color-coded nodes showing embedding values
- **Real-time Calculations**: Step-by-step mathematical operations
- **Node Transformation Flow**: Visual journey of how nodes change through layers
- **Timeline Visualization**: Bar charts showing embedding evolution
- **Responsive Design**: Modern, clean interface with hover effects

## 🚀 Quick Start

### Option 1: Standalone Version (Recommended for Testing)
1. Open `test_standalone.html` directly in your web browser
2. No server setup required - works offline!

### Option 2: Flask Server Version
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000`

## 📖 How to Use

### 🎛️ **Controls**
- **Model**: Select from GCN, GIN, GraphSage, or GAN
- **Layers**: Choose 1-4 layers to see how embeddings evolve
- **Pooling**: Select mean, max, or attention pooling
- **Run Button**: Execute the GNN computation

### 🔍 **Understanding the Interface**

#### **Graph Visualization**
- **Initial Embeddings**: Shows starting node values
- **Final Embeddings**: Shows computed values after GNN processing
- **Color Coding**: Red (low values) to Blue (high values)
- **Node Labels**: Display exact embedding values

#### **Calculation Details**
- **Model Configuration**: Shows selected parameters and formulas
- **Node Transformation Flow**: Visual journey of first 3 nodes through layers
- **Detailed Calculations**: Step-by-step math for each node operation
- **Change Indicators**: Shows absolute and percentage changes

## 🧮 Mathematical Background

### **Message Passing Framework**
Each GNN layer performs the following operations:

1. **Aggregation**: Collect information from neighboring nodes
2. **Combination**: Combine self-information with aggregated neighbors
3. **Transformation**: Apply model-specific update function

### **Model Formulas**

#### **GCN (Graph Convolutional Network)**
```
h^(l+1) = 0.5 × h^(l) + 0.5 × AGGREGATE({h^(l)_j : j ∈ N(i)})
```

#### **GIN (Graph Isomorphism Network)**
```
h^(l+1) = 1.2 × h^(l) + 0.8 × AGGREGATE({h^(l)_j : j ∈ N(i)})
```

#### **GraphSage**
```
h^(l+1) = tanh(h^(l) + AGGREGATE({h^(l)_j : j ∈ N(i)}))
```

#### **GAN (Graph Attention Network)**
```
gate = σ(h^(l))
h^(l+1) = gate × h^(l) + (1-gate) × AGGREGATE({h^(l)_j : j ∈ N(i)})
```

### **Pooling Functions**

#### **Mean Pooling**
```
AGGREGATE({h_j}) = (1/|N(i)|) × Σ(h_j)
```

#### **Max Pooling**
```
AGGREGATE({h_j}) = max({h_j})
```

#### **Attention Pooling**
```
scores_j = exp(h_j)
weights_j = scores_j / Σ(scores_k)
AGGREGATE({h_j}) = Σ(weights_j × h_j)
```

## 🏗️ Project Structure

```
GraphNeuralNetworkDemo/
├── app.py                 # Flask backend server
├── main.py               # Entry point (unused)
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
├── templates/
│   └── index.html       # Main web interface
├── test_standalone.html  # Standalone version (no server needed)
└── README.md            # This file
```

## 🔧 Technical Details

### **Backend (Flask)**
- **Graph Generation**: Creates deterministic random graphs with 10 nodes and 20 edges
- **GNN Implementation**: Pure Python implementation of message passing
- **API Endpoint**: `/compute` accepts POST requests with model parameters

### **Frontend (JavaScript)**
- **Graph Rendering**: SVG-based circular layout with D3-style positioning
- **Real-time Computation**: Client-side GNN calculations matching backend
- **Visual Enhancements**: CSS animations, gradients, and responsive design

### **Dependencies**
- **Flask**: Web framework for the backend API
- **NetworkX**: Graph manipulation and analysis
- **NumPy**: Numerical computations
- **No Frontend Dependencies**: Pure HTML, CSS, and JavaScript

## 🎓 Educational Value

This demo is perfect for:
- **Students** learning about Graph Neural Networks
- **Researchers** understanding message passing mechanisms
- **Developers** implementing GNN architectures
- **Educators** teaching graph machine learning concepts

### **Key Learning Outcomes**
1. **Message Passing**: How information flows between connected nodes
2. **Layer-wise Processing**: How multiple layers create deeper representations
3. **Model Differences**: How different architectures affect node updates
4. **Pooling Strategies**: How neighbor information is aggregated
5. **Visual Intuition**: Connecting mathematical operations to visual results

## 🚀 Advanced Usage

### **Customizing the Demo**
- **Graph Size**: Modify `N` in the code to change number of nodes
- **Initial Embeddings**: Change the initial values array
- **Model Parameters**: Adjust weights in the update functions
- **Visual Styling**: Customize CSS for different themes

### **Extending Functionality**
- **New Models**: Add additional GNN architectures
- **Different Graphs**: Implement various graph generation methods
- **Export Features**: Add data export capabilities
- **Animation**: Implement step-by-step animation of calculations

## 🐛 Troubleshooting

### **Common Issues**

1. **Flask Server Won't Start**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check if port 5000 is available
   - Try running with `python -m flask run`

2. **Calculations Not Showing**
   - Use the standalone version (`test_standalone.html`) for immediate testing
   - Check browser console for JavaScript errors
   - Ensure JavaScript is enabled

3. **Visual Issues**
   - Try refreshing the page
   - Check browser compatibility (modern browsers recommended)
   - Clear browser cache if needed

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional GNN models (GAT, GraphTransformer, etc.)
- More sophisticated graph generation
- Enhanced visualizations
- Performance optimizations
- Documentation improvements

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by the Graph Neural Network research community
- Built for educational purposes to make GNNs more accessible
- Uses standard web technologies for broad compatibility

---

**Happy Learning! 🎉**

*Explore the fascinating world of Graph Neural Networks through interactive visualization and detailed mathematical explanations.*