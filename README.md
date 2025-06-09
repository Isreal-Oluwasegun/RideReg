# LaneWatch - Neural Vehicle Type Classifier for Law Enforcement

**LaneWatch** is a deep learning-powered image classifier that identifies vehicle types — specifically distinguishing between **CNG vehicles** and **buses** — using computer vision. Built with future integration in mind (e.g. **live video feeds**, **traffic enforcement**, and **smart city AI systems**), LaneWatch helps automate rule enforcement and environmental compliance in urban mobility.

---

## Key Features

- **Binary Image Classification** - CNG vs Bus  
- **Law Enforcement Use Case** - Ideal for surveillance camera feeds  
- **Built for Extensibility** - Future-ready for live video input & additional vehicle classes  
- **Streamlit Frontend** - Upload images & receive instant classification  
- **Model Explainability** - Ready for integration with Grad-CAM & attention visualization

---

## Use Cases

- **Law Enforcement**: Monitor restricted lanes and catch violations (e.g. non-CNGs in CNG-only lanes)
- **Environmental Compliance**: Classify cleaner-energy vehicles on the road
- **Transit Analytics**: Track proportions of buses vs small-scale public vehicles
- **Smart City Infrastructure**: Feed traffic insights into central dashboards
---
## Tech Stack

- Python 3.10+
- TensorFlow/Keras or PyTorch
- OpenCV / PIL
- Streamlit (for deployment)
- Matplotlib & Scikit-learn (for evaluation)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Isreal-Oluwasegun/LaneWatch.git
cd LaneWatch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

### 4. Run the App (Streamlit)

```bash
streamlit run app.py
```

---

---

## Performance Snapshot

Performance
🔹 Achieved 76% accuracy with well-balanced precision and recall. 🔹 Custom loss weighting to improve model fairness in imbalanced datasets. 🔹 Supports easy fine-tuning for enhanced results.

---

## Future Roadmap

**Live Video Stream Support**
-Add **YOLO-based detection** before classification
ashboard for **real-time statistics**
Explainability with **Grad-CAM** visualizations

---

## Contributing

Pull requests, feedback, and collaborations are welcome!

```bash
git checkout -b feature/new-idea
git commit -m "Added new idea"
git push origin feature/new-idea
```

---

## License

MIT License © 2025 Israel Oluwasegun

---

## Contact

Email: marksegman2@gmail.com 
GitHub: [github.com/Isreal-Oluwasegun](https://github.com/Isreal-Oluwasegun)  

---

> LaneWatch is a step toward safer, smarter urban mobility — using AI for social good.
