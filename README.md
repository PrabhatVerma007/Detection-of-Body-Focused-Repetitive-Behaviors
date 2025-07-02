# Machine Learning Model

A robust machine learning model with comprehensive cross-validation evaluation and consistent performance across multiple folds.

## 🌟 Competition Overview

This model was developed for a Kaggle competition hosted by the **Child Mind Institute**, held between **May 30 to August 27** (3 months). The competition aimed at advancing gesture recognition through wearable sensor data.

**Prizes & Awards:**

* Total prize pool: **\$50,000**
* Awards: **Points & Medals** for top-performing teams

## 🎯 Performance Overview

| Metric                  | Score      | Standard Deviation |
| ----------------------- | ---------- | ------------------ |
| **Test Score**          | **74.01%** | ±1.53%             |
| **Test Accuracy**       | **59.72%** | ±1.97%             |
| **Validation Score**    | **73.45%** | ±1.45%             |
| **Validation Accuracy** | **58.95%** | ±2.11%             |

## 📊 Cross-Validation Results

The model was evaluated using **5-fold cross-validation** to ensure robust performance assessment:

### Fold-by-Fold Performance

| Fold | Val Accuracy | Val Score | Test Accuracy | Test Score |
| ---- | ------------ | --------- | ------------- | ---------- |
| 1    | 63.09%       | 76.19%    | 63.55%        | **76.93%** |
| 2    | 58.52%       | 73.32%    | 59.31%        | 73.35%     |
| 3    | 57.48%       | 72.94%    | 58.34%        | 73.81%     |
| 4    | 57.48%       | 71.87%    | 58.15%        | 72.42%     |
| 5    | 58.19%       | 72.95%    | 59.23%        | 73.55%     |

## ✅ Key Findings

* **🎯 Consistent Performance**: Low standard deviation (< 2%) across all folds indicates model stability
* **🚫 No Overfitting**: Test performance matches validation performance, showing good generalization
* **📈 Reliable Results**: Small variance across folds demonstrates reproducible performance
* **🏆 Best Performance**: Fold 1 achieved the highest test score of 76.93%

```
├── src/
│   ├── model.py          # Model architecture
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation utilities
├── data/
│   ├── train/            # Training data
│   └── test/             # Test data
├── checkpoints/
│   └── model_checkpoint.pth
├── results/
│   └── kfold_results.json
├── requirements.txt
└── README.md
```

## 🔧 Model Details

* **Architecture**: \[Add your model architecture details]
* **Training Strategy**: 5-fold Cross-Validation
* **Evaluation Metrics**: Custom score metric and accuracy
* **Framework**: PyTorch
* **Device Support**: CUDA/CPU compatible

## 🛠️ Technical Notes

### Loading Checkpoints

Due to PyTorch 2.6+ security changes, use:

```python
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
```

Or for safer loading:

```python
import torch.serialization
with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
    checkpoint = torch.load(model_path, map_location=device)
```



## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

* \[Contest Link](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)

---

**Note**: Replace placeholder values like `YourModelClass`, `yourusername`, and file paths with your actual project details.
