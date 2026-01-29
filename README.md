# Image Detection CNN with ResNet

This project uses a pretrained **ResNet** model to classify images into multiple categories. You can define as many classes as needed.

**Class mapping example:**

```
0 – Class 1
1 – Class 2
2 – Class 3
N – Class N (add as many as required)
```

## Folder Structure

```
data/          # Dataset folder
src/           # Source code
checkpoints/   # Saved model weights
run.py         # Script to train and evaluate the model
```

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

To train and evaluate the model, run:

```bash
python run.py
```
