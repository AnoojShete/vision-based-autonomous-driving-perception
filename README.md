# Autonomous Driving Perception Toolkit

Desktop toolkit for autonomous driving perception experiments, combining deep learning and classical machine learning in a single GUI application.

## Scope
- Vision Studio: traffic sign recognition, lane detection, pedestrian detection, and combined perception pipeline.
- Driving ML Lab: driving-oriented tabular ML workflow with metrics and report artifacts.
- CSV Dataset Lab: generic dataset experimentation (import, train, confusion matrix, report).

## Tech Stack
- Python
- Tkinter + ttkbootstrap
- OpenCV
- TensorFlow / Keras
- Ultralytics YOLOv8
- scikit-learn

## Quick Start
1. Install dependencies:
   `pip install -r requirements.txt`
2. Launch application:
   `python main.py`

## Repository Layout
- `gui/`: desktop UI and orchestration
- `dl_module/`: perception pipelines (traffic sign, lane, pedestrian)
- `ml_module/`: training and evaluation workflows
- `data/`: datasets and metadata
- `models/`: trained model artifacts
- `Public/`: generated reports and visual outputs

## Figures (Mermaid)

### Figure 1: High-Level System Architecture
```mermaid
flowchart LR
	U[User] --> GUI[GUI Layer\nTkinter App]

	GUI --> VS[Vision Studio]
	GUI --> DLAB[Driving ML Lab]
	GUI --> CSVLAB[CSV Dataset Lab]

	VS --> TS[Traffic Sign Module\nKeras CNN]
	VS --> LN[Lane Module\nOpenCV Pipeline]
	VS --> PD[Pedestrian Module\nYOLOv8]
	VS --> FUSION[Full Perception Pipeline]

	DLAB --> MLTASK[Driving-Oriented ML Tasks]
	CSVLAB --> GENTASK[Generic ML Tasks]

	MLTASK --> REPORTS[Reports\nCM, Classification, Grouped Errors]
	GENTASK --> REPORTS
	FUSION --> VIS[Annotated Visual Output]
```

### Figure 2: Automated ML Pipeline
```mermaid
flowchart LR
	A[Data Ingestion] --> B[Preprocessing]
	B --> C[Model Training]
	C --> D[Evaluation]

	A1[Load CSV / Driving Dataset] --> A
	B1[Null Handling\nEncoding\nFeature Selection] --> B
	C1[Decision Tree / NB / SVM / RF] --> C
	D1[Accuracy\nPrecision/Recall/F1\nConfusion Matrix\nReport Export] --> D
```

### Figure 3: Custom Traffic Sign CNN Architecture
```mermaid
flowchart TD
	I[Input 30x30x3] --> C1[Conv2D 32, 5x5, ReLU]
	C1 --> C2[Conv2D 32, 5x5, ReLU]
	C2 --> P1[MaxPool 2x2]
	P1 --> D1[Dropout 0.25]

	D1 --> C3[Conv2D 64, 3x3, ReLU]
	C3 --> C4[Conv2D 64, 3x3, ReLU]
	C4 --> P2[MaxPool 2x2]
	P2 --> D2[Dropout 0.25]

	D2 --> F[Flatten]
	F --> FC1[Dense 256, ReLU]
	FC1 --> D3[Dropout 0.5]
	D3 --> O[Dense 43, Softmax]
```

### Figure 4: Advanced Lane Pipeline
```mermaid
flowchart LR
	F0[Input Road Frame] --> W[Perspective Warping\nBird's Eye Transform]
	W --> TH[Color Thresholding\nWhite/Yellow Mask]
	TH --> SW[Sliding Window Search\nLane Pixel Extraction]
	SW --> PF[Polynomial Fit\nLeft/Right Lane Curves]
	PF --> OV[Lane Area Overlay]
	OV --> OUT[Final Annotated Frame]
```

## Notes
- Designed for CPU-friendly execution.
- Outputs are saved under `Public/` for reproducibility and review.
