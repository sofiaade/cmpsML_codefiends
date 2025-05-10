# cmpsML_codefiends

A full machine learning application developed for CMPS 470/570 to classify human mood states from lifestyle metrics such as steps, sleep, and hydration.

---

## Project Objective

To build an end-to-end ML pipeline that reads time-series lifestyle data and predicts mood states using a combination of machine learning algorithms:  
- SVM  
- Decision Tree  
- K-Nearest Neighbors  
- Artificial Neural Network  
- Random Forest  
- Voting Ensemble 

---

## Project Structure

cmpsML_codefiends/
├── CODE/ # All modular Python code
│ ├── preprocess.py
│ ├── models.py
│ ├── visualize.py
│ ├── utils.py
│ └── test_models.py
├── INPUT/
│ └── TRAIN/ # Preprocessed CSV input
├── MODEL/
│ └── saved_models/ # Saved .pkl model files
├── OUTPUT/
│ ├── plots/ # All visualization outputs
│ ├── evaluation/ # Summary metrics CSV
│ └── predictions.csv # Final predictions
├── DOC/
│ └── Final_Report.docx # Final report
├── TaskProgressReport.xlsx # Task completion report
└── README.md # This file


---

## 🧪 How to Run the Project

### Install dependencies:
```bash
pip install -r requirements.txt


