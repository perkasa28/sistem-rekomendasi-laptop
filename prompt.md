# Laptop Recommendation System Development Prompt

## Task Description
Create a laptop recommendation system using content-based filtering with the following specifications:

## Development Steps

### 1. Project Structure
Create the following directory structure:
```
d:\PROPOSAL\SistemRekomendasi\
├── data\
│   └── dummy_laptops.csv
├── src\
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessor.py
│   ├── recommender.py
│   └── app.py
├── tests\
│   └── test_recommender.py
└── requirements.txt
```

### 2. Generate Dataset
Create a dummy dataset with 150 laptop entries containing:
- Basic specs (brand, model, price)
- Performance specs (CPU, GPU, RAM)
- Display specs (screen size, type)
- Physical specs (weight, battery)
- Usage category (Gaming, Office, Student, Creator)

### 3. Core Features
1. Data preprocessing:
   - Normalize numerical values
   - Encode categorical variables
   - Create feature vectors

2. Recommendation engine:
   - Implement content-based filtering
   - Use cosine similarity for matching
   - Support user preference weighting

3. User interface:
   - Input: usage purpose, budget range, priorities
   - Output: top 5 recommended laptops with specs
   - Simple feedback mechanism

### 4. Technical Requirements
- Python 3.8+
- Essential libraries: pandas, numpy, scikit-learn, streamlit
- Code documentation and type hints
- Unit tests for core functions
- Error handling for user inputs

### 5. Evaluation Criteria
- Code quality and organization
- Recommendation relevance
- System response time
- User interface usability

### 6. Deliverables
1. Working prototype with dummy data
2. Documentation of implementation
3. Test results and metrics
4. Usage instructions

### 7. Notes
- Focus on modularity and clean code
- Prioritize recommendation accuracy
- Keep UI simple but functional
- Document assumptions and limitations

##