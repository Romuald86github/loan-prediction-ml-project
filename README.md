# Loan Prediction ML Project

## Problem Statement

Financial institutions face significant challenges in making quick and accurate loan approval decisions:

1. **Complex Decision Making**: Traditional loan approval processes involve manual review of multiple factors, making it time-consuming and potentially inconsistent.

2. **Risk Assessment**: Accurately assessing loan default risk while maintaining a competitive approval rate is challenging.

3. **Processing Efficiency**: Manual loan processing leads to longer wait times for applicants and higher operational costs for institutions.

4. **Data Utilization**: Financial institutions often have access to relevant data but lack efficient ways to leverage it for decision-making.

The consequences of not addressing these challenges include:
- Increased loan defaults due to inconsistent assessment
- Lost business opportunities due to slow processing
- Higher operational costs from manual processing
- Customer dissatisfaction from lengthy approval times
- Potential bias in loan approval decisions

## Solution Approach

Machine Learning (ML) has been chosen as the primary solution approach for several compelling reasons:

1. **Why Machine Learning?**
   - Ability to process multiple variables simultaneously
   - Consistent decision-making process
   - Quick and automated predictions
   - Data-driven insights for risk assessment
   - Scalable processing of loan applications

2. **ML vs Traditional Methods**
   Traditional approaches like:
   - Manual review: Time-consuming and subjective
   - Rule-based systems: Inflexible and difficult to maintain
   - Credit scoring alone: Limited in scope
   
   ML overcomes these limitations through:
   - Automated pattern recognition
   - Adaptive learning from historical data
   - Consistent and unbiased decisions
   - Comprehensive risk assessment

3. **Selected ML Models and Justification**
   - Random Forest: Main model for robust predictions and feature importance analysis
   - Standard Scaler: For normalizing numeric features
   - One-Hot Encoding: For handling categorical variables
   - Label Encoding: For target variable transformation

4. **Specific ML Objectives**
   - Predict loan approval with high accuracy
   - Identify key factors influencing loan decisions
   - Provide quick and automated decisions
   - Ensure consistent evaluation criteria

## Implementation Methodology

The project follows industry best practices in ML development:

1. **Data Pipeline**
   - Automated data cleaning and validation
   - Systematic handling of missing values
   - Feature engineering and preprocessing
   - Data standardization

2. **Model Development**
   - Random Forest classifier as the core model
   - Feature importance analysis
   - Model persistence for deployment
   - Comprehensive preprocessing pipeline

3. **Production Pipeline**
   - Flask API with RESTful endpoints
   - Docker containerization
   - Deployable on Oracle Cloud Infrastructure
   - User-friendly web interface

4. **Quality Assurance**
   - Model validation
   - Error handling
   - User input validation
   - Responsive web design

## Live Application

The application is deployed and accessible at:
http://141.147.69.176:8000

You can try the loan prediction model directly by:
1. Filling out the loan application form with your details
2. Clicking "Predict" to get the loan approval prediction
3. Viewing the color-coded result (green for approved, red for rejected)

## How to Reproduce the Work

### 1. Prerequisites
- Python 3.9 or later
- Git
- Docker
- Oracle Cloud Infrastructure (OCI) Account

### 2. Local Development Setup

Clone the repository:
```bash
git clone https://github.com/your-username/loan-prediction-ml-project.git
cd loan-prediction-ml-project
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Application Locally

```bash
python app.py
```
Access the application at http://localhost:5000

### 4. Docker Deployment

Build the Docker image:
```bash
docker build -t loan-app:v1 .
```

Run the container:
```bash
docker run -d -p 8000:8000 --name loan-app-container loan-app:v1
```

### 5. OCI Deployment

#### Setting up OCI Account
1. Create an Oracle Cloud Account:
   - Visit https://www.oracle.com/cloud/free/
   - Sign up for a free tier account
   - Complete verification process

2. Create a Compartment:
   - Navigate to Identity & Security > Compartments
   - Click "Create Compartment"
   - Name it (e.g., "Loan-Prediction")
   - Add description and click "Create"

3. Set up Virtual Cloud Network (VCN):
   - Go to Networking > Virtual Cloud Networks
   - Click "Create VCN"
   - Set up with Internet Connectivity
   - Create public subnet

4. Create Compute Instance:
   - Go to Compute > Instances
   - Click "Create Instance"
   - Choose "Always Free" eligible configuration
   - Select Oracle Linux 8
   - Generate SSH key pair
   - Save private key securely

#### Instance Configuration

1. Connect to your instance:
```bash
ssh -i <private_key> opc@<instance_ip>
```

2. Update system and install dependencies:
```bash
sudo dnf update -y
sudo dnf config-manager --enable ol8_appstream
sudo dnf config-manager --enable ol8_addons
sudo dnf install -y git
```

3. Install Docker:
```bash
sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```
Log out and log back in for group changes to take effect.

4. Clone and deploy the application:
```bash
git clone https://github.com/your-username/loan-prediction-ml-project.git
cd loan-prediction-ml-project
docker build -t loan-app:v1 .
docker run -d -p 8000:8000 --name loan-app-container loan-app:v1
```

5. Configure Security List:
   - Go to your VCN's security list
   - Add ingress rule:
     - Source: 0.0.0.0/0
     - Protocol: TCP
     - Destination Port: 8000

#### Accessing the Application

Access your application at:
```
http://<instance_public_ip>:8000
```

#### Troubleshooting

If the application is not accessible:
1. Check Docker container status:
```bash
docker ps
docker logs loan-app-container
```

2. Verify port binding:
```bash
sudo netstat -tulpn | grep :8000
```

3. Check firewall:
```bash
sudo firewall-cmd --list-all
```

4. Review security list rules in OCI Console

## Project Structure
```
loan-prediction-ml-project/
├── app.py                     # Flask application
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── models/                    # Saved ML models
│   ├── Random_Forest_model.pkl
│   └── preprocessing_pipeline.pkl
├── src/                      # Source code
│   ├── config.py
│   ├── data_loader.py
│   ├── model_trainer.py
│   └── preprocessing_pipeline.py
├── static/                   # Static files
│   └── style.css
└── templates/               # HTML templates
    └── index.html
```

## you can find the notebook here
Notebooks/Loan_Romuald (3).ipynb


## Future Enhancements

1. Model Improvements
   - Additional feature engineering
   - Model ensemble approaches
   - Regular model retraining

2. Infrastructure
   - HTTPS support
   - Load balancing
   - Automated backups

3. User Interface
   - Enhanced error handling
   - Additional input validations
   - Mobile responsiveness

4. Monitoring
   - Performance metrics
   - Usage analytics
   - Error tracking

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
