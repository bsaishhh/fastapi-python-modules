#!/usr/bin/env python3
"""Generate static benchmark JD files (run once, store forever)."""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "modules" / "resume_ats" / "data" / "jds"

JD_DEFINITIONS = {
    "ai_ml_engineer": {
        "role_key": "AI_ML_ENGINEER",
        "family": "AI_ML_DATA_SCIENCE",
        "sub_roles": ["AI Engineer", "ML Engineer", "LLM Engineer", "GenAI Engineer", "Research Scientist"],
        "required_skills": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow"],
        "preferred_skills": ["LangChain", "RAG", "Transformers", "MLOps", "Vector Databases"],
        "tools": ["Docker", "AWS", "Git", "MLflow", "Kubernetes"],
        "frameworks": ["FastAPI", "scikit-learn", "Hugging Face Transformers", "LangChain"],
        "responsibilities": [
            "Design and deploy ML models in production",
            "Build LLM and RAG pipelines",
            "Optimize model inference and training",
            "Collaborate with data and platform teams",
        ],
        "keywords": ["Python", "PyTorch", "TensorFlow", "Transformers", "LLM", "RAG", "LangChain", "Vector Database", "FastAPI", "Docker", "AWS", "MLOps"],
    },
    "software_engineer": {
        "role_key": "SOFTWARE_ENGINEER",
        "family": "SOFTWARE_ENGINEERING",
        "sub_roles": ["Backend", "Frontend", "Full Stack", "Mobile", "Platform", "DevOps", "SRE"],
        "required_skills": ["Python", "Java", "JavaScript", "Data Structures", "Algorithms", "System Design"],
        "preferred_skills": ["TypeScript", "React", "Go", "Microservices", "CI/CD"],
        "tools": ["Git", "Docker", "Kubernetes", "AWS", "PostgreSQL", "Redis"],
        "frameworks": ["FastAPI", "Django", "Spring", "React", "Node.js"],
        "responsibilities": ["Build scalable backend services", "Design REST APIs", "Write clean maintainable code", "Participate in code reviews"],
        "keywords": ["Python", "Java", "JavaScript", "REST API", "Microservices", "Docker", "Kubernetes", "PostgreSQL", "System Design", "CI/CD"],
    },
    "data_engineer": {
        "role_key": "DATA_ENGINEER",
        "family": "DATA_ENGINEERING_ANALYTICS",
        "sub_roles": ["Data Engineer", "BI", "Analytics", "ETL", "Data Platform"],
        "required_skills": ["Python", "SQL", "ETL", "Data Pipelines", "Spark"],
        "preferred_skills": ["Airflow", "dbt", "Kafka", "Snowflake", "BigQuery"],
        "tools": ["Apache Spark", "Airflow", "PostgreSQL", "AWS", "Docker"],
        "frameworks": ["PySpark", "dbt", "Pandas"],
        "responsibilities": ["Build data pipelines", "Design data warehouses", "Ensure data quality", "Optimize ETL jobs"],
        "keywords": ["Python", "SQL", "Spark", "Airflow", "ETL", "Data Pipeline", "dbt", "Kafka", "AWS", "Data Warehouse"],
    },
    "quant_finance": {
        "role_key": "QUANT_FINANCE",
        "family": "QUANT_FINANCE",
        "sub_roles": ["Quant Research", "Quant Developer", "Trader", "IB Analyst"],
        "required_skills": ["Python", "C++", "Statistics", "Probability", "Financial Modeling"],
        "preferred_skills": ["R", "MATLAB", "Time Series", "Stochastic Calculus", "Risk Models"],
        "tools": ["Bloomberg", "Git", "Linux", "SQL"],
        "frameworks": ["NumPy", "Pandas", "QuantLib"],
        "responsibilities": ["Develop trading strategies", "Build pricing models", "Backtest algorithms", "Analyze market data"],
        "keywords": ["Python", "C++", "Statistics", "Quantitative", "Derivatives", "Risk", "Backtesting", "Financial Modeling"],
    },
    "consulting_strategy": {
        "role_key": "CONSULTING_STRATEGY",
        "family": "CONSULTING_STRATEGY",
        "sub_roles": ["Business Analyst", "Consultant", "Associate", "M&A", "Program Manager"],
        "required_skills": ["Problem Solving", "Excel", "PowerPoint", "Business Analysis", "Communication"],
        "preferred_skills": ["SQL", "Python", "Financial Modeling", "Market Research"],
        "tools": ["Excel", "PowerPoint", "Tableau", "SQL"],
        "frameworks": [],
        "responsibilities": ["Conduct market analysis", "Develop strategic recommendations", "Client presentations", "Data-driven insights"],
        "keywords": ["Strategy", "Consulting", "Business Analysis", "Excel", "PowerPoint", "Market Research", "Financial Modeling"],
    },
    "product_design": {
        "role_key": "PRODUCT_DESIGN",
        "family": "PRODUCT_DESIGN",
        "sub_roles": ["APM", "PM", "UX", "UI", "Product Engineer"],
        "required_skills": ["Product Management", "User Research", "Roadmapping", "Agile", "Communication"],
        "preferred_skills": ["Figma", "SQL", "A/B Testing", "Analytics", "Prototyping"],
        "tools": ["Figma", "Jira", "Amplitude", "Mixpanel"],
        "frameworks": ["React"],
        "responsibilities": ["Define product vision", "Prioritize features", "Work with engineering", "User research and testing"],
        "keywords": ["Product Management", "UX", "UI", "Figma", "Agile", "Roadmap", "User Research", "A/B Testing"],
    },
    "mechanical_manufacturing": {
        "role_key": "MECHANICAL_MANUFACTURING",
        "family": "MECHANICAL_MANUFACTURING",
        "sub_roles": ["Mechanical Design", "CFD", "FEA", "Thermal", "Materials", "Manufacturing"],
        "required_skills": ["CAD", "SolidWorks", "Mechanical Design", "Thermodynamics", "Manufacturing Processes"],
        "preferred_skills": ["ANSYS", "CFD", "FEA", "AutoCAD", "GD&T"],
        "tools": ["SolidWorks", "ANSYS", "AutoCAD", "MATLAB"],
        "frameworks": [],
        "responsibilities": ["Design mechanical systems", "Perform FEA/CFD analysis", "Optimize manufacturing", "Prototype testing"],
        "keywords": ["CAD", "SolidWorks", "FEA", "CFD", "Mechanical Design", "Manufacturing", "ANSYS", "Thermal"],
    },
    "electrical_electronics": {
        "role_key": "ELECTRICAL_ELECTRONICS",
        "family": "ELECTRICAL_ELECTRONICS",
        "sub_roles": ["VLSI", "ASIC", "Embedded", "Firmware", "RF", "Power Electronics"],
        "required_skills": ["Circuit Design", "Embedded Systems", "C", "Verilog", "VHDL"],
        "preferred_skills": ["FPGA", "RTOS", "PCB Design", "Signal Processing"],
        "tools": ["Altium", "Cadence", "Oscilloscope", "MATLAB"],
        "frameworks": [],
        "responsibilities": ["Design circuits and PCBs", "Develop firmware", "VLSI/ASIC design", "Hardware testing"],
        "keywords": ["Embedded", "VLSI", "Verilog", "FPGA", "Circuit Design", "Firmware", "PCB", "Signal Processing"],
    },
    "aerospace_defence": {
        "role_key": "AEROSPACE_DEFENCE",
        "family": "AEROSPACE_DEFENCE",
        "sub_roles": ["Avionics", "Propulsion", "GNC", "Structures", "Flight Testing"],
        "required_skills": ["Aerodynamics", "MATLAB", "Control Systems", "Structures", "CFD"],
        "preferred_skills": ["Simulink", "FEA", "Avionics", "Propulsion"],
        "tools": ["MATLAB", "ANSYS", "CATIA", "SolidWorks"],
        "frameworks": [],
        "responsibilities": ["Aerospace system design", "GNC algorithm development", "Structural analysis", "Flight test support"],
        "keywords": ["Aerospace", "Avionics", "GNC", "Propulsion", "CFD", "Control Systems", "Structures", "MATLAB"],
    },
    "core_science_rnd": {
        "role_key": "CORE_SCIENCE_RND",
        "family": "CORE_SCIENCE_RND",
        "sub_roles": ["Physics", "Chemistry", "Math", "Research", "Operations Research"],
        "required_skills": ["Research", "Statistics", "Python", "MATLAB", "Scientific Writing"],
        "preferred_skills": ["LaTeX", "R", "Simulation", "Lab Techniques"],
        "tools": ["MATLAB", "Python", "LaTeX", "Jupyter"],
        "frameworks": ["NumPy", "SciPy"],
        "responsibilities": ["Conduct research experiments", "Publish papers", "Data analysis", "Model development"],
        "keywords": ["Research", "Publications", "Statistics", "Python", "MATLAB", "Simulation", "Scientific Writing"],
    },
    "civil_infrastructure": {
        "role_key": "CIVIL_INFRASTRUCTURE",
        "family": "CIVIL_INFRASTRUCTURE",
        "sub_roles": ["Structural", "Geotech", "BIM", "Construction"],
        "required_skills": ["Structural Analysis", "AutoCAD", "STAAD", "Concrete Design", "Surveying"],
        "preferred_skills": ["Revit", "BIM", "ETABS", "SAP2000", "Geotechnical"],
        "tools": ["AutoCAD", "Revit", "STAAD Pro", "ETABS"],
        "frameworks": [],
        "responsibilities": ["Structural design", "BIM modeling", "Site supervision", "Infrastructure planning"],
        "keywords": ["Structural", "BIM", "AutoCAD", "Revit", "Construction", "Geotech", "STAAD", "Infrastructure"],
    },
    "robotics_autonomous": {
        "role_key": "ROBOTICS_AUTONOMOUS",
        "family": "ROBOTICS_AUTONOMOUS",
        "sub_roles": ["Robotics", "Perception", "SLAM", "ROS", "Controls", "Automation", "Drones"],
        "required_skills": ["ROS", "Python", "C++", "Computer Vision", "Control Systems"],
        "preferred_skills": ["SLAM", "OpenCV", "PyTorch", "Gazebo", "Motion Planning"],
        "tools": ["ROS", "Gazebo", "OpenCV", "Docker"],
        "frameworks": ["PyTorch", "OpenCV"],
        "responsibilities": ["Build autonomous systems", "Perception and SLAM", "Robot control", "Sensor integration"],
        "keywords": ["ROS", "SLAM", "Computer Vision", "Robotics", "OpenCV", "Control Systems", "Autonomous", "Drones"],
    },
    "founders_office": {
        "role_key": "FOUNDERS_OFFICE",
        "family": "FOUNDERS_OFFICE",
        "sub_roles": ["Growth", "CEO Office", "Strategy Ops", "Startup Generalist"],
        "required_skills": ["Strategy", "Operations", "Communication", "Excel", "Problem Solving"],
        "preferred_skills": ["SQL", "Python", "Fundraising", "Market Research", "Product"],
        "tools": ["Excel", "Notion", "SQL", "CRM"],
        "frameworks": [],
        "responsibilities": ["Support founder initiatives", "Cross-functional projects", "Investor relations", "Growth experiments"],
        "keywords": ["Strategy", "Operations", "Startup", "Growth", "Fundraising", "Cross-functional", "CEO Office"],
    },
    "education_edtech": {
        "role_key": "EDUCATION_EDTECH",
        "family": "EDUCATION_EDTECH",
        "sub_roles": ["Faculty", "SME", "JEE", "NEET", "Instructional Design"],
        "required_skills": ["Teaching", "Curriculum Design", "Subject Matter Expertise", "Communication"],
        "preferred_skills": ["EdTech", "LMS", "Assessment Design", "Video Content"],
        "tools": ["LMS", "Google Classroom", "Zoom", "Canva"],
        "frameworks": [],
        "responsibilities": ["Design curriculum", "Create learning content", "Student assessment", "EdTech product input"],
        "keywords": ["Teaching", "Curriculum", "EdTech", "Instructional Design", "Assessment", "JEE", "NEET"],
    },
    "gaming_graphics": {
        "role_key": "GAMING_GRAPHICS",
        "family": "GAMING_GRAPHICS",
        "sub_roles": ["Game Development", "Rendering", "Graphics", "3D Art"],
        "required_skills": ["C++", "Unity", "Unreal Engine", "Game Design", "3D Graphics"],
        "preferred_skills": ["OpenGL", "Vulkan", "Blender", "Shader Programming"],
        "tools": ["Unity", "Unreal Engine", "Blender", "Git"],
        "frameworks": [],
        "responsibilities": ["Game development", "Graphics rendering", "Gameplay programming", "Performance optimization"],
        "keywords": ["Unity", "Unreal Engine", "C++", "Game Development", "Rendering", "OpenGL", "3D", "Graphics"],
    },
    "supply_chain_operations": {
        "role_key": "SUPPLY_CHAIN_OPERATIONS",
        "family": "SUPPLY_CHAIN_OPERATIONS",
        "sub_roles": ["SCM", "Logistics", "Procurement", "Planning", "Operations"],
        "required_skills": ["Supply Chain", "Operations", "Excel", "Inventory Management", "Logistics"],
        "preferred_skills": ["SAP", "SQL", "Python", "Forecasting", "Lean Six Sigma"],
        "tools": ["SAP", "Excel", "ERP", "Tableau"],
        "frameworks": [],
        "responsibilities": ["Supply chain optimization", "Inventory planning", "Vendor management", "Operations analytics"],
        "keywords": ["Supply Chain", "Logistics", "Procurement", "Inventory", "SAP", "Operations", "Forecasting"],
    },
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, jd in JD_DEFINITIONS.items():
        path = OUTPUT_DIR / f"{filename}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(jd, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
