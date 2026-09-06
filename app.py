from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)


# =========================================================
# LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# =========================================================

with open("job_description_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# SKILL DETECTION
# =========================================================

def detect_skills(text):

    text_lower = text.lower()

    # Normalize different forms
    text_lower = text_lower.replace(
        "problem-solving",
        "problem solving"
    )

    text_lower = text_lower.replace(
        "problem–solving",
        "problem solving"
    )

    text_lower = text_lower.replace(
        "problem—solving",
        "problem solving"
    )

    skills_patterns = {

        "python": r"\bpython\b",

        # Java must be a separate word
        "java": r"\bjava\b",

        "javascript": r"\bjavascript\b",

        "c++": r"\bc\+\+\b",

        "c#": r"\bc#\b",

        "sql": r"\bsql\b",

        "html": r"\bhtml\b",

        "css": r"\bcss\b",

        "react": r"\breact\b",

        "angular": r"\bangular\b",

        "node.js": r"\bnode\.js\b",

        "node": r"\bnode\b",

        "flask": r"\bflask\b",

        "django": r"\bdjango\b",

        "machine learning": r"\bmachine learning\b",

        "deep learning": r"\bdeep learning\b",

        "tensorflow": r"\btensorflow\b",

        "pytorch": r"\bpytorch\b",

        "mongodb": r"\bmongodb\b",

        "mysql": r"\bmysql\b",

        "git": r"\bgit\b",

        "github": r"\bgithub\b",

        "docker": r"\bdocker\b",

        "aws": r"\baws\b",

        "azure": r"\bazure\b",

        "linux": r"\blinux\b",

        "data analysis": r"\bdata analysis\b",

        "pandas": r"\bpandas\b",

        "numpy": r"\bnumpy\b",

        "scikit-learn": r"\bscikit[- ]learn\b",

        "rest api": r"\brest api\b",

        "api": r"\bapi\b",

        "php": r"\bphp\b",

        "laravel": r"\blaravel\b",

        "typescript": r"\btypescript\b",

        "spring boot": r"\bspring boot\b",

        "figma": r"\bfigma\b",

        "communication": r"\bcommunication\b",

        "leadership": r"\bleadership\b",

        "problem solving": r"\bproblem solving\b"
    }

    detected = []

    for skill, pattern in skills_patterns.items():

        if re.search(pattern, text_lower):
            detected.append(skill)

    return list(dict.fromkeys(detected))


# =========================================================
# MISSING / RECOMMENDED SKILLS
# =========================================================

def detect_missing_skills(detected_skills):

    recommended_skills = [
        "communication",
        "problem solving",
        "git",
        "sql",
        "python",
        "javascript",
        "api"
    ]

    missing = []

    for skill in recommended_skills:

        if skill not in detected_skills:
            missing.append(skill)

    return missing


# =========================================================
# EXPERIENCE DETECTION
# =========================================================

def detect_experience(text):

    patterns = [

        r"(\d+)\+?\s*years?\s*of\s*experience",

        r"(\d+)\+?\s*years?\s*experience",

        r"experience\s*of\s*(\d+)\+?\s*years?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text.lower()
        )

        if match:
            return match.group(1) + " years"

    return "Not specified"


# =========================================================
# EDUCATION DETECTION
# =========================================================

def detect_education(text):

    text_lower = text.lower()

    education_patterns = [

        # Bachelor
        ("Bachelor's Degree", r"\bbachelor'?s?\s+degree\b"),
        ("Bachelor", r"\bbachelor'?s?\b"),

        # BS / BSc
        ("BS", r"\bbs\b"),
        ("BSc", r"\bb\.?sc\.?\b"),

        # Computer Science
        ("Computer Science", r"\bcomputer\s+science\b"),

        # Master
        ("Master's Degree", r"\bmaster'?s?\s+degree\b"),
        ("Master", r"\bmaster'?s?\b"),

        # MS / MSc
        ("MS", r"\bms\b"),
        ("MSc", r"\bm\.?sc\.?\b"),

        # PhD
        ("PhD", r"\bph\.?d\.?\b"),

        # MBA
        ("MBA", r"\bmba\b"),

        # Engineering
        ("Engineering", r"\bengineering\b"),

        # General degree
        ("Degree", r"\bdegree\b")
    ]

    found = []

    for name, pattern in education_patterns:

        if re.search(pattern, text_lower):

            found.append(name)

    # Remove duplicates while preserving order
    found = list(dict.fromkeys(found))

    if found:
        return ", ".join(found)

    return "Not specified"


# =========================================================
# SALARY DETECTION
# =========================================================

def detect_salary(text):

    salary_patterns = [

        # Example: 80000-120000
        r"\b\d{4,}\s*[-to]+\s*\d{4,}\b",

        # Example: 80000 PKR
        r"\b\d{4,}\s*(?:pkr|usd|dollars?)\b",

        # Example: Salary is 80000
        r"(?:salary|pay|compensation)[^\d]{0,20}\d{4,}"
    ]

    for pattern in salary_patterns:

        match = re.search(
            pattern,
            text.lower()
        )

        if match:
            return match.group(0)

    return "Not specified"


# =========================================================
# RESPONSIBILITY DETECTION
# =========================================================

def detect_responsibilities(text):

    text_lower = text.lower()

    responsibility_patterns = [

        r"\bresponsibilities\b",

        r"\bresponsible for\b",

        r"\bduties\b",

        r"\bdeveloping\b",

        r"\bdevelopment\b",

        r"\bdevelop\b",

        r"\bmaintaining\b",

        r"\bmaintenance\b",

        r"\bmanage\b",

        r"\bmanaging\b",

        r"\bdesign\b",

        r"\bdesigning\b",

        r"\bimplement\b",

        r"\bimplementing\b",

        r"\bbuild\b",

        r"\bbuilding\b",

        r"\bcreate\b",

        r"\bcreating\b",

        r"\bwork on\b",

        r"\bworking on\b",

        r"\bwebsite updates\b",

        r"\bupdates and maintenance\b",

        r"\bfixing bugs\b",

        r"\bwrite code\b",

        r"\bwriting code\b"
    ]

    for pattern in responsibility_patterns:

        if re.search(pattern, text_lower):

            return True

    return False


# =========================================================
# ISSUE DETECTION
# =========================================================

def generate_issues(
    text,
    skills,
    experience,
    education,
    salary
):

    issues = []

    # -----------------------------------------
    # Short Description
    # -----------------------------------------

    if len(text.split()) < 40:

        issues.append(
            "Job description is too short."
        )

    # -----------------------------------------
    # Skills
    # -----------------------------------------

    if not skills:

        issues.append(
            "Required skills are not clearly mentioned."
        )

    # -----------------------------------------
    # Experience
    # -----------------------------------------

    if experience == "Not specified":

        issues.append(
            "Required experience is not specified."
        )

    # -----------------------------------------
    # Education
    # -----------------------------------------

    if education == "Not specified":

        issues.append(
            "Required education is not specified."
        )

    # -----------------------------------------
    # Responsibilities
    # -----------------------------------------

    if not detect_responsibilities(text):

        issues.append(
            "Job responsibilities are not clearly mentioned."
        )

    # -----------------------------------------
    # Salary
    # -----------------------------------------

    if salary == "Not specified":

        issues.append(
            "Salary information is not provided."
        )

    return issues


# =========================================================
# SUGGESTIONS / RECOMMENDATIONS
# =========================================================

def generate_suggestions(
    issues,
    skills,
    missing_skills,
    experience,
    education,
    salary
):

    suggestions = []

    # -----------------------------------------
    # Missing Skills
    # -----------------------------------------

    if missing_skills:

        skill_text = ", ".join(
            missing_skills[:5]
        )

        suggestions.append(
            f"Consider adding relevant skills such as: {skill_text}."
        )

    # -----------------------------------------
    # Experience
    # -----------------------------------------

    if experience == "Not specified":

        suggestions.append(
            "Mention the required years of experience, "
            "for example 2+ years."
        )

    # -----------------------------------------
    # Education
    # -----------------------------------------

    if education == "Not specified":

        suggestions.append(
            "Specify the minimum educational qualification "
            "required for the position."
        )

    # -----------------------------------------
    # Salary
    # -----------------------------------------

    if salary == "Not specified":

        suggestions.append(
            "Consider adding a salary range or compensation details."
        )

    # -----------------------------------------
    # Responsibilities
    # -----------------------------------------

    if "Job responsibilities are not clearly mentioned." in issues:

        suggestions.append(
            "Add clear and measurable job responsibilities "
            "and daily tasks."
        )

    # -----------------------------------------
    # Short Description
    # -----------------------------------------

    if "Job description is too short." in issues:

        suggestions.append(
            "Add more details about the role, company, "
            "responsibilities and requirements."
        )

    # -----------------------------------------
    # No Suggestions
    # -----------------------------------------

    if not suggestions:

        suggestions.append(
            "The job description looks complete. "
            "No major improvements are required."
        )

    return suggestions


# =========================================================
# QUALITY SCORE
# =========================================================

def calculate_score(
    label,
    confidence,
    issues
):

    score = confidence * 100

    # Deduct 5 points for each issue
    score -= len(issues) * 5

    # Label adjustment
    if label == "Good":

        score += 5

    elif label == "Poor":

        score -= 5

    # Keep score between 0 and 100
    score = max(
        0,
        min(100, score)
    )

    return round(
        score,
        2
    )


# =========================================================
# MAIN ROUTE
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    result = None

    if request.method == "POST":

        job_description = request.form.get(
            "job_description",
            ""
        ).strip()

        if job_description:

            # -----------------------------------------
            # Clean Text
            # -----------------------------------------

            cleaned = clean_text(
                job_description
            )

            # -----------------------------------------
            # TF-IDF Transformation
            # -----------------------------------------

            X = vectorizer.transform(
                [cleaned]
            )

            # -----------------------------------------
            # ML Prediction
            # -----------------------------------------

            prediction = model.predict(X)[0]

            probabilities = model.predict_proba(X)[0]

            confidence = max(
                probabilities
            )

            # -----------------------------------------
            # Detect Skills
            # -----------------------------------------

            skills = detect_skills(
                job_description
            )

            # -----------------------------------------
            # Detect Missing Skills
            # -----------------------------------------

            missing_skills = detect_missing_skills(
                skills
            )

            # -----------------------------------------
            # Detect Experience
            # -----------------------------------------

            experience = detect_experience(
                job_description
            )

            # -----------------------------------------
            # Detect Education
            # -----------------------------------------

            education = detect_education(
                job_description
            )

            # -----------------------------------------
            # Detect Salary
            # -----------------------------------------

            salary = detect_salary(
                job_description
            )

            # -----------------------------------------
            # Generate Issues
            # -----------------------------------------

            issues = generate_issues(
                job_description,
                skills,
                experience,
                education,
                salary
            )

            # -----------------------------------------
            # Generate Suggestions
            # -----------------------------------------

            suggestions = generate_suggestions(
                issues,
                skills,
                missing_skills,
                experience,
                education,
                salary
            )

            # -----------------------------------------
            # Calculate Score
            # -----------------------------------------

            score = calculate_score(
                prediction,
                confidence,
                issues
            )

            # -----------------------------------------
            # Final Result
            # -----------------------------------------

            result = {

                "label": prediction,

                "score": score,

                "confidence": round(
                    confidence * 100,
                    2
                ),

                "skills": skills,

                "missing_skills": missing_skills,

                "experience": experience,

                "education": education,

                "salary": salary,

                "issues": issues,

                "suggestions": suggestions
            }

    return render_template(
        "index.html",
        result=result
    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )