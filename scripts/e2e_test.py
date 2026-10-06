import sys
import time
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path("d:/TrustLens")
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

results = {
    "project": "TrustLens",
    "test_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "environment": {},
    "summary": {"total": 0, "passed": 0, "failed": 0, "warnings": 0, "blocked": 0},
    "tests": [],
    "ocr_comparison": {},
    "performance": {},
    "issues": [],
    "recommendations": [],
    "readiness": "UNKNOWN"
}

def record_test(name, status, details=None, latency=None):
    results["tests"].append({"name": name, "status": status, "details": details, "latency": latency})
    key_map = {"PASS": "passed", "FAIL": "failed", "WARNING": "warnings", "BLOCKED": "blocked"}
    if status in key_map:
        results["summary"][key_map[status]] += 1
    results["summary"]["total"] += 1

# Environment
def get_env():
    env = {}
    env["python"] = sys.version.split(" ")[0]
    for pkg in ["easyocr", "fastapi", "pydantic", "opencv-python", "pillow", "langdetect", "pytest"]:
        try:
            out = subprocess.check_output([sys.executable, "-m", "pip", "show", pkg], text=True)
            for line in out.splitlines():
                if line.startswith("Version:"):
                    env[pkg] = line.split(" ")[1].strip()
                    break
        except Exception:
            env[pkg] = "unknown"
    results["environment"] = env

# Phase 2: Compileall
def test_compile():
    try:
        subprocess.check_call([sys.executable, "-m", "compileall", "app"])
        record_test("Python Compilation", "PASS")
    except subprocess.CalledProcessError as e:
        record_test("Python Compilation", "FAIL", str(e))
        results["issues"].append("Python compilation failed.")

# Phase 3: Pytest
def test_pytest():
    t0 = time.time()
    try:
        out = subprocess.check_output([sys.executable, "-m", "pytest", "tests/", "-v"], text=True, stderr=subprocess.STDOUT)
        passed = out.count("PASSED")
        failed = out.count("FAILED")
        latency = (time.time() - t0) * 1000
        record_test("Automated Tests", "PASS" if failed == 0 else "FAIL", f"Passed: {passed}, Failed: {failed}", latency)
        if failed > 0:
            results["issues"].append(f"{failed} automated tests failed.")
    except subprocess.CalledProcessError as e:
        latency = (time.time() - t0) * 1000
        out = e.output
        passed = out.count("PASSED")
        failed = out.count("FAILED")
        record_test("Automated Tests", "FAIL", f"Passed: {passed}, Failed: {failed}\n{out[-500:]}", latency)
        results["issues"].append(f"Pytest failed with exit code {e.returncode}.")

# Phase 4 & 5: Health & Swagger
def test_health():
    t0 = time.time()
    r = client.get("/health")
    lat = (time.time() - t0) * 1000
    if r.status_code == 200:
        record_test("Health Endpoint", "PASS", r.json(), lat)
    else:
        record_test("Health Endpoint", "FAIL", f"Code {r.status_code}", lat)
        results["issues"].append("Health endpoint failed.")
    
    r = client.get("/docs")
    if r.status_code == 200:
        record_test("Swagger Availability", "PASS")
    else:
        record_test("Swagger Availability", "FAIL")

# Phase 6: Text Input
def test_text():
    tests = [
        ("Text - English", "This investment opportunity guarantees high returns with zero risk.", "en"),
        ("Text - Tamil", "இந்த முதலீட்டு வாய்ப்பு மிகவும் நல்லது.", "ta"),
        ("Text - Tamil+English", "இந்த job மிகவும் நல்லது, apply pannunga.", "mixed"),
        ("Text - Tanglish", "Indha job romba nalla irukku, apply pannunga.", "ta"), # trans
        ("Text - Empty", "", "error"),
        ("Text - Whitespace", "     ", "error")
    ]
    for name, text, expected in tests:
        t0 = time.time()
        r = client.post("/api/input/text", json={"text": text})
        lat = (time.time() - t0) * 1000
        if expected == "error":
            if r.status_code == 400:
                record_test(name, "PASS", r.json())
            else:
                record_test(name, "FAIL", f"Expected 400, got {r.status_code}")
        else:
            if r.status_code == 200:
                data = r.json()
                lang_data = data.get("metadata", {}).get("language_analysis", {})
                
                # Check specifics
                if expected == "mixed":
                    if lang_data.get("is_code_mixed"):
                        record_test(name, "PASS", lang_data, lat)
                    else:
                        record_test(name, "WARNING", f"Expected mixed, got {lang_data}", lat)
                elif expected == "ta" and name == "Text - Tanglish":
                    if lang_data.get("transliteration_candidate"):
                        record_test(name, "PASS", lang_data, lat)
                    else:
                        record_test(name, "WARNING", f"Expected transliteration, got {lang_data}", lat)
                else:
                    if lang_data.get("primary_language") == expected:
                        record_test(name, "PASS", lang_data, lat)
                    else:
                        record_test(name, "WARNING", f"Expected {expected}, got {lang_data}", lat)
                results["performance"][name] = lat
            else:
                record_test(name, "FAIL", f"Expected 200, got {r.status_code}", lat)

# Phase 7, 8, 10, 11: OCR
def create_img(text, filename="test.png", size=(600,300), blur=False):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill="black")
    if blur:
        # manual blur via resize
        img = img.resize((size[0]//10, size[1]//10)).resize(size, Image.NEAREST)
    img.save(filename)
    return filename

def test_ocr():
    # En
    create_img("Investment Growth Plan 2024\nMonthly Return: 15%\nMinimum Investment: $500", "en.png")
    # Ta (if font works, else just PIL default which might just render ascii, but let's try)
    create_img("முதலீட்டு திட்டம் 2024\nமாத வருமானம்: 15%", "ta.png")
    # Mixed
    create_img("இந்த முதலீட்டு திட்டம் மிகவும் நல்லது\nInvestment Plan\nமாதம் ₹5,000 முதலீடு\nSIP", "mix.png")
    # Blurred
    create_img("This is blurry text", "blur.png", blur=True)
    
    tests = [
        ("OCR - English", "en.png", "PASS"),
        ("OCR - Tamil", "ta.png", "PASS"),
        ("OCR - Mixed", "mix.png", "PASS"),
        ("OCR - Unreliable", "blur.png", "unreliable")
    ]
    
    for name, fn, exp in tests:
        t0 = time.time()
        with open(fn, "rb") as f:
            r = client.post("/api/input/screenshot", files={"file": (fn, f, "image/png")})
        lat = (time.time() - t0) * 1000
        
        if r.status_code == 200:
            md = r.json().get("metadata", {})
            unreliable = md.get("ocr_unreliable", True)
            
            if exp == "unreliable":
                if unreliable:
                    record_test(name, "PASS", md, lat)
                else:
                    record_test(name, "FAIL", "Expected unreliable OCR, but was marked reliable", lat)
            else:
                if not unreliable:
                    record_test(name, "PASS", md, lat)
                else:
                    record_test(name, "WARNING", f"Marked unreliable. Conf: {md.get('ocr_confidence')}", lat)
                
            if name == "OCR - Tamil":
                results["ocr_comparison"]["tamil"] = {
                    "v1_confidence": 0.265,
                    "v2_confidence": md.get("ocr_confidence", 0),
                    "v1_reliable": False,
                    "v2_reliable": not unreliable,
                    "v1_lang": ["en"],
                    "v2_lang": md.get("ocr_selected_language_mode", []),
                    "candidates": md.get("ocr_candidates", [])
                }
            if name == "OCR - English":
                results["ocr_comparison"]["english"] = {
                    "v1_confidence": 0.882,
                    "v2_confidence": md.get("ocr_confidence", 0),
                    "v2_reliable": not unreliable
                }
            results["performance"][name] = lat
        else:
            record_test(name, "FAIL", f"HTTP {r.status_code}", lat)

def test_image_validation():
    tests = [
        ("Image Val - Empty", b"", "not-image.png", "error"),
        ("Image Val - Text", b"this is some text", "not-image.png", "error"),
    ]
    for name, content, fn, exp in tests:
        with open("tmp.png", "wb") as f: f.write(content)
        with open("tmp.png", "rb") as f:
            r = client.post("/api/input/screenshot", files={"file": (fn, f, "image/png")})
        if r.status_code == 400:
            record_test(name, "PASS", r.json())
        else:
            record_test(name, "FAIL", f"Expected 400, got {r.status_code}")

def test_reddit():
    r = client.post("/api/input/reddit", json={"url": "https://www.reddit.com/r/example/comments/abc123/example/"})
    if r.status_code in [200, 503]:
        record_test("Reddit Endpoint", "BLOCKED" if r.status_code == 503 else "PASS", r.json() if r.status_code == 200 else "API unavailable")
    else:
        record_test("Reddit Endpoint", "FAIL", f"Code {r.status_code}")

def run_all():
    get_env()
    test_compile()
    test_pytest()
    test_health()
    test_text()
    test_ocr()
    test_image_validation()
    test_reddit()
    
    # Readiness check
    if results["summary"]["failed"] == 0:
        if results["summary"]["warnings"] > 0:
            results["readiness"] = "PARTIALLY READY"
        else:
            results["readiness"] = "READY FOR NEXT STEP"
    else:
        results["readiness"] = "NOT READY - FIX REQUIRED"

    with open(PROJECT_ROOT / "reports/trustlens_input_ocr_test_report.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_all()
