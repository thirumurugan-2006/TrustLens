import os
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')

scenarios = [
    "ABC Investments was founded in 2018.",
    "ABC Investments was founded in 2018. The company has 50,000 customers. Its headquarters are in Chennai.",
    "If you invest ₹5,000 today, you will receive ₹50,000 within 30 days.",
    "ABC Wealth guarantees 20% monthly returns.",
    "ABC Wealth is not government approved.",
    "According to the company, investors can earn 20% monthly returns.",
    "Is ABC Wealth really government approved?",
    "இந்த நிறுவனம் மாதம் 20% லாபம் தருவதாக உறுதி செய்கிறது.",
    "Indha company monthly 20% return guarantee pannudhu."
]

print("==================================================")
print("TRUSTLENS - CLI VERIFICATION SUITE")
print("==================================================\n")

for i, text in enumerate(scenarios, 1):
    print(f"\n==================== TEST {i} ====================")
    process = subprocess.Popen(
        ["python", "-m", "app.cli"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    out, err = process.communicate(input=f"2\n{text}\n4\n")
    
    print(out)
