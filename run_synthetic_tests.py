
import sys
import os
import json

# Add the guardrails directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "guardrails")))
from guardrails.input_scanners.injection_detector import detect_injection

def run_synthetic_tests():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, "guardrails", "synthetic_insurance_tests.json")
    
    if not os.path.exists(test_path):
        # Try relative to CWD if script_dir logic fails
        test_path = os.path.join(os.getcwd(), "guardrails", "synthetic_insurance_tests.json")
        
    print(f"Loading tests from: {test_path}")
    with open(test_path, "r") as f:
        tests = json.load(f)
    
    print(f"{'='*80}")
    print(f"{'Synthetic Insurance Injection Test Suite':^80}")
    print(f"{'='*80}")
    
    passed_count = 0
    for test in tests:
        name = test["name"]
        text = test["text"]
        expected = test["expected"]
        
        print(f"\n[TEST] {name}")
        print(f"Goal: {test['twisty_factor']}")
        
        results = detect_injection(text)
        is_blocked = len(results) > 0
        actual = "BLOCK" if is_blocked else "PASS"
        
        status = "✅ SUCCESS" if actual == expected else "❌ FAILED"
        print(f"Result: {actual} (Expected: {expected}) | {status}")
        
        if is_blocked:
            for res in results:
                print(f"  - Score: {res['score']:.4f}")
                print(f"  - Risk Breakdown: {res['risk_breakdown']}")
        
        if actual == expected:
            passed_count += 1
            
    print(f"\n{'='*80}")
    print(f"Summary: {passed_count}/{len(tests)} tests passed.")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_synthetic_tests()
