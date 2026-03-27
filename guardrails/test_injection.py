
import sys
import os
import json

sys.path.append(os.getcwd())

from input_scanners.injection_detector import detect_injection, calculate_ml_score

def test_text(label, content):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Content length: {len(content)} characters")
    print("-" * 40)
    
    issues = detect_injection(content)
    
    if issues:
        print("[BLOCK] INJECTION DETECTED!")
        print(json.dumps(issues, indent=2))
    else:
        print("[PASS] No injection detected")
    return issues

def test_raw_score(label, content):
    print(f"\nRAW SCORE TEST: {label}")
    score, block, segment = calculate_ml_score(content)
    print(f"ML Score: {score:.4f}")
    print(f"Hard Block: {block}")
    print(f"Segment: {segment[:100]}...")

if __name__ == "__main__":
    # Test with a formal insurance document (Safe context)
    import os
    txt_path = os.path.join(os.path.dirname(__file__), "test.txt")
    with open(txt_path, 'r', encoding='utf-8') as f:
        doc_content = f.read()
    test_text("Legitimate insurance document (test.txt)", doc_content)


    print(f"\n{'='*60}")
    print("All tests completed.")
