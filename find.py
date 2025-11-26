import os
import re

patterns = [
    r"0\.5[^0-9]",      # 0.5 not followed by digit
    r"0\.3[^0-9]",      # 0.3
    r"0\.01[^0-9]",     # 0.01
    r"'risk_score',\s*0\.",  # risk_score': 0.X
    r"risk_profile\.get\(",  # .get() with defaults
]

directories = ['.', 'trainer', 'env', 'utils', 'policy', 'dataloader']

for directory in directories:
    if not os.path.isdir(directory):
        continue
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py'):
                continue
            
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    for line_no, line in enumerate(f, 1):
                        for pattern in patterns:
                            if re.search(pattern, line):
                                print(f"\n{filepath}:{line_no}")
                                print(f"  {line.strip()}")
            except:
                pass