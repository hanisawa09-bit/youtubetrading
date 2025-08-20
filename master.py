import subprocess
import time

# List all your trading bot scripts here
scripts = [
    "EURUSD.py",
    "GBPUSD.py",
    "GBPJPY.py",
    "USDJPY.py",
    "GOLD.py"
]

# Loop through each script and start it
for script in scripts:
    print(f"Starting {script}...")
    subprocess.Popen(["python", script])  # Starts the script without blocking
    time.sleep(5)  # Wait 5 seconds before starting the next script to reduce resource conflicts
