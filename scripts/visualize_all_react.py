import os
import subprocess
import sys
import glob
from pathlib import Path

def main():
    base_dir = "/proj/vondrick3/sruthi/discovery/DiscoveryWorld/output_dir/baselines_apr29/react/Easy_100env_gpt-4o-2024-05-13_s123"
    script_path = "/proj/vondrick3/sruthi/discovery/DiscoveryWorld/visualize_run.py"
    
    # Use glob to find all files ending with _data.json recursively
    search_pattern = os.path.join(base_dir, "**", "*_data.json")
    data_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(data_files)} files matching '*_data.json'")
    
    for i, data_file in enumerate(data_files):
        print(f"[{i+1}/{len(data_files)}] Generating visualization for: {data_file}")
        try:
            # Run the visualize_run.py script on the found data.json file
            subprocess.run([sys.executable, script_path, data_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running visualization for {data_file}: {e}")

if __name__ == "__main__":
    main()
