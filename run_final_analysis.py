"""Current revision entry; default is a no-training plan, not the legacy run."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/"code"))
from final_reproduction import main

if __name__=="__main__":
    main()
