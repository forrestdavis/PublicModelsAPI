conda create --name mapi python=3.9
conda activate mapi
pip install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r MACrequirements.txt
