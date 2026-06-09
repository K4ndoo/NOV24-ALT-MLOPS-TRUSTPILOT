## Installation

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS/WSL
# ou
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

# Pour CPU
```bash
pip install torch
```

# Pour GPU CUDA
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
Remplacez `cu128` par votre version de CUDA (`cu118`, `cu121`, etc.).
