# Signal System Playground

An interactive web application for signal processing and system analysis, built with Python and Streamlit.

## Features
- Generate standard signals (sine, ramp, unit step, square, impulse, triangular) or custom Laplace-domain signals
- Analyze system properties: linearity, time-invariance, causality, stability
- Add Gaussian noise and filter with low-pass, high-pass, and band-pass filters
- Sample, reconstruct, and detect aliasing in signals
- Real-time visualization and interactive controls

## Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the URL shown in the terminal (usually http://localhost:8501)

## Laplace Input Note
- Use `*` for multiplication in Laplace expressions (e.g., `1/(2*s+1)`)

## Technologies
- Python, Streamlit, NumPy, SciPy, SymPy

---

Feel free to explore, experiment, and extend the app for your own signal processing projects! 