# ⚙️ Full Installation Guide

This guide sets up the complete Python and R environment required for this project, including `oblique.tree` integration via `rpy2`.

> ❗ **Do not use pip** for package management.

---

## 1. 🧱 Create Conda Environment

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if not already installed.

Create and activate the environment in your CMD:

```text
conda create -n oblique-env python=3.11
conda activate oblique-env
```

---

### 2. 📦 Install All Python Dependencies

Install all required Python packages from `conda-forge`. This includes core numerical libraries, plotting tools, interactive widgets, and experimental design modules:

```text
 conda install -c conda-forge ipython ipywidgets notebook ipykernel matplotlib numpy pandas pydoe2 scipy seaborn scikit-learn shapely tqdm
```

> 🧪 These exact versions match the development environment used to validate and benchmark all models.

---

### 3. 📐 Install R (Base Environment)

You must have **R installed separately** on your system. The latest version from [CRAN](https://cran.r-project.org/) is recommended (e.g., R 4.4.x).

- **Windows**, use the `.exe` installer provided on CRAN.
- **macOS**: use Homebrew (brew install r)
- **Linux**: use your package manager (sudo apt install r-base)

No manual changes to `PATH` are needed if installed normally.

---

### 4. 📜 Install Required R Packages Automatically

This project includes a Python script that uses `Rscript` to check and install the required R packages (`tree` and `oblique.tree`). 

To run the script, click the green button below, or execute the following command in your terminal from this location:

```bash
python src/install_oblique_tree_r.py
```

This script will:

- Detect your `Rscript` binary across Windows/macOS/Linux.
- Print the R version in use.
- Install `tree` from CRAN if not already installed.
- Install the archived `oblique.tree` package from the CRAN archive (source install).
- Confirm installation success or clearly report errors.

> 🛠 If R is installed in a non-standard directory, open the script and set the `manual_rscript_path` variable to your full Rscript path. Example:
>
> ```python
> manual_rscript_path = r"C:\Custom\Path\To\Rscript.exe"
> ```

---

### 5. 🧠 Optional: Configure Conda Environment in PyCharm

If you use **PyCharm**, follow these steps to attach the Conda environment:

1. Open your project.
2. Go to `File > Settings > Project: <your project> > Python Interpreter`.
3. Click the ⚙️ icon > "Add".
4. Choose "Conda Environment" > "Existing Environment".
5. Browse to the environment path (e.g., `~/miniconda3/envs/oblique-env`).
6. Apply and set it as your project interpreter.
