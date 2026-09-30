"""
AgriSense — Crop Water Requirement Prediction System
====================================================
Advanced Tkinter app with:
  • SQLite-backed user authentication (register / login, hashed passwords)
  • Per-user prediction history
  • Dark themed dashboard with 4 tabs: Train | EDA | Predict | History
  • Pluggable sensor inputs (manual now, DTH + soil sensor later)

Run:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    python water_app.py
"""

import os
import sqlite3
import hashlib
import secrets
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==================== CONFIG ====================
CSV_PATH = "Agri_yield_prediction.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "Agri_yield_prediction.csv"
MODEL_PATH = "water_model.joblib"
DB_PATH = "agrisense.db"

NUM_COLS = ["Temperature", "Humidity", "Soil_Moisture"]
CAT_COLS = ["Region", "Season", "Crop_Type"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "Water_Required"

CROPS   = ["Maize", "Rice", "Soybean", "Wheat"]
REGIONS = ["North", "South", "East", "West"]
SEASONS = ["Kharif", "Rabi", "Zaid"]

# ==================== THEME ====================
BG          = "#0f1623"
PANEL       = "#1a2332"
PANEL_LIGHT = "#243144"
ACCENT      = "#22d3a0"
ACCENT_HOV  = "#1bb88a"
TEXT        = "#e6edf3"
MUTED       = "#8b9bb4"
DANGER      = "#ef5b5b"
WARN        = "#f5b942"
BORDER      = "#2c3a52"

FONT_H1   = ("Segoe UI", 22, "bold")
FONT_BODY = ("Segoe UI", 10)


def apply_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)

    style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    font=FONT_BODY)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT, font=FONT_BODY)
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    style.configure("H1.TLabel", background=BG, foreground=TEXT, font=FONT_H1)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))

    style.configure("TEntry", fieldbackground=PANEL_LIGHT, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    insertcolor=TEXT, padding=6)
    style.configure("TCombobox", fieldbackground=PANEL_LIGHT, background=PANEL_LIGHT,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER, padding=4)
    root.option_add("*TCombobox*Listbox.background", PANEL_LIGHT)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    style.configure("Accent.TButton", background=ACCENT, foreground="#062019",
                    font=("Segoe UI", 10, "bold"), padding=(18, 9), borderwidth=0)
    style.map("Accent.TButton", background=[("active", ACCENT_HOV), ("pressed", ACCENT_HOV)])

    style.configure("Ghost.TButton", background=PANEL, foreground=TEXT,
                    padding=(14, 8), borderwidth=1, bordercolor=BORDER)
    style.map("Ghost.TButton", background=[("active", PANEL_LIGHT)])

    style.configure("Link.TButton", background=BG, foreground=ACCENT,
                    borderwidth=0, font=("Segoe UI", 9, "underline"))
    style.map("Link.TButton", background=[("active", BG)], foreground=[("active", ACCENT_HOV)])

    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                    padding=(20, 10), borderwidth=0, font=("Segoe UI", 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", ACCENT)])

    style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                    bordercolor=BORDER, rowheight=26, font=FONT_BODY)
    style.configure("Treeview.Heading", background=PANEL_LIGHT, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"), borderwidth=0, padding=6)
    style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "#062019")])

    style.configure("TLabelframe", background=PANEL, foreground=ACCENT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"))

    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL,
        "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "axes.titlecolor": TEXT,
        "grid.color": BORDER, "savefig.facecolor": PANEL,
    })


# ==================== DATABASE ====================
def db_init():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        full_name TEXT,
        created_at TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        ts TEXT NOT NULL,
        temperature REAL, humidity REAL, soil_moisture REAL,
        region TEXT, season TEXT, crop TEXT,
        water_mm REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    con.commit(); con.close()


def hash_pw(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


def db_register(username, password, full_name):
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    try:
        salt = secrets.token_hex(16)
        cur.execute(
            "INSERT INTO users(username,password_hash,salt,full_name,created_at) VALUES (?,?,?,?,?)",
            (username, hash_pw(password, salt), salt, full_name,
             dt.datetime.now().isoformat(timespec="seconds")))
        con.commit()
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        con.close()


def db_login(username, password):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT id, password_hash, salt, full_name FROM users WHERE username=?", (username,))
    row = cur.fetchone(); con.close()
    if not row: return None
    uid, h, salt, fn = row
    return {"id": uid, "username": username, "full_name": fn} if hash_pw(password, salt) == h else None


def db_save_prediction(user_id, row, water):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""INSERT INTO predictions
        (user_id, ts, temperature, humidity, soil_moisture, region, season, crop, water_mm)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (user_id, dt.datetime.now().isoformat(timespec="seconds"),
         row["Temperature"], row["Humidity"], row["Soil_Moisture"],
         row["Region"], row["Season"], row["Crop_Type"], water))
    con.commit(); con.close()


def db_get_history(user_id, limit=200):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""SELECT ts, temperature, humidity, soil_moisture, region, season, crop, water_mm
        FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT ?""", (user_id, limit))
    rows = cur.fetchall(); con.close()
    return rows


# ==================== ML PIPELINE ====================
def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    rng = np.random.default_rng(42)

    sm = 0.6 * df["Water_Holding_Capacity"] + 0.05 * df["Rainfall"] + rng.normal(0, 5, len(df))
    df["Soil_Moisture"] = np.clip(sm, 5, 95)

    kc_map     = {"Rice": 1.20, "Maize": 0.95, "Wheat": 0.85, "Soybean": 0.90}
    season_mul = {"Zaid": 1.20, "Kharif": 1.00, "Rabi": 0.85}
    region_mul = {"South": 1.10, "West": 1.05, "North": 0.95, "East": 1.00}

    et0 = 0.0023 * (df["Temperature"] + 17.8) * np.sqrt(np.abs(df["Temperature"] - 5)) \
          * (100 - df["Humidity"]) / 100 * 8
    et0 = np.clip(et0, 1, 12)
    sm_deficit = np.clip((70 - df["Soil_Moisture"]) / 50, 0.2, 1.5)

    water = (et0
             * df["Crop_Type"].map(kc_map)
             * df["Season"].map(season_mul)
             * df["Region"].map(region_mul)
             * sm_deficit) + rng.normal(0, 0.3, len(df))
    df[TARGET] = np.clip(water, 0.5, 20).round(2)
    return df[FEATURES + [TARGET]]


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def train_models(df):
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    best_name, best_r2, best_pipe = None, -np.inf, None
    for name, mdl in candidates.items():
        pipe = build_pipeline(mdl); pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = {
            "MAE":  mean_absolute_error(yte, pred),
            "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
            "R2":   r2_score(yte, pred),
            "y_test": yte, "y_pred": pred,
        }
        if results[name]["R2"] > best_r2:
            best_r2, best_name, best_pipe = results[name]["R2"], name, pipe

    joblib.dump(best_pipe, MODEL_PATH)
    return results, best_name


# ==================== AUTH WINDOW ====================
class AuthWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AgriSense · Sign In")
        self.geometry("900x560")
        self.minsize(900, 560)
        apply_theme(self)
        self.user = None

        left = tk.Frame(self, bg=PANEL, width=420)
        left.pack(side="left", fill="both"); left.pack_propagate(False)
        tk.Label(left, text="🌱", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI Emoji", 72)).pack(pady=(80, 10))
        tk.Label(left, text="AgriSense", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 28, "bold")).pack()
        tk.Label(left, text="Smart Irrigation Intelligence", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 11)).pack(pady=(2, 30))
        for line in ["• Real-time water requirement forecasting",
                     "• ML-driven crop and soil analytics",
                     "• Sensor-ready: DTH + soil moisture",
                     "• Secure per-user prediction history"]:
            tk.Label(left, text=line, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 10), anchor="w").pack(fill="x", padx=50, pady=3)

        self.right = tk.Frame(self, bg=BG); self.right.pack(side="right", fill="both", expand=True)
        self._show_login()

    def _clear_right(self):
        for w in self.right.winfo_children(): w.destroy()

    def _show_login(self):
        self._clear_right()
        wrap = tk.Frame(self.right, bg=BG); wrap.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(wrap, text="Welcome back", style="H1.TLabel").pack(anchor="w")
        ttk.Label(wrap, text="Sign in to your AgriSense account",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 22))

        ttk.Label(wrap, text="Username").pack(anchor="w")
        u = ttk.Entry(wrap, width=34); u.pack(pady=(4, 12)); u.focus()
        ttk.Label(wrap, text="Password").pack(anchor="w")
        p = ttk.Entry(wrap, width=34, show="•"); p.pack(pady=(4, 18))

        msg = ttk.Label(wrap, text="", foreground=DANGER, background=BG); msg.pack(anchor="w")

        def do_login(*_):
            user = db_login(u.get().strip(), p.get())
            if user:
                self.user = user; self.destroy()
            else:
                msg.config(text="✗ Invalid username or password.")
        p.bind("<Return>", do_login); u.bind("<Return>", do_login)

        ttk.Button(wrap, text="Sign In", style="Accent.TButton", command=do_login)\
            .pack(fill="x", pady=(6, 10))

        row = tk.Frame(wrap, bg=BG); row.pack(fill="x")
        ttk.Label(row, text="New here?", style="Muted.TLabel").pack(side="left")
        ttk.Button(row, text="Create an account", style="Link.TButton",
                   command=self._show_register).pack(side="left", padx=4)

    def _show_register(self):
        self._clear_right()
        wrap = tk.Frame(self.right, bg=BG); wrap.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(wrap, text="Create account", style="H1.TLabel").pack(anchor="w")
        ttk.Label(wrap, text="Start predicting in under a minute",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 22))

        ttk.Label(wrap, text="Full name").pack(anchor="w")
        fn = ttk.Entry(wrap, width=34); fn.pack(pady=(4, 10))
        ttk.Label(wrap, text="Username").pack(anchor="w")
        u  = ttk.Entry(wrap, width=34); u.pack(pady=(4, 10))
        ttk.Label(wrap, text="Password").pack(anchor="w")
        p  = ttk.Entry(wrap, width=34, show="•"); p.pack(pady=(4, 18))

        msg = ttk.Label(wrap, text="", background=BG); msg.pack(anchor="w")

        def do_register():
            ok, m = db_register(u.get().strip(), p.get(), fn.get().strip())
            msg.config(text=("✓ " if ok else "✗ ") + m,
                       foreground=(ACCENT if ok else DANGER))
            if ok: self.after(900, self._show_login)

        ttk.Button(wrap, text="Create Account", style="Accent.TButton",
                   command=do_register).pack(fill="x", pady=(6, 10))

        row = tk.Frame(wrap, bg=BG); row.pack(fill="x")
        ttk.Label(row, text="Already have an account?", style="Muted.TLabel").pack(side="left")
        ttk.Button(row, text="Sign in", style="Link.TButton",
                   command=self._show_login).pack(side="left", padx=4)


# ==================== DASHBOARD ====================
class Dashboard(tk.Tk):
    def __init__(self, user):
        super().__init__()
        self.user = user
        self.title("AgriSense · Dashboard")
        self.geometry("1200x780")
        self.minsize(1100, 720)
        apply_theme(self)
        self.df = None; self.results = None; self.best_name = None
        self._build_header()
        self._build_tabs()
        self._build_status_bar()

    def _build_header(self):
        bar = tk.Frame(self, bg=PANEL, height=64); bar.pack(fill="x"); bar.pack_propagate(False)
        tk.Label(bar, text="🌱  AgriSense", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=20)
        tk.Label(bar, text="Crop Water Requirement Prediction System",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(side="left")

        right = tk.Frame(bar, bg=PANEL); right.pack(side="right", padx=20)
        name = self.user["full_name"] or self.user["username"]
        tk.Label(right, text=f"  {name}", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(right, text=f"@{self.user['username']}", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(6, 14))
        ttk.Button(right, text="Logout", style="Ghost.TButton", command=self._logout).pack(side="left")

    def _logout(self):
        if messagebox.askyesno("Logout", "Sign out of AgriSense?"):
            self.destroy(); main()

    def _build_tabs(self):
        outer = tk.Frame(self, bg=BG); outer.pack(fill="both", expand=True, padx=14, pady=10)
        self.nb = ttk.Notebook(outer); self.nb.pack(fill="both", expand=True)
        self.tab_train  = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_train,  text="  Model Training  ")
        self.tab_eda    = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_eda,    text="  EDA Analytics  ")
        self.tab_pred   = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_pred,   text="  Predict  ")
        self.tab_hist   = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_hist,   text="  History  ")
        self._build_train_tab(); self._build_eda_tab(); self._build_pred_tab(); self._build_hist_tab()

    def _build_status_bar(self):
        self.status = tk.Label(self, text="● Ready", bg=PANEL, fg=ACCENT,
                               anchor="w", padx=14, font=("Segoe UI", 9))
        self.status.pack(side="bottom", fill="x")

    def set_status(self, text, color=ACCENT):
        self.status.config(text="● " + text, fg=color); self.update_idletasks()

    def _kpi(self, parent, title, value, accent=ACCENT):
        card = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(card, text=title.upper(), bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(12, 0))
        lbl = tk.Label(card, text=value, bg=PANEL, fg=accent, font=("Segoe UI", 20, "bold"))
        lbl.pack(anchor="w", padx=16, pady=(2, 14))
        card.value_label = lbl
        return card

    # ----- Train -----
    def _build_train_tab(self):
        top = tk.Frame(self.tab_train, bg=BG); top.pack(fill="x", pady=(12, 8), padx=4)
        ttk.Label(top, text="Model Training", style="H1.TLabel").pack(side="left")
        ttk.Button(top, text="🚀  Train Models", style="Accent.TButton",
                   command=self.run_training).pack(side="right")
        ttk.Label(self.tab_train,
                  text="Trains Linear Regression, Random Forest and Gradient Boosting. "
                       "The best model is saved to disk for predictions.",
                  style="Muted.TLabel").pack(anchor="w", padx=4, pady=(0, 10))

        kpi_row = tk.Frame(self.tab_train, bg=BG); kpi_row.pack(fill="x", pady=6)
        self.k_rows = self._kpi(kpi_row, "Rows",       "—")
        self.k_feat = self._kpi(kpi_row, "Features",   str(len(FEATURES)))
        self.k_best = self._kpi(kpi_row, "Best Model", "—")
        self.k_r2   = self._kpi(kpi_row, "Best R²",    "—")
        for w in (self.k_rows, self.k_feat, self.k_best, self.k_r2):
            w.pack(side="left", fill="both", expand=True, padx=6, ipady=2)

        body = tk.Frame(self.tab_train, bg=BG); body.pack(fill="both", expand=True, pady=10)
        left = ttk.LabelFrame(body, text=" Model Comparison ")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        cols = ("model", "mae", "rmse", "r2")
        self.train_tree = ttk.Treeview(left, columns=cols, show="headings", height=8)
        for c, t, w in [("model","Model",170), ("mae","MAE",90),
                        ("rmse","RMSE",90), ("r2","R²",90)]:
            self.train_tree.heading(c, text=t); self.train_tree.column(c, width=w, anchor="center")
        self.train_tree.pack(fill="both", expand=True, padx=10, pady=10)

        self.plot_frame = ttk.LabelFrame(body, text=" Actual vs Predicted ")
        self.plot_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

    def run_training(self):
        try:
            self.set_status("Loading dataset…", WARN)
            self.df = load_and_prepare()
            self.set_status("Training models…", WARN)
            self.results, self.best_name = train_models(self.df)

            self.k_rows.value_label.config(text=f"{len(self.df):,}")
            self.k_best.value_label.config(text=self.best_name)
            self.k_r2.value_label.config(text=f"{self.results[self.best_name]['R2']:.3f}")

            for r in self.train_tree.get_children(): self.train_tree.delete(r)
            for name, r in self.results.items():
                tag = "best" if name == self.best_name else ""
                self.train_tree.insert("", "end",
                    values=(name, f"{r['MAE']:.3f}", f"{r['RMSE']:.3f}", f"{r['R2']:.3f}"),
                    tags=(tag,))
            self.train_tree.tag_configure("best", foreground=ACCENT)

            for w in self.plot_frame.winfo_children(): w.destroy()
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            r = self.results[self.best_name]
            ax.scatter(r["y_test"], r["y_pred"], alpha=0.4, s=10, color=ACCENT, edgecolor="none")
            lims = [r["y_test"].min(), r["y_test"].max()]
            ax.plot(lims, lims, "--", color=DANGER, lw=1.5)
            ax.set_xlabel("Actual (mm/day)"); ax.set_ylabel("Predicted (mm/day)")
            ax.set_title(f"{self.best_name}   R² = {r['R2']:.3f}")
            fig.tight_layout()
            FigureCanvasTkAgg(fig, master=self.plot_frame).get_tk_widget()\
                .pack(fill="both", expand=True, padx=8, pady=8)
            self.set_status(f"Training complete. Best model: {self.best_name} (saved to {MODEL_PATH})")
        except Exception as e:
            self.set_status("Training failed.", DANGER)
            messagebox.showerror("Training error", str(e))

    # ----- EDA -----
    def _build_eda_tab(self):
        top = tk.Frame(self.tab_eda, bg=BG); top.pack(fill="x", pady=(12, 8), padx=4)
        ttk.Label(top, text="Exploratory Data Analysis", style="H1.TLabel").pack(side="left")
        ttk.Button(top, text="📊  Generate Plots", style="Accent.TButton",
                   command=self.run_eda).pack(side="right")
        ttk.Label(self.tab_eda,
                  text="Distribution, group comparisons, scatter relationships and correlation heatmap.",
                  style="Muted.TLabel").pack(anchor="w", padx=4, pady=(0, 10))
        self.eda_frame = tk.Frame(self.tab_eda, bg=BG); self.eda_frame.pack(fill="both", expand=True)

    def run_eda(self):
        try:
            self.set_status("Building EDA plots…", WARN)
            if self.df is None: self.df = load_and_prepare()
            for w in self.eda_frame.winfo_children(): w.destroy()

            fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
            axes[0,0].hist(self.df[TARGET], bins=40, color=ACCENT, edgecolor=BG)
            axes[0,0].set_title("Water Required Distribution"); axes[0,0].set_xlabel("mm/day")

            sns.boxplot(data=self.df, x="Crop_Type", y=TARGET, ax=axes[0,1],
                        order=sorted(self.df["Crop_Type"].unique()), color=ACCENT)
            axes[0,1].set_title("Water by Crop"); axes[0,1].set_xlabel("")

            sns.boxplot(data=self.df, x="Season", y=TARGET, ax=axes[0,2],
                        order=SEASONS, color=WARN)
            axes[0,2].set_title("Water by Season"); axes[0,2].set_xlabel("")

            axes[1,0].scatter(self.df["Temperature"], self.df[TARGET],
                              alpha=0.25, s=6, color=DANGER, edgecolor="none")
            axes[1,0].set_xlabel("Temperature (°C)"); axes[1,0].set_ylabel("Water (mm/day)")
            axes[1,0].set_title("Temperature vs Water")

            axes[1,1].scatter(self.df["Soil_Moisture"], self.df[TARGET],
                              alpha=0.25, s=6, color=ACCENT, edgecolor="none")
            axes[1,1].set_xlabel("Soil Moisture (%)"); axes[1,1].set_ylabel("Water (mm/day)")
            axes[1,1].set_title("Soil Moisture vs Water")

            corr = self.df[NUM_COLS + [TARGET]].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako",
                        ax=axes[1,2], cbar=False, annot_kws={"color": TEXT})
            axes[1,2].set_title("Feature Correlations")

            fig.tight_layout()
            FigureCanvasTkAgg(fig, master=self.eda_frame).get_tk_widget()\
                .pack(fill="both", expand=True, padx=4, pady=4)
            self.set_status("EDA plots generated.")
        except Exception as e:
            self.set_status("EDA failed.", DANGER)
            messagebox.showerror("EDA error", str(e))

    # ----- Predict -----
    def _build_pred_tab(self):
        top = tk.Frame(self.tab_pred, bg=BG); top.pack(fill="x", pady=(12, 8), padx=4)
        ttk.Label(top, text="Water Requirement Prediction", style="H1.TLabel").pack(side="left")
        ttk.Label(self.tab_pred,
                  text="Sensors will replace these inputs later (DTH → temp/humidity, soil sensor → moisture).",
                  style="Muted.TLabel").pack(anchor="w", padx=4, pady=(0, 12))

        body = tk.Frame(self.tab_pred, bg=BG); body.pack(fill="both", expand=True)
        form = ttk.LabelFrame(body, text=" Sensor & Field Inputs ")
        form.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=4)

        self.entries = {}
        defaults = [("Temperature (°C)", "Temperature", "28"),
                    ("Humidity (%)",     "Humidity",    "60"),
                    ("Soil Moisture / Wetness (%)", "Soil_Moisture", "40")]
        for i, (lbl, key, d) in enumerate(defaults):
            ttk.Label(form, text=lbl, style="Panel.TLabel")\
                .grid(row=i, column=0, sticky="w", padx=18, pady=10)
            e = ttk.Entry(form, width=22); e.insert(0, d)
            e.grid(row=i, column=1, padx=18, pady=10, sticky="w")
            self.entries[key] = e

        ttk.Label(form, text="Region", style="Panel.TLabel")\
            .grid(row=3, column=0, sticky="w", padx=18, pady=10)
        self.region_cb = ttk.Combobox(form, values=REGIONS, state="readonly", width=20)
        self.region_cb.current(0); self.region_cb.grid(row=3, column=1, padx=18, pady=10, sticky="w")

        ttk.Label(form, text="Season", style="Panel.TLabel")\
            .grid(row=4, column=0, sticky="w", padx=18, pady=10)
        self.season_cb = ttk.Combobox(form, values=SEASONS, state="readonly", width=20)
        self.season_cb.current(0); self.season_cb.grid(row=4, column=1, padx=18, pady=10, sticky="w")

        ttk.Label(form, text="Crop", style="Panel.TLabel")\
            .grid(row=5, column=0, sticky="w", padx=18, pady=10)
        self.crop_cb = ttk.Combobox(form, values=CROPS, state="readonly", width=20)
        self.crop_cb.current(0); self.crop_cb.grid(row=5, column=1, padx=18, pady=10, sticky="w")

        ttk.Button(form, text="💧  Predict Water Requirement", style="Accent.TButton",
                   command=self.predict).grid(row=6, column=0, columnspan=2,
                                              padx=18, pady=(18, 18), sticky="we")

        right = ttk.LabelFrame(body, text=" Prediction Result ")
        right.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=4)
        self.result_big = tk.Label(right, text="—", bg=PANEL, fg=ACCENT,
                                   font=("Segoe UI", 48, "bold"))
        self.result_big.pack(pady=(40, 0))
        self.result_unit = tk.Label(right, text="mm / day", bg=PANEL, fg=MUTED,
                                    font=("Segoe UI", 12))
        self.result_unit.pack()
        self.result_sub = tk.Label(right, text="Run a prediction to see results",
                                   bg=PANEL, fg=MUTED, font=("Segoe UI", 10))
        self.result_sub.pack(pady=(18, 8))
        self.result_advice = tk.Label(right, text="", bg=PANEL, fg=WARN,
                                      font=("Segoe UI", 10, "italic"),
                                      wraplength=380, justify="center")
        self.result_advice.pack(pady=(4, 30), padx=20)

    def predict(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning("No model", "Please train the model first (Model Training tab).")
            return
        try:
            row = {
                "Temperature":   float(self.entries["Temperature"].get()),
                "Humidity":      float(self.entries["Humidity"].get()),
                "Soil_Moisture": float(self.entries["Soil_Moisture"].get()),
                "Region":        self.region_cb.get(),
                "Season":        self.season_cb.get(),
                "Crop_Type":     self.crop_cb.get(),
            }
            pipe = joblib.load(MODEL_PATH)
            water = float(pipe.predict(pd.DataFrame([row]))[0])

            self.result_big.config(text=f"{water:.2f}")
            self.result_sub.config(text=f"≈ {water*10:.1f} m³ per hectare per day")

            if water < 1.5:
                advice, color = "Low demand — skip or minimal irrigation today.", ACCENT
            elif water < 4:
                advice, color = "Moderate demand — schedule standard irrigation.", WARN
            else:
                advice, color = "High demand — prioritise irrigation, check coverage.", DANGER
            self.result_advice.config(text=advice, fg=color)

            db_save_prediction(self.user["id"], row, water)
            self._reload_history()
            self.set_status(f"Predicted {water:.2f} mm/day · saved to history.")
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

    # ----- History -----
    def _build_hist_tab(self):
        top = tk.Frame(self.tab_hist, bg=BG); top.pack(fill="x", pady=(12, 8), padx=4)
        ttk.Label(top, text="Prediction History", style="H1.TLabel").pack(side="left")
        ttk.Button(top, text="↻  Refresh", style="Ghost.TButton",
                   command=self._reload_history).pack(side="right")
        ttk.Label(self.tab_hist, text="Your last 200 predictions.",
                  style="Muted.TLabel").pack(anchor="w", padx=4, pady=(0, 10))

        cols = ("ts","temp","hum","sm","region","season","crop","water")
        self.hist_tree = ttk.Treeview(self.tab_hist, columns=cols, show="headings", height=22)
        for c, t, w in [("ts","Timestamp",160),("temp","Temp °C",80),("hum","Humidity %",90),
                        ("sm","Soil M %",80),("region","Region",90),("season","Season",90),
                        ("crop","Crop",90),("water","Water mm/day",110)]:
            self.hist_tree.heading(c, text=t); self.hist_tree.column(c, width=w, anchor="center")
        self.hist_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._reload_history()

    def _reload_history(self):
        if not hasattr(self, "hist_tree"): return
        for r in self.hist_tree.get_children(): self.hist_tree.delete(r)
        for row in db_get_history(self.user["id"]):
            ts, t, h, sm, reg, sea, crop, w = row
            self.hist_tree.insert("", "end", values=(ts, f"{t:.1f}", f"{h:.1f}", f"{sm:.1f}",
                                                     reg, sea, crop, f"{w:.2f}"))


# ==================== ENTRY POINT ====================
def main():
    db_init()
    auth = AuthWindow(); auth.mainloop()
    if auth.user:
        Dashboard(auth.user).mainloop()


if __name__ == "__main__":
    main()