import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ---------------- CONFIG ----------------
CSV_PATH = "Agri_yield_prediction.csv"
MODEL_PATH = "water_model.joblib"

NUM_COLS = ["Temperature", "Humidity", "Soil_Moisture"]
CAT_COLS = ["Region", "Season", "Crop_Type"]

FEATURES = NUM_COLS + CAT_COLS
TARGET = "Water_Required"

REGIONS = ["North", "South", "East", "West"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
CROPS = ["Rice", "Maize", "Wheat", "Soybean"]


# ---------------- DATA PREP ----------------
def load_dataset():
    df = pd.read_csv(CSV_PATH)

    rng = np.random.default_rng(42)

    df["Soil_Moisture"] = np.clip(
        0.6 * df["Water_Holding_Capacity"]
        + 0.05 * df["Rainfall"]
        + rng.normal(0, 5, len(df)),
        5,
        95
    )

    kc = {
        "Rice": 1.2,
        "Maize": 0.95,
        "Wheat": 0.85,
        "Soybean": 0.90
    }

    season_mul = {
        "Kharif": 1.0,
        "Rabi": 0.85,
        "Zaid": 1.2
    }

    region_mul = {
        "North": 0.95,
        "South": 1.10,
        "East": 1.0,
        "West": 1.05
    }

    et0 = (
        0.0023
        * (df["Temperature"] + 17.8)
        * np.sqrt(abs(df["Temperature"] - 5))
        * (100 - df["Humidity"]) / 100
        * 8
    )

    sm_deficit = np.clip((70 - df["Soil_Moisture"]) / 50, 0.2, 1.5)

    df[TARGET] = (
        et0
        * df["Crop_Type"].map(kc)
        * df["Season"].map(season_mul)
        * df["Region"].map(region_mul)
        * sm_deficit
    )

    df[TARGET] = np.clip(df[TARGET], 0.5, 20)

    return df


# ---------------- TRAIN ----------------
def train_model():

    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
    ])

    model = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    return True


# ---------------- GUI ----------------
class WaterApp(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title("AgriSense")
        self.geometry("700x500")

        title = tk.Label(
            self,
            text="Crop Water Requirement Prediction",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=10)

        ttk.Button(
            self,
            text="Train Model",
            command=self.train_clicked
        ).pack(pady=10)

        form = tk.Frame(self)
        form.pack(pady=10)

        self.temp = tk.Entry(form)
        self.hum = tk.Entry(form)
        self.moist = tk.Entry(form)

        tk.Label(form, text="Temperature").grid(row=0, column=0)
        self.temp.grid(row=0, column=1)

        tk.Label(form, text="Humidity").grid(row=1, column=0)
        self.hum.grid(row=1, column=1)

        tk.Label(form, text="Soil Moisture").grid(row=2, column=0)
        self.moist.grid(row=2, column=1)

        self.region = ttk.Combobox(
            form,
            values=REGIONS,
            state="readonly"
        )
        self.region.grid(row=3, column=1)
        self.region.current(0)

        tk.Label(form, text="Region").grid(row=3, column=0)

        self.season = ttk.Combobox(
            form,
            values=SEASONS,
            state="readonly"
        )
        self.season.grid(row=4, column=1)
        self.season.current(0)

        tk.Label(form, text="Season").grid(row=4, column=0)

        self.crop = ttk.Combobox(
            form,
            values=CROPS,
            state="readonly"
        )
        self.crop.grid(row=5, column=1)
        self.crop.current(0)

        tk.Label(form, text="Crop").grid(row=5, column=0)

        ttk.Button(
            self,
            text="Predict Water Requirement",
            command=self.predict
        ).pack(pady=20)

        self.result = tk.Label(
            self,
            text="Prediction will appear here",
            font=("Arial", 16, "bold"),
            fg="blue"
        )
        self.result.pack()

    def train_clicked(self):

        try:
            train_model()
            messagebox.showinfo(
                "Success",
                "Model trained successfully"
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e)
            )

    def predict(self):

        try:

            model = joblib.load(MODEL_PATH)

            data = pd.DataFrame([{
                "Temperature": float(self.temp.get()),
                "Humidity": float(self.hum.get()),
                "Soil_Moisture": float(self.moist.get()),
                "Region": self.region.get(),
                "Season": self.season.get(),
                "Crop_Type": self.crop.get()
            }])

            prediction = model.predict(data)[0]

            self.result.config(
                text=f"{prediction:.2f} mm/day"
            )

        except FileNotFoundError:
            messagebox.showwarning(
                "Train Model",
                "Please train the model first."
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e)
            )


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = WaterApp()
    app.mainloop()