"""Feature selector GUI for VCBench.

Saves a JSON file with the selected feature names.
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from paths import CONFIG_DIR

BASELINE_FEATURES = [
    ("top_university", "Any education with QS ranking â‰¤ 50."),
    ("has_phd", "Has a PhD or doctorate."),
    ("has_mba", "Has an MBA or business-oriented master's."),
    ("stem_degree", "Degree field includes STEM keywords."),
    ("prior_exit", "Any IPO or acquisition (>= 1 total)."),
    ("multiple_exits", "Two or more IPOs/acquisitions."),
    ("senior_leadership", "Held senior role (CEO/CTO/VP, etc)."),
    ("large_company_exp", "Worked at company size â‰¥ 1000."),
    ("startup_exp", "Worked at company size 1â€“50."),
    ("long_experience", "Total career duration > 5 years."),
    ("serial_founder", "Founder roles in 2+ jobs."),
    ("technical_role", "Technical job roles (engineer/CTO/etc)."),
    ("many_prior_roles", "4+ roles in job history."),
    ("short_tenure_pattern", "Majority of roles are short (<2 years)."),
    ("industry_match", "Job industry matches startup industry."),
]

CUSTOM_FEATURES = [
    ("qs_top_25", "Any education with QS ranking 1-25."),
    ("qs_top_50", "Any education with QS ranking 1-50."),
    ("qs_top_100", "Any education with QS ranking 1-100."),
    ("qs_top_200", "Any education with QS ranking 1-200."),
    ("qs_inverse_best", "1 / best (lowest) QS ranking; missing = 0."),
    ("prior_ipos", "At least one IPO."),
    ("prior_acquisitions", "At least one acquisition."),
    ("large_company_years", "Total years at companies with >= 1000 employees."),
    ("experience_duration", "Total years across all jobs."),
    ("technical_experience_duration", "Total years in technical roles."),
]


class FeatureSelectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VCBench Feature Selector")
        self.geometry("760x700")

        self.vars: dict[str, tk.BooleanVar] = {}

        # Sticky top controls
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=12, pady=8)

        ttk.Button(top, text="Select All", command=self.select_all).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(top, text="Clear All", command=self.clear_all).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(top, text="Save JSON", command=self.save_json).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(top, text="Load JSON", command=self.load_json).pack(
            side=tk.LEFT, padx=6
        )

        # Scrollable area
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable = ttk.Frame(canvas)

        self.scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_section("Baseline (15)", BASELINE_FEATURES, self.scrollable)
        self._build_section("Custom Registry", CUSTOM_FEATURES, self.scrollable)
        self._build_llm_controls(self.scrollable)

    def _build_section(self, title: str, items: list[tuple[str, str]], parent) -> None:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)

        # Section select all
        section_var = tk.BooleanVar(value=False)
        section_chk = ttk.Checkbutton(
            frame,
            text="Select all in section",
            variable=section_var,
            command=lambda v=section_var, it=items: self._toggle_section(v, it),
        )
        section_chk.pack(anchor="w", padx=8, pady=4)

        for name, desc in items:
            var = tk.BooleanVar(value=False)
            self.vars[name] = var
            chk = ttk.Checkbutton(frame, text=f"{name} â€” {desc}", variable=var)
            chk.pack(anchor="w", padx=8, pady=2)

    def _build_llm_controls(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="LLM Features")
        frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)

        self.use_llm_var = tk.BooleanVar(value=False)
        use_chk = ttk.Checkbutton(frame, text="Use LLM features", variable=self.use_llm_var)
        use_chk.pack(anchor="w", padx=8, pady=4)

        count_row = ttk.Frame(frame)
        count_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(count_row, text="LLM features count").pack(side=tk.LEFT)
        self.llm_count_var = tk.IntVar(value=8)
        count_entry = ttk.Entry(count_row, textvariable=self.llm_count_var, width=8)
        count_entry.pack(side=tk.LEFT, padx=6)

    def _toggle_section(self, section_var: tk.BooleanVar, items) -> None:
        for name, _ in items:
            self.vars[name].set(section_var.get())

    def select_all(self) -> None:
        for v in self.vars.values():
            v.set(True)

    def clear_all(self) -> None:
        for v in self.vars.values():
            v.set(False)

    def save_json(self) -> None:
        selected = [name for name, v in self.vars.items() if v.get()]
        default_path = CONFIG_DIR / "features.json"
        path = filedialog.asksaveasfilename(
            title="Save feature selection",
            defaultextension=".json",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        payload = {}
        try:
            if Path(path).exists():
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    if not isinstance(payload, dict):
                        payload = {}
        except Exception:
            payload = {}
        payload.update(
            {
                "features": selected,
                "use_llm": bool(self.use_llm_var.get()),
                "llm_n_features": int(self.llm_count_var.get()),
            }
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        messagebox.showinfo("Saved", f"Saved {len(selected)} features to:\n{path}")

    def load_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load feature selection",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        selected = set(data.get("features", []))
        for name, v in self.vars.items():
            v.set(name in selected)
        self.use_llm_var.set(bool(data.get("use_llm", False)))
        try:
            self.llm_count_var.set(int(data.get("llm_n_features", 8)))
        except Exception:
            self.llm_count_var.set(8)
        messagebox.showinfo("Loaded", f"Loaded {len(selected)} features from:\n{path}")


if __name__ == "__main__":
    app = FeatureSelectorApp()
    app.mainloop()
