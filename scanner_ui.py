#!/usr/bin/env python3
"""
scanner_ui.py

Drag-and-drop GUI for stitch_remove_scanline.py.
Drop scan 1 and scan 2, choose DPI, click Run.
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stitch_remove_scanline as pipeline

# ── Colours ──────────────────────────────────────────────────────────────────
BG          = "#1C1C1E"
PANEL       = "#2C2C2E"
BORDER      = "#3A3A3C"
ACCENT      = "#0A84FF"
TEXT        = "#F2F2F7"
SUBTEXT     = "#8E8E93"
DROP_BG     = "#232325"
THUMB_W     = 280
THUMB_H     = 370


# ── Drop zone ─────────────────────────────────────────────────────────────────
class DropZone(tk.Frame):
    def __init__(self, parent, label, on_file, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.on_file = on_file
        self.path    = None
        self._photo  = None

        self.canvas = tk.Canvas(self, width=THUMB_W, height=THUMB_H,
                                bg=DROP_BG, highlightthickness=2,
                                highlightbackground=BORDER, cursor="hand2")
        self.canvas.pack(padx=14, pady=(14, 6))

        tk.Label(self, text=label, bg=PANEL, fg=TEXT,
                 font=("Helvetica", 13, "bold")).pack()

        self.sub = tk.Label(self, text="Drop image here", bg=PANEL, fg=SUBTEXT,
                            font=("Helvetica", 11))
        self.sub.pack(pady=(2, 0))

        tk.Button(self, text="Browse…", bg=PANEL, fg=ACCENT,
                  font=("Helvetica", 11), bd=0, cursor="hand2",
                  activebackground=PANEL, activeforeground=ACCENT,
                  command=self._browse).pack(pady=(4, 14))

        self._placeholder()

        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind("<<Drop>>",      self._on_drop)
        self.canvas.dnd_bind("<<DragEnter>>", lambda e: self.canvas.config(highlightbackground=ACCENT))
        self.canvas.dnd_bind("<<DragLeave>>", lambda e: self.canvas.config(highlightbackground=BORDER))

    def _placeholder(self):
        self.canvas.delete("all")
        w, h = THUMB_W, THUMB_H
        self.canvas.create_rectangle(10, 10, w-10, h-10,
                                     outline=BORDER, dash=(6, 4), width=2)
        self.canvas.create_text(w//2, h//2 - 16, text="⬇", fill=BORDER,
                                font=("Helvetica", 36))
        self.canvas.create_text(w//2, h//2 + 28, text="drop here or browse",
                                fill=SUBTEXT, font=("Helvetica", 11))

    def _parse(self, raw):
        p = raw.strip()
        return p[1:-1] if p.startswith("{") and p.endswith("}") else p

    def _on_drop(self, event):
        self.canvas.config(highlightbackground=BORDER)
        self.set_file(self._parse(event.data))

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All", "*.*")])
        if p:
            self.set_file(p)

    def set_file(self, path):
        self.path = path
        self.sub.config(text=os.path.basename(path), fg=TEXT)
        self._thumbnail(path)
        self.on_file(path)

    def _thumbnail(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((THUMB_W - 20, THUMB_H - 20), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(THUMB_W // 2, THUMB_H // 2,
                                     anchor="center", image=self._photo)
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(THUMB_W // 2, THUMB_H // 2,
                                    text=str(e), fill="red",
                                    font=("Helvetica", 10), width=THUMB_W - 20)


# ── Main app ──────────────────────────────────────────────────────────────────
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scanner Line Fix")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._build()
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w,  h  = self.winfo_width(),       self.winfo_height()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _build(self):
        # Header
        tk.Label(self, text="Scanner Line Fix", bg=BG, fg=TEXT,
                 font=("Helvetica", 20, "bold")).pack(pady=(22, 4))
        tk.Label(self, text="Remove the blue scan-line artifact by stitching two shifted scans",
                 bg=BG, fg=SUBTEXT, font=("Helvetica", 11)).pack(pady=(0, 18))

        # Drop zones
        row = tk.Frame(self, bg=BG)
        row.pack(padx=24)
        self.z1 = DropZone(row, "Scan 1  —  base", self._on_img1)
        self.z1.pack(side="left", padx=(0, 12))
        self.z2 = DropZone(row, "Scan 2  —  shifted", self._on_img2)
        self.z2.pack(side="left")

        # Options
        opts = tk.Frame(self, bg=BG)
        opts.pack(padx=24, pady=16, fill="x")

        tk.Label(opts, text="DPI:", bg=BG, fg=TEXT,
                 font=("Helvetica", 12)).grid(row=0, column=0, sticky="w")

        cfg = pipeline.load_config()
        dpi_choices = [k.replace("dpi_", "") for k in cfg if k.startswith("dpi_")] or ["600"]
        self.dpi_var = tk.StringVar(value=dpi_choices[0])
        ttk.Combobox(opts, textvariable=self.dpi_var, values=dpi_choices,
                     width=7, state="readonly").grid(row=0, column=1, padx=(6, 28), sticky="w")

        tk.Label(opts, text="Output:", bg=BG, fg=TEXT,
                 font=("Helvetica", 12)).grid(row=0, column=2, sticky="w")
        self.out_var = tk.StringVar(value="stitched_result.png")
        tk.Entry(opts, textvariable=self.out_var, bg=PANEL, fg=TEXT,
                 insertbackground=TEXT, font=("Helvetica", 12),
                 relief="flat", width=32).grid(row=0, column=3, padx=(6, 4))
        tk.Button(opts, text="…", bg=PANEL, fg=TEXT, bd=0, cursor="hand2",
                  font=("Helvetica", 12),
                  command=self._pick_out).grid(row=0, column=4)

        # Run button
        self.run_btn = tk.Button(self, text="Run", bg=ACCENT, fg="white",
                                 font=("Helvetica", 15, "bold"), relief="flat",
                                 padx=48, pady=12, cursor="hand2",
                                 activebackground="#0071E3", activeforeground="white",
                                 command=self._run)
        self.run_btn.pack(pady=(0, 18))

        # Status + progress
        foot = tk.Frame(self, bg=BG)
        foot.pack(padx=24, pady=(0, 22), fill="x")
        self.status_var = tk.StringVar(value="Ready — drop two scans and click Run")
        tk.Label(foot, textvariable=self.status_var, bg=BG, fg=SUBTEXT,
                 font=("Helvetica", 11), anchor="w").pack(fill="x")
        self.bar = ttk.Progressbar(foot, mode="indeterminate")
        self.bar.pack(fill="x", pady=(6, 0))

    def _on_img1(self, path):
        base = os.path.splitext(os.path.basename(path))[0]
        default_out = os.path.join(os.path.dirname(path), f"{base}_stitched.png")
        self.out_var.set(default_out)

    def _on_img2(self, _):
        pass

    def _pick_out(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if p:
            self.out_var.set(p)

    def _run(self):
        if not self.z1.path or not self.z2.path:
            messagebox.showwarning("Missing images",
                                   "Drop both Scan 1 and Scan 2 before running.")
            return
        self.run_btn.config(state="disabled")
        self.bar.start(10)
        threading.Thread(target=self._pipeline,
                         args=(self.z1.path, self.z2.path,
                               self.out_var.get(), self.dpi_var.get()),
                         daemon=True).start()

    # ── Pipeline (runs in background thread) ─────────────────────────────────
    def _pipeline(self, img1_path, img2_path, output, dpi_str):
        try:
            info = pipeline.run_pipeline(img1_path, img2_path, output,
                                         dpi=dpi_str, log=self._log)
            self._set_status(f"Done  ✓  {os.path.basename(info['output'])}",
                             done=True, path=info["output"])
        except Exception as exc:
            self._set_status(f"Error: {exc}", error=True)

    def _log(self, msg):
        self._set_status(msg.strip())

    def _set_status(self, msg, done=False, error=False, path=None):
        def _update():
            self.status_var.set(msg)
            if done or error:
                self.bar.stop()
                self.bar["value"] = 0
                self.run_btn.config(state="normal")
            if done and path:
                if messagebox.askyesno("Done!", f"Saved to:\n{path}\n\nReveal in Finder?"):
                    subprocess.run(["open", "-R", path])
        self.after(0, _update)


if __name__ == "__main__":
    App().mainloop()
