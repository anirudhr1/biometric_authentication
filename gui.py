import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import subprocess
import os
import fingerprint_auth

# === GUI Functions ===

def run_face_login():
    subprocess.run(["python", "login.py"])

def run_fingerprint_login():
    file_path = filedialog.askopenfilename(
        title="Select Fingerprint Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        result = fingerprint_auth.match_fingerprint(file_path)
        if result:
            messagebox.showinfo("Login Result", f"✅ Fingerprint Matched: {result}")
        else:
            messagebox.showerror("Login Result", "❌ No Fingerprint Match Found.")

def register_new_face():
    username = simpledialog.askstring("Register Face", "Enter new username:")
    if not username:
        return

    # Run capture.py with username as argument
    subprocess.run(["python", "capture.py", username])

    # Then retrain
    subprocess.run(["python", "train.py"])

    messagebox.showinfo("Registration", f"✅ Face registered for {username}.")

# === GUI Setup ===

root = tk.Tk()
root.title("Biometric Authentication")
root.geometry("400x350")
root.resizable(False, False)

title = tk.Label(root, text="Biometric Authentication", font=("Arial", 18, "bold"))
title.pack(pady=20)

btn_register = tk.Button(root, text="🆕 Register New Face", font=("Arial", 14), width=25, command=register_new_face)
btn_register.pack(pady=10)

btn_face = tk.Button(root, text="🔵 Face Login", font=("Arial", 14), width=25, command=run_face_login)
btn_face.pack(pady=10)

btn_finger = tk.Button(root, text="🟢 Fingerprint Login", font=("Arial", 14), width=25, command=run_fingerprint_login)
btn_finger.pack(pady=10)

btn_exit = tk.Button(root, text="❌ Exit", font=("Arial", 12), width=15, command=root.destroy)
btn_exit.pack(pady=20)

root.mainloop()
