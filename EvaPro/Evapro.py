import os
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import mdr_data_processing as mdr   # eigenes Modul

SETTINGS_FILE = "settings.json"


class FolderApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("EvaPro - Datenverarbeitung")
        self.geometry("1200x700")

        # Hauptlayout
        self.grid_columnconfigure(0, weight=1, uniform="group1")
        self.grid_columnconfigure(1, weight=3, uniform="group1")
        self.grid_rowconfigure(0, weight=1)

        # --- Notebook (Tabs) links ---
        self.sidebar_notebook = ttk.Notebook(self)
        self.sidebar_notebook.grid(row=0, column=0, sticky="nswe")

        # Tabs
        self.mdr_frame = tk.Frame(self.sidebar_notebook, bg="#f0f0f0")
        self.sidebar_notebook.add(self.mdr_frame, text="MDR")

        self.tensile_frame = tk.Frame(self.sidebar_notebook, bg="#f0f0f0")
        self.sidebar_notebook.add(self.tensile_frame, text="Tensile Test")

        # Tab-Inhalte aufbauen
        self.setup_tab(self.mdr_frame, "mdr", add_process_button=True)
        self.setup_tab(self.tensile_frame, "tensile")

        # --- Hauptbereich rechts ---
        self.main_frame = tk.Frame(self, bg="white")
        self.main_frame.grid(row=0, column=1, sticky="nswe")

        # Hauptbereich Layout: oben Platz für Inhalte, unten Log
        self.main_frame.grid_rowconfigure(0, weight=3)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Log-Ausgabe
        log_frame = tk.Frame(self.main_frame)
        log_frame.grid(row=1, column=0, sticky="nsew")

        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled", bg="#1e1e1e", fg="white")
        self.log_text.pack(side="left", fill="both", expand=True)

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")

        # gespeicherte Einstellungen laden
        self.settings = self.load_settings()

        # Ordnerstruktur aus gespeicherten Pfaden wiederherstellen
        for key, tree in [("mdr", self.mdr_tree), ("tensile", self.tensile_tree)]:
            folder_path = self.settings.get(key)
            if folder_path and os.path.isdir(folder_path):
                self.insert_root_with_children(tree, folder_path)

    # ---------------- Tab aufbauen ----------------
    def setup_tab(self, parent, key, add_process_button=False):
        parent.grid_rowconfigure(3, weight=1)

        # Ordner wählen
        button = tk.Button(parent, text="Ordner wählen", command=lambda: self.choose_folder(key))
        button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Suchleiste mit 🔍 und ✕
        search_frame = tk.Frame(parent)
        search_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        search_frame.grid_columnconfigure(0, weight=1)

        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var)
        search_entry.grid(row=0, column=0, sticky="ew")

        search_btn = tk.Button(search_frame, text="🔍", command=lambda k=key: self.filter_tree(k))
        search_btn.grid(row=0, column=1, padx=2)

        clear_btn = tk.Button(search_frame, text="✕", command=lambda: self.clear_search(key))
        clear_btn.grid(row=0, column=2, padx=2)

        # Treeview
        tree = ttk.Treeview(parent)
        tree.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        tree_scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=2, column=1, sticky="ns")

        tree.bind("<<TreeviewOpen>>", lambda e, t=tree: self.on_open_folder(e, t))

        # Referenzen speichern
        if key == "mdr":
            self.mdr_tree = tree
            self.mdr_search_var = search_var
        elif key == "tensile":
            self.tensile_tree = tree
            self.tensile_search_var = search_var

        # Live-Filter während der Eingabe
        search_var.trace("w", lambda *args, k=key: self.filter_tree(k))

        # Optional: Daten bearbeiten Button im MDR-Tab
        if add_process_button:
            process_btn = tk.Button(parent, text="Daten bearbeiten", command=self.process_data)
            process_btn.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    # ---------------- Ordner wählen ----------------
    def choose_folder(self, key):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.settings[key] = folder_path
            self.save_settings()

            tree = self.mdr_tree if key == "mdr" else self.tensile_tree
            tree.delete(*tree.get_children())
            self.insert_root_with_children(tree, folder_path)

    # ---------------- Ordnerstruktur laden ----------------
    def insert_root_with_children(self, tree, folder_path):
        root_node = tree.insert("", "end", text=folder_path, values=[folder_path])
        try:
            for item in sorted(os.listdir(folder_path)):
                fullpath = os.path.join(folder_path, item)
                self.insert_node(tree, root_node, item, fullpath)
        except PermissionError:
            pass
        tree.item(root_node, open=True)

    def insert_node(self, tree, parent, text, abspath):
        node = tree.insert(parent, "end", text=text, values=[abspath])
        if os.path.isdir(abspath):
            tree.insert(node, "end")  # Dummy

    def on_open_folder(self, event, tree):
        node = tree.focus()
        abspath = tree.item(node, "values")[0]

        children = tree.get_children(node)
        if len(children) == 1 and tree.item(children[0], "values") == "":
            tree.delete(children[0])
            try:
                for item in sorted(os.listdir(abspath)):
                    fullpath = os.path.join(abspath, item)
                    self.insert_node(tree, node, item, fullpath)
            except PermissionError:
                pass

    # ---------------- Filter / Suche ----------------
    def filter_tree(self, key):
        search_text = (self.mdr_search_var.get() if key == "mdr" else self.tensile_search_var.get()).lower()
        tree = self.mdr_tree if key == "mdr" else self.tensile_tree

        # alles zurücksetzen
        for item in tree.get_children(""):
            self._filter_recursive(tree, item, search_text)

    def _filter_recursive(self, tree, node, search_text):
        text = tree.item(node, "text").lower()
        match = search_text in text if search_text else True

        child_matches = False
        for child in tree.get_children(node):
            if self._filter_recursive(tree, child, search_text):
                child_matches = True

        if match or child_matches:
            tree.item(node, open=bool(search_text))
            return True
        else:
            tree.detach(node)
            return False

    def clear_search(self, key):
        if key == "mdr":
            self.mdr_search_var.set("")
        else:
            self.tensile_search_var.set("")
        self.filter_tree(key)

    # ---------------- Datenverarbeitung ----------------
    def process_data(self):
        try:
            excel_file = filedialog.askopenfilename(
                title="Excel-Datei auswählen", filetypes=[("Excel files", "*.xlsx")]
            )
            if not excel_file:
                return
            df_excel = mdr.load_excel_sheet(excel_file, 'Tabelle2')

            folder_path = self.settings.get("mdr")
            if folder_path:
                versuche_path = filedialog.askdirectory(title="Zielordner für MDR auswählen")
                if not versuche_path:
                    return

                # Dateien verschieben (nur neue)
                mdr.move_files_to_destination_folder(folder_path, 'txt', df_excel, versuche_path, log_func=self.log)

                # CSVs nur berechnen, falls nicht vorhanden
                mdr.load_txt_files_into_dataframe(versuche_path, log_func=self.log)

            messagebox.showinfo("Fertig", "MDR-Datenverarbeitung abgeschlossen!")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    # ---------------- Settings ----------------
    def save_settings(self):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.settings, f)

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        return {"mdr": None, "tensile": None}

    # ---------------- Logging ----------------
    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()


if __name__ == "__main__":
    app = FolderApp()
    app.mainloop()
