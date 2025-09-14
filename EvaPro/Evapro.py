import os
import json
import tkinter as tk
from tkinter import filedialog, ttk

SETTINGS_FILE = "settings.json"


class FolderApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Ordner Navigation mit Tabs pro Bereich")
        self.geometry("1000x600")

        # Hauptlayout: links Navigation, rechts Hauptfläche
        self.grid_columnconfigure(0, weight=1, uniform="group1")
        self.grid_columnconfigure(1, weight=3, uniform="group1")
        self.grid_rowconfigure(0, weight=1)

        # --- Notebook (Tabs) in der Navigationsleiste ---
        self.sidebar_notebook = ttk.Notebook(self)
        self.sidebar_notebook.grid(row=0, column=0, sticky="nswe")

        # Tab MDR
        self.mdr_frame = tk.Frame(self.sidebar_notebook, bg="#f0f0f0")
        self.sidebar_notebook.add(self.mdr_frame, text="MDR")

        # Tab Tensile Test
        self.tensile_frame = tk.Frame(self.sidebar_notebook, bg="#f0f0f0")
        self.sidebar_notebook.add(self.tensile_frame, text="Tensile Test")

        # Setup Inhalte für Tabs
        self.setup_tab(self.mdr_frame, "mdr")
        self.setup_tab(self.tensile_frame, "tensile")

        # --- Hauptbereich rechts (noch leer) ---
        self.main_frame = tk.Frame(self, bg="white")
        self.main_frame.grid(row=0, column=1, sticky="nswe")

        # Lade gespeicherte Einstellungen
        self.settings = self.load_settings()

        # Ordnerstrukturen wiederherstellen
        for key, tree in [("mdr", self.mdr_tree), ("tensile", self.tensile_tree)]:
            folder_path = self.settings.get(key)
            if folder_path and os.path.isdir(folder_path):
                self.insert_node(tree, "", folder_path, folder_path)

    def setup_tab(self, parent, key):
        """Erzeugt UI für einen Tab (Button + Treeview)"""
        parent.grid_rowconfigure(1, weight=1)

        # Button
        button = tk.Button(
            parent,
            text="Ordner wählen",
            command=lambda: self.choose_folder(key),
        )
        button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Treeview
        tree = ttk.Treeview(parent)
        tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Scrollbar
        tree_scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=1, column=1, sticky="ns")

        # Event für Aufklappen
        tree.bind("<<TreeviewOpen>>", lambda e, t=tree: self.on_open_folder(e, t))

        # Speichere Referenz
        if key == "mdr":
            self.mdr_tree = tree
        elif key == "tensile":
            self.tensile_tree = tree

    def choose_folder(self, key):
        """Ordner für einen Tab auswählen"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.settings[key] = folder_path
            self.save_settings()

            # richtigen Tree holen
            tree = self.mdr_tree if key == "mdr" else self.tensile_tree

            # Tree leeren und neuen Pfad einfügen
            tree.delete(*tree.get_children())
            self.insert_node(tree, "", folder_path, folder_path)

    def insert_node(self, tree, parent, text, abspath):
        """Einen Knoten (Ordner oder Datei) einfügen"""
        node = tree.insert(parent, "end", text=text, values=[abspath])
        if os.path.isdir(abspath):
            # Dummy-Kind, damit der Ordner aufklappbar aussieht
            tree.insert(node, "end")

    def on_open_folder(self, event, tree):
        """Wenn ein Ordner aufgeklappt wird, Unterordner laden"""
        node = tree.focus()
        abspath = tree.item(node, "values")[0]

        # Dummy-Eintrag löschen
        children = tree.get_children(node)
        if len(children) == 1 and tree.item(children[0], "values") == "":
            tree.delete(children[0])

            try:
                for item in sorted(os.listdir(abspath)):
                    fullpath = os.path.join(abspath, item)
                    self.insert_node(tree, node, item, fullpath)
            except PermissionError:
                pass  # Ordner ohne Zugriffsrechte überspringen

    def save_settings(self):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.settings, f)

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        return {"mdr": None, "tensile": None}


if __name__ == "__main__":
    app = FolderApp()
    app.mainloop()
