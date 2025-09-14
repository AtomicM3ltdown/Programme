import os
import json
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import mdr_data_processing as mdr  # importiere das neue Modul

SETTINGS_FILE = "settings.json"

class FolderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ordner Navigation & Datenverarbeitung")
        self.geometry("1200x600")
        self.grid_columnconfigure(0, weight=1, uniform="group1")
        self.grid_columnconfigure(1, weight=2, uniform="group1")
        self.grid_rowconfigure(0, weight=1)

        # Notebook Tabs
        self.sidebar_notebook = ttk.Notebook(self)
        self.sidebar_notebook.grid(row=0, column=0, sticky="nswe")

        self.mdr_frame = tk.Frame(self.sidebar_notebook, bg="#f0f0f0")
        self.sidebar_notebook.add(self.mdr_frame, text="MDR")

        self.tensile_frame = tk.Frame(self.sidebar_notebook, bg="#f0f0f0")
        self.sidebar_notebook.add(self.tensile_frame, text="Tensile Test")

        self.setup_tab(self.mdr_frame, "mdr")
        self.setup_tab(self.tensile_frame, "tensile")

        # Rechts: Log-Fenster und Button
        self.right_frame = tk.Frame(self, bg="white")
        self.right_frame.grid(row=0, column=1, sticky="nswe")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(self.right_frame, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nswe", padx=5, pady=5)

        self.process_button = tk.Button(self.right_frame, text="Daten verarbeiten", command=self.process_data)
        self.process_button.grid(row=1, column=0, padx=20, pady=10)

        self.settings = self.load_settings()
        self.current_paths = {"mdr": None, "tensile": None}

        # Ordnerstrukturen wiederherstellen
        for key, tree in [("mdr", self.mdr_tree), ("tensile", self.tensile_tree)]:
            folder_path = self.settings.get(key)
            if folder_path and os.path.isdir(folder_path):
                self.current_paths[key] = folder_path
                self.insert_root_with_children(tree, folder_path)

    # ---------------- GUI Tab Setup ----------------
    def setup_tab(self, parent, key):
        parent.grid_rowconfigure(2, weight=1)
        button = tk.Button(parent, text="Ordner wählen", command=lambda: self.choose_folder(key))
        button.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        search_var = tk.StringVar()
        search_entry = tk.Entry(parent, textvariable=search_var)
        search_entry.grid(row=1, column=0, padx=10, pady=(0,5), sticky="ew")
        search_var.trace_add("write", lambda *a, k=key, v=search_var: self.filter_tree(k, v.get()))

        tree = ttk.Treeview(parent)
        tree.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        tree_scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=2, column=1, sticky="ns")
        tree.bind("<<TreeviewOpen>>", lambda e, t=tree: self.on_open_folder(e, t))

        if key == "mdr":
            self.mdr_tree = tree
            self.mdr_search = search_var
        else:
            self.tensile_tree = tree
            self.tensile_search = search_var

    # ---------------- GUI Funktionen ----------------
    def choose_folder(self, key):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.settings[key] = folder_path
            self.current_paths[key] = folder_path
            self.save_settings()
            tree = self.mdr_tree if key == "mdr" else self.tensile_tree
            tree.delete(*tree.get_children())
            self.insert_root_with_children(tree, folder_path)

    def insert_root_with_children(self, tree, folder_path):
        root_node = tree.insert("", "end", text=folder_path, values=[folder_path])
        try:
            for item in sorted(os.listdir(folder_path)):
                fullpath = os.path.join(folder_path, item)
                node = tree.insert(root_node, "end", text=item, values=[fullpath])
                if os.path.isdir(fullpath):
                    tree.insert(node, "end")  # Dummy
        except PermissionError:
            pass
        tree.item(root_node, open=True)

    def on_open_folder(self, event, tree):
        node = tree.focus()
        abspath = tree.item(node, "values")[0]
        children = tree.get_children(node)
        if len(children) == 1 and tree.item(children[0], "values") == "":
            tree.delete(children[0])
            try:
                for item in sorted(os.listdir(abspath)):
                    fullpath = os.path.join(abspath, item)
                    n = tree.insert(node, "end", text=item, values=[fullpath])
                    if os.path.isdir(fullpath):
                        tree.insert(n, "end")
            except PermissionError:
                pass

    def filter_tree(self, key, query):
        tree = self.mdr_tree if key == "mdr" else self.tensile_tree
        folder_path = self.current_paths.get(key)
        if not folder_path:
            return
        tree.delete(*tree.get_children())
        root_node = tree.insert("", "end", text=folder_path, values=[folder_path])
        if not query.strip():
            self.insert_root_with_children(tree, folder_path)
            return

        def search_and_insert(parent_node, current_path):
            try:
                for item in sorted(os.listdir(current_path)):
                    fullpath = os.path.join(current_path, item)
                    if query.lower() in item.lower():
                        node = tree.insert(parent_node, "end", text=item, values=[fullpath])
                        if os.path.isdir(fullpath):
                            tree.insert(node, "end")
                    elif os.path.isdir(fullpath):
                        child_node = tree.insert(parent_node, "end", text=item, values=[fullpath])
                        search_and_insert(child_node, fullpath)
                        if not tree.get_children(child_node):
                            tree.delete(child_node)
            except PermissionError:
                pass
        search_and_insert(root_node, folder_path)
        tree.item(root_node, open=True)

    # ---------------- Settings ----------------
    def save_settings(self):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.settings, f)

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        return {"mdr": None, "tensile": None}

    # ---------------- Log Funktion ----------------
    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.yview(tk.END)
        self.log_text.configure(state="disabled")

    # ---------------- Datenverarbeitung ----------------
    def process_data(self):
        try:
            excel_file = filedialog.askopenfilename(title="Excel-Datei auswählen", filetypes=[("Excel files", "*.xlsx")])
            if not excel_file:
                return
            df_excel = mdr.load_excel_sheet(excel_file, 'Tabelle2')

            for key in ["mdr", "tensile"]:
                folder_path = self.current_paths.get(key)
                if folder_path:
                    versuche_path = filedialog.askdirectory(title=f"Zielordner für {key} auswählen")
                    if not versuche_path:
                        continue
                    mdr.move_files_to_destination_folder(folder_path, 'txt', df_excel, versuche_path, log_func=self.log)
                    mdr.load_txt_files_into_dataframe(versuche_path, log_func=self.log)

            messagebox.showinfo("Fertig", "Datenverarbeitung abgeschlossen!")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))


if __name__ == "__main__":
    app = FolderApp()
    app.mainloop()
