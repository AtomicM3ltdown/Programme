```mermaid

graph TD

A[Start] --> B[Load Excel sheet]
B --> C{Iterate over files}
C -->|Yes| D[Extract filename without extension]
D --> E[Search for match in Excel sheet]
E -->|Yes| F[Extract content of column]
F --> G[Define destination folder path]
G --> H[Create destination folder if it doesn't exist]
H --> I[Define source and destination file paths]
I --> J[Move file to destination folder]
J --> C
C -->|No| K[End iteration]
K --> L[Define function remove_quotes]
L --> M[Define counter x = 0]
M --> N[Define main folder path]
N --> O[Define Calculation DataFrame]
O --> P{Loop through subfolders}
P -->|Yes| Q[Get paths of all txt files]
Q --> R{Check if 3 txt files}
R -->|Yes| S[Load data from txt files into separate DataFrames]
S --> T[Calculate mean and deviation]
T --> U[Save Calculation DataFrame to CSV file]
U --> P
R -->|No| V[End loop]
V --> W[End program]

