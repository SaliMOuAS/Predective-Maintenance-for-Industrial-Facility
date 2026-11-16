import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sqlite3

# Stil für bessere Visualisierungen
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Ergebnisse-Ordner erstellen, falls nicht vorhanden
os.makedirs('results', exist_ok=True)

# Daten laden
try:
    df = pd.read_csv('data/predictive_maintenance.csv')
    print("✅ Daten erfolgreich geladen!")
except FileNotFoundError:
    print("❌ Datei nicht gefunden. Bitte überprüfe den Pfad:")
    print("   Erwarteter Pfad: 'data/predictive_maintenance.csv'")
    # Alternativ für den AI4I Datensatz:
    # df = pd.read_csv('ai4i2020.csv')  # Original-Dateiname des Datensatzes

# Erste Einsicht in die Daten
print("\n" + "="*50)
print("DATENÜBERSICHT")
print("="*50)
print(df.info())

print("\n" + "="*50)
print("ERSTE 5 ZEILEN")
print("="*50)
print(df.head())

print("\n" + "="*50)
print("STATISTISCHE ZUSAMMENFASSUNG")
print("="*50)
print(df.describe())

# Fehlende Werte checken
print("\n" + "="*50)
print("FEHLENDE WERTE")
print("="*50)
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percent
})
print(missing_df[missing_df['Missing Values'] > 0])

if missing_df['Missing Values'].sum() == 0:
    print("✅ Keine fehlenden Werte gefunden!")

# Explorative Datenanalyse (EDA)


# Ausfalltypen Analyse
print("\nAusfalltypen Verteilung:")
print(df['Failure Type'].value_counts())

# Einfaches Diagramm
plt.figure(figsize=(10, 5))

# Balkendiagramm
df['Failure Type'].value_counts().plot(kind='bar')
plt.title('Ausfalltypen Verteilung')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/failure_type_simple.png')
plt.show()

# 2. Korrelationsanalyse - Optimiert für Predictive Maintenance
print("\n" + "="*50)
print("KORRELATIONSANALYSE")
print("="*50)

# Wichtige numerische Sensordaten auswählen
sensor_columns = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Korrelationsmatrix berechnen
corr_matrix = df[sensor_columns].corr()
print("Korrelationsmatrix der Sensordaten:")
print(corr_matrix.round(3))

# Einfache Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Korrelation zwischen Sensordaten')
plt.tight_layout()
plt.savefig('results/correlation_simple.png')
plt.show()

# Starke Korrelationen finden und anzeigen
print("\nStarke Korrelationen (|r| > 0.5):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > 0.5:
            print(f"  {corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_value:.3f}")

# 3. Boxplots für Temperaturen bei Ausfall vs. kein Ausfall
print("\n" + "="*50)
print("VERGLEICH: AUSFALL VS. KEIN AUSFALL")
print("="*50)

# Binäre Variable für Ausfall erstellen
df['is_failure'] = df['Failure Type'] != 'No Failure'

# Mögliche Temperatur-Spalten finden
temperature_columns = [col for col in df.columns if 'temperature' in col.lower() or 'temp' in col.lower()]
print(f"Gefundene Temperatur-Spalten: {temperature_columns}")

if temperature_columns:
    fig, axes = plt.subplots(1, len(temperature_columns), figsize=(15, 5))
    if len(temperature_columns) == 1:
        axes = [axes]
    
    for i, temp_col in enumerate(temperature_columns):
        sns.boxplot(data=df, x='is_failure', y=temp_col, ax=axes[i])
        axes[i].set_title(f'{temp_col} nach Ausfallstatus')
        axes[i].set_xlabel('Ausfall aufgetreten')
        axes[i].set_ylabel(temp_col)
        axes[i].set_xticklabels(['Nein', 'Ja'])
    
    plt.tight_layout()
    plt.savefig('results/temperature_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. Zusätzliche Analysen

# Tool Wear Verteilung nach Ausfalltyp
if 'Tool wear' in df.columns or 'Tool wear [min]' in df.columns:
    tool_wear_col = 'Tool wear [min]' if 'Tool wear [min]' in df.columns else 'Tool wear'
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Failure Type', y=tool_wear_col)
    plt.title('Tool Wear Verteilung nach Ausfalltyp', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/tool_wear_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. Zeitliche Entwicklung (falls Zeitstempel vorhanden)
date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
if date_columns:
    print(f"\nZeitliche Spalten gefunden: {date_columns}")
    # Hier könntest du Zeitreihenanalyse durchführen

# 6. Zusammenfassung der wichtigsten Erkenntnisse
print("\n" + "="*50)
print("ZUSAMMENFASSUNG DER EDA")
print("="*50)
print(f"Gesamtdatensatz: {len(df)} Zeilen, {len(df.columns)} Spalten")
print(f"Ausfallrate: {(df['is_failure'].sum() / len(df) * 100):.2f}%")
print(f"Anzahl verschiedener Ausfalltypen: {df['Failure Type'].nunique()}")

if 'Type' in df.columns:
    print(f"\nMaschinentypen:")
    for machine_type, count in df['Type'].value_counts().items():
        failure_rate = (df[df['Type'] == machine_type]['is_failure'].sum() / count * 100)
        print(f"  {machine_type}: {count} Maschinen ({failure_rate:.1f}% Ausfallrate)")







