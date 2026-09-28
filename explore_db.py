import sqlite3

con = sqlite3.connect("data/tension.db")
cur = con.cursor()

tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
for t in tables:
    count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})")]
    print(f"{t} ({count} rows): {', '.join(cols)}")


for t in ("layouts", "product_sizes"):
    print(f"\n{t}:")
    try:
        for row in cur.execute(f"SELECT * FROM '{t}'"):
            print(row)
    except sqlite3.OperationalError as e:
        print("  not found:", e)   
        