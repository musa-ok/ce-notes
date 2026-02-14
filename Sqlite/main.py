from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional,Union
import sqlite3

class Urun(BaseModel):
    isim: str
    fiyat: float
    stok: int
    kategori: Union[str,None] = None
    stoktavar: Optional[bool] = None

def init_db():
    conn = sqlite3.connect("veritabanı.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS urunler(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        isim TEXT,
        fiyat INTEGER,
        stok INTEGER,
        kategori TEXT,
        stoktavar BOOLEAN
    )
    """)
    conn.commit()
    conn.close()

app = FastAPI(on_startup=[init_db])

@app.get("/")
def read_root():
    return [1,2,3,4,5]

@app.get("/soz/{kisi}")
def soz(kisi: str):
    sozler = {
        "ataturk": "Yurtta sulh, cihanda sulh.",
        "mevlana": "Ne olursan ol yine gel.",
        "albert": "Hayal gücü bilgiden daha önemlidir."
    }
    cikti = sozler.get(kisi.lower())

    return {"output": cikti}
@app.post("/urun/")
def urun(urun:Urun):
    urun_dict = urun.dict()
    urun.stok = max(0,urun.stok)
    urun_dict.update({"stok": urun.stok})
    urun_dict.update({"stoktavar": urun.stok > 0})
    return {"output":urun_dict}
@app.post("/urun/ekle")
def urun_ekle(urun: Urun):
    conn=sqlite3.connect("veritabanı.db")
    c = conn.cursor()
    c.execute("INSERT INTO urunler(isim,fiyat,stok,kategori,stoktavar) VALUES(?,?,?,?,?)",(urun.isim,urun.fiyat,urun.stok,urun.kategori,urun.stoktavar))
    conn.commit()
    conn.close()
    return {"outout": "ürün başarıyla eklendi."}