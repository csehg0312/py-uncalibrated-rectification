# SIFT Sztereó Feldolgozási Benchmark

Ez a tárhely egy Python-alapú benchmarkot biztosít a SIFT-alapú sztereó feldolgozás teljesítményének értékeléséhez.

## Telepítés

1. **A tárhely klónozása:**

 ```bash
 git clone https://github.com/your-username/sift-stereo-processing-benchmark.git
 cd sift-stereo-processing-benchmark
 ```
   
### A Poetry telepítése:

```bash
pip install poetry
```

### A függőségek telepítése:

```bash
poetry install
```
### Egy Poetry shell indítása:

```bash
poetry shell
```

## Futtatás:

### Az app.py fájlhoz:
```bash
python app.py
```

### A stereoprocessing_sad_benchmark.py fájlhoz:
```bash
python stereoprocessing_sad_benchmark.py
```

### Az uncalibrated_rectify_orb.py fájlhoz:

```bash
python uncalibrated_rectify_orb.py
```

### Az uncalibrated_rectify_sift.py fájlhoz:
```bash
python uncalibrated_rectify_sift.py
```

### Kilépése a virtuális környezetből:
```bash
exit
```

## Használat

Minden Python fájl tartalmazza a saját implementációját a sztereó feldolgozásnak SIFT jellemzők segítségével. Különböznek a következő szempontokban:

app.py: Alapértelmezett OpenCV jellemzőillesztés és sztereó feldolgozás használatával.
stereoprocessing_sad_benchmark.py: Egyedi SAD illesztő algoritmus implementációja a hatékonyabb jellemzőillesztéshez.
uncalibrated_rectify_orb.py: ORB jellemződetektálás használatával a SIFT helyett.
uncalibrated_rectify_sift.py: Az alapértelmezett SIFT jellemződetektálást használja és belefoglalja az illesztett kulcspontok vizualizálását is.
Minden fájl tartalmaz teljesítményfigyelést és jelentést is a végrehajtási idő, a CPU-használat és a memóriahasználat nyomon követéséhez.

## Megjegyzések

Meg kell adnia a bal és jobb képek elérési útját minden Python fájlban.
A visualize_matches() függvény a stereoprocessing_sad_benchmark.py fájlban lehetővé teszi az illesztett kulcspontok vizualizálását.

## Közreműködés

A közreműködések szívesen fogadottak! Szabadon elágazhat a tárhelyről, végezhet változtatásokat és beküldhet pull requesteket.

